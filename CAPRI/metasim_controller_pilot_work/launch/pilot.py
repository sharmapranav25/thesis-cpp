"""Small, fixed-budget City controller trial with frozen appearance feedback."""
import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pilot_render as P

ROOT=Path(__file__).resolve().parents[1]
SPECIAL=ROOT.parent/'iteration2_specialists_work/run_20261003_01'
SIMPLE=ROOT.parent/'iter2_full_target_recreation_01'
DINO_CODE=Path('C:/Users/pranav/.cache/torch/hub/facebookresearch_dino_main')
DINO_WEIGHTS=Path('C:/Users/pranav/.cache/torch/hub/checkpoints/dino_deitsmall8_pretrain.pth')
SEED=20261004


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,default=default)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def status(out,stage,**values):
    value=dict(stage=stage,seconds=round(time.perf_counter()-START,1),**values)
    dump(out/'status.json',value)
    with (out/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,default=default)+'\n')
    print(json.dumps(value,default=default),flush=True)


class Features:
    def __init__(self):
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        sys.path.insert(0,str(DINO_CODE))
        import vision_transformer as V
        self.torch=torch
        self.model=V.vit_small(patch_size=8,num_classes=0)
        self.model.load_state_dict(torch.load(DINO_WEIGHTS,map_location='cpu',weights_only=True))
        self.model.requires_grad_(False);self.model.cuda().eval()
        self.mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
        self.std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)

    def __call__(self,images):
        vectors=[];stats=[];audit=[]
        for image in images:
            rgb=np.asarray(Image.fromarray(image).resize((128,32),Image.Resampling.BILINEAR),float)/255
            grey=rgb@np.array([.299,.587,.114]);dx=np.diff(grey,axis=1);dy=np.diff(grey,axis=0)
            stat=np.r_[rgb.mean((0,1)),rgb.std((0,1)),np.quantile(grey,[.05,.25,.5,.75,.95]),
                (rgb.max(2)-rgb.min(2)).mean(),(grey>.95).mean(),(grey<.35).mean(),
                abs(dx).mean(),abs(dy).mean(),grey.mean(1)[::2],grey.mean(0)[::8]]
            stats.append(stat)
            small=np.asarray(Image.fromarray((grey*255).astype(np.uint8)).resize((32,8),Image.Resampling.BOX),float).ravel()/255
            gy,gx=np.gradient(grey);mag=np.sqrt(gx**2+gy**2);angles=(np.arctan2(gy,gx)+np.pi)%(np.pi)
            hist=[]
            for y in range(4):
                for x in range(8):
                    area=(slice(y*8,(y+1)*8),slice(x*16,(x+1)*16))
                    hist.extend(np.histogram(angles[area],bins=8,range=(0,np.pi),weights=mag[area])[0])
            hist=np.asarray(hist)/(mag.sum()+1e-8)
            audit.append(np.r_[small,hist])
        torch=self.torch
        with torch.inference_mode():
            for start in range(0,len(images),32):
                pixels=np.stack([np.asarray(Image.fromarray(image).resize((320,64),Image.Resampling.BILINEAR)) for image in images[start:start+32]])
                x=torch.from_numpy(pixels.transpose(0,3,1,2).copy()).cuda().float()/255
                with torch.autocast('cuda',dtype=torch.bfloat16):v=self.model((x-self.mean)/self.std)
                v=v.float();v=v/v.norm(dim=1,keepdim=True).clamp(min=1e-8)
                vectors.append(v.cpu().numpy())
        return dict(dino=np.concatenate(vectors),stats=np.asarray(stats),audit=np.asarray(audit))


def distance(a,b):return np.maximum(0,(a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T)
def kernel(a,b,bandwidth):
    d=distance(a,b)
    return sum(np.exp(-d/(2*bandwidth*scale)) for scale in (.5,1,2))/3
def bandwidth(x):
    values=distance(x,x);return max(1e-5,float(np.median(values[np.triu_indices(len(x),1)])))


class Feedback:
    def __init__(self,fit):
        self.mean=fit['stats'].mean(0);self.std=np.maximum(fit['stats'].std(0),.04)
        self.audit_mean=fit['audit'].mean(0);self.audit_std=np.maximum(fit['audit'].std(0),.02)
        self.widths=dict(dino=bandwidth(fit['dino']),stats=bandwidth(self.normalize(fit)['stats']))
        directions=np.random.default_rng(SEED+8).normal(size=(512,128));directions/=np.linalg.norm(directions,axis=0)
        self.directions=directions

    def normalize(self,f):return dict(dino=f['dino'],stats=(f['stats']-self.mean)/self.std)
    def components(self,a,b):
        a,b=self.normalize(a),self.normalize(b)
        return {key:float(kernel(a[key],a[key],self.widths[key]).mean()+kernel(b[key],b[key],self.widths[key]).mean()-2*kernel(a[key],b[key],self.widths[key]).mean()) for key in self.widths}
    def audit(self,a,b):
        aa=((a['audit']-self.audit_mean)/self.audit_std)@self.directions
        bb=((b['audit']-self.audit_mean)/self.audit_std)@self.directions
        return float(np.mean(abs(np.quantile(aa,np.linspace(.01,.99,128),axis=0)-np.quantile(bb,np.linspace(.01,.99,128),axis=0))))
    def score(self,f,real,rows):
        components=self.components(f,real)
        usable=float(np.mean([row['usable'] for row in rows]));strict=float(np.mean([row['strict'] for row in rows]))
        appearance=.6*components['dino']+.4*components['stats']
        return dict(**components,appearance=appearance,usable=usable,strict=strict,
                    objective=appearance+.8*(1-usable)+.15*(1-strict))
    def bootstrap(self,before,after,real,n=200):
        b,a,r=map(self.normalize,(before,after,real));rng=np.random.default_rng(SEED+37)
        kernels={key:(kernel(b[key],b[key],self.widths[key]),kernel(a[key],a[key],self.widths[key]),
                      kernel(r[key],r[key],self.widths[key]),kernel(b[key],r[key],self.widths[key]),
                      kernel(a[key],r[key],self.widths[key])) for key in self.widths}
        delta=[]
        for _ in range(n):
            si=rng.integers(len(b['dino']),size=len(b['dino']));ri=rng.integers(len(r['dino']),size=len(r['dino']))
            change=0
            for key,weight in [('dino',.6),('stats',.4)]:
                kb,ka,kr,kbr,kar=kernels[key]
                change+=weight*(ka[np.ix_(si,si)].mean()-kb[np.ix_(si,si)].mean()-2*(kar[np.ix_(si,ri)].mean()-kbr[np.ix_(si,ri)].mean()))
            delta.append(change)
        return dict(after_minus_before_95pct=list(np.quantile(delta,[.025,.975])),resamples=n)


def real_images(rows):
    result=[]
    for row in rows:
        with Image.open(row['source']) as image:result.append(np.asarray(image.convert('RGB')))
    return result


def profile(out,rows,name):
    from feature_bank.iter2.controller import build_profiles,FONT_KEYS
    directory=out/name;directory.mkdir()
    clean=[]
    for index,row in enumerate(rows):
        clean.append(dict(family='City',index=index,source=row['source'],length=row['length'],
            font={key:row['font'][key] for key in FONT_KEYS},effects=row['effects'],text_features=row['text_features']))
    with (directory/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for row in clean:stream.write(json.dumps(row)+'\n')
    dump(directory/'manifest.json',dict(total=len(clean),counts=dict(City=len(clean)),real_answers_used=False))
    build_profiles(directory/'predictions.jsonl',directory/'manifest.json',directory/'profiles')
    return read(directory/'profiles/City.json')


def recipes(pool,profile,count,seed):
    proto=profile['joint_distribution'];weights=np.asarray([r['count'] for r in proto],float);weights/=weights.sum()
    rng=np.random.default_rng(seed)
    items=[(proto[int(rng.choice(len(proto),p=weights))]['description'],int(rng.integers(1,2**30)),i) for i in range(count)]
    return list(pool.map(P.prepare,items))


def generated(pool,recipes,vector,fonts):return list(pool.map(P.render,[(recipe,vector,fonts) for recipe in recipes]))
def features(model,rows):return model([row['image'] for row in rows])
def save_group(out,name,rows):
    folder=out/name;folder.mkdir(exist_ok=True);(folder/'images').mkdir(exist_ok=True)
    labels=[]
    for i,row in enumerate(rows):
        path=folder/'images'/f'{i:04d}.png';Image.fromarray(row['image']).save(path)
        labels.append({key:value for key,value in row.items() if key!='image'}|dict(image=str(path.relative_to(out))))
    with (folder/'labels.jsonl').open('w',encoding='utf-8') as stream:
        for row in labels:stream.write(json.dumps(row,default=default)+'\n')
    return labels


def gallery(out,real,groups):
    sheet=Image.new('RGB',(1440,42+12*120),'white');draw=ImageDraw.Draw(sheet)
    names=['Held-out City references','Simple initial','Specialist initial','Specialist adjusted']
    for j,name in enumerate(names):draw.text((j*360+8,12),name,fill='black')
    for i in range(12):
        for j,image in enumerate([real[i]]+[group[i]['image'] for group in groups]):
            im=Image.fromarray(image);im.thumbnail((348,90));x=j*360+6;y=42+i*120
            sheet.paste(im,(x,y+(90-im.height)//2))
            label='reference (unpaired)' if j==0 else f"{i}: {groups[j-1][i].get('text','render failure')}"
            draw.text((x,y+94),label,fill='black')
    sheet.save(out/'comparison.jpg',quality=94)


def main():
    global START
    START=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source_hashes={str(path):sha(path) for path in [SPECIAL/'final/hybrid.pt',P.PROJECT/'runs/iter2_baseline_15_20261003_04/baseline_training/epoch_15/ema.pt',DINO_WEIGHTS]}
    protocol=dict(family='City',seed=SEED,target_splits=dict(fit=256,selection=128,test=512),
        rounds=8,population=8,fit_renders_per_candidate=24,selection_candidates=3,selection_renders=48,test_renders_per_group=96,
        controls=dict(zip(P.NAMES,P.BOUNDS.tolist())),neutral=dict(zip(P.NAMES,P.NEUTRAL.tolist())),
        method='Meta-Sim-inspired black-box distribution search, not full Meta-Sim or GAN training',
        source_hashes=source_hashes,neural_updates=0,feedback='0.6 DINO multi-kernel MMD + 0.4 appearance-statistics MMD + .8 unusable + .15 strict-readability failures',
        selection='Lowest selection objective among three best fit controllers passing usability/readability within 5 points of neutral; neutral always eligible.',
        test_policy='Test subset never scored until selected_controls.json is saved. No tuning after test.',
        dtype='Frozen DINO ViT-S/8, 320x64 RGB resize, ImageNet normalization, CUDA bf16; no downloads',
        fonts='Eight own training-bank candidates chosen from starting synthetic recipes; no target font files',
        renderer_extension='Pilot-only foreground/background colour hooks and optional band-slope multiplier; all labels measured after rendering',
        real_answers_used=False,solver_training_started=False)
    dump(out/'protocol.json',protocol)
    allrows=[json.loads(line) for line in (SPECIAL/'real_inference/predictions.jsonl').read_text().splitlines()]
    city=[row for row in allrows if row['family']=='City'];assert len(city)==2000
    unique={};duplicates=0
    for row in city:
        with Image.open(row['source']) as image:pixel_hash=hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()
        if pixel_hash in unique:duplicates+=1
        else:unique[pixel_hash]=dict(row,pixel_sha256=pixel_hash)
    rows=list(unique.values());order=np.random.default_rng(SEED).permutation(len(rows));assert len(rows)>=896
    split=dict(fit=[rows[i] for i in order[:256]],selection=[rows[i] for i in order[256:384]],test=[rows[i] for i in order[384:896]])
    dump(out/'target_split.json',dict(duplicates_excluded=duplicates,total_unique=len(rows),splits={s:[dict(source=r['source'],pixel_sha256=r['pixel_sha256']) for r in values] for s,values in split.items()}))
    specialist=profile(out,split['fit'],'specialist_initial_profile')
    simple_rows={row['source'].lower():row for row in map(json.loads,(SIMPLE/'predictions.jsonl').read_text().splitlines()) if row['family']=='City'}
    simple=profile(out,[simple_rows[row['source'].lower()] for row in split['fit']],'simple_initial_profile')
    status(out,'extracting_target_fit_and_selection',unique_targets=len(rows),duplicates_excluded=duplicates)
    model=Features();real_fit=model(real_images(split['fit']));real_selection=model(real_images(split['selection']));feedback=Feedback(real_fit)
    np.savez(out/'feedback_scaling.npz',stats_mean=feedback.mean,stats_std=feedback.std,audit_mean=feedback.audit_mean,audit_std=feedback.audit_std,directions=feedback.directions)
    dump(out/'kernel_bandwidths.json',feedback.widths)
    with ProcessPoolExecutor(max_workers=6,initializer=P.install) as pool:
        status(out,'preparing_starting_recipes')
        common=recipes(pool,specialist,24,SEED+10)
        font_counts=Counter(recipe['text_spec']['fonts'][0] for recipe in common)
        fonts=[key for key,_ in font_counts.most_common(8)]
        for font in P.bank().pool('train',category='Sans Serif',caseless=True):
            if len(fonts)==8:break
            if font['key'] not in fonts:fonts.append(font['key'])
        assert len(fonts)==8;dump(out/'font_candidates.json',[P.bank().by_key[key] for key in fonts])
        dump(out/'fit_recipes.json',common)
        neutral_rows=generated(pool,common,P.NEUTRAL,fonts);neutral_features=features(model,neutral_rows)
        initial_score=feedback.score(neutral_features,real_fit,neutral_rows)
        # Wrapped neutral controls must reproduce an unmodified renderer exactly.
        original=P.plain(common[0]);assert np.array_equal(original['image'],neutral_rows[0]['image'])
        assert original['text']['text']==neutral_rows[0]['text']
        dump(out/'neutral_replay_check.json',dict(passed=True,pixel_max_abs_difference=0))
        status(out,'search_start',initial=initial_score)
        rng=np.random.default_rng(SEED+20);mean=P.NEUTRAL.copy();span=P.BOUNDS[:,1]-P.BOUNDS[:,0];std=.35*span
        history=[];evaluated=0
        for iteration in range(8):
            for member in range(8):
                if iteration==0 and member==0:vector=P.NEUTRAL.copy()
                elif iteration==0 or rng.random()<.25:vector=rng.uniform(P.BOUNDS[:,0],P.BOUNDS[:,1])
                else:vector=np.clip(rng.normal(mean,std),P.BOUNDS[:,0],P.BOUNDS[:,1])
                rendered=generated(pool,common,vector,fonts);f=features(model,rendered);score=feedback.score(f,real_fit,rendered)
                history.append(dict(round=iteration+1,member=member,vector=vector.tolist(),**score));evaluated+=24
                status(out,'searching',round=iteration+1,member=member+1,candidate_renders=evaluated,score=score['objective'],best=min(row['objective'] for row in history))
                dump(out/'search_history.json',history)
            elite=sorted(history,key=lambda row:row['objective'])[:4];values=np.array([row['vector'] for row in elite])
            mean=.5*mean+.5*values.mean(0);std=np.maximum(.12*span,.5*std+.5*values.std(0))
        best=[];seen=set()
        for candidate in sorted(history,key=lambda row:row['objective']):
            identity=tuple(np.round(candidate['vector'],5))
            if identity not in seen:best.append(candidate);seen.add(identity)
            if len(best)==3:break
        status(out,'selecting_on_separate_target_subset')
        selection_recipes=recipes(pool,specialist,48,SEED+30)
        neutral_selection=generated(pool,selection_recipes,P.NEUTRAL,fonts)
        baseline=feedback.score(features(model,neutral_selection),real_selection,neutral_selection)
        choices=[dict(vector=P.NEUTRAL.tolist(),score=baseline,eligible=True,source='neutral')]
        for index,candidate in enumerate(best):
            rendered=generated(pool,selection_recipes,candidate['vector'],fonts);score=feedback.score(features(model,rendered),real_selection,rendered)
            eligible=score['usable']>=baseline['usable']-.05 and score['strict']>=baseline['strict']-.05
            choices.append(dict(vector=candidate['vector'],score=score,eligible=eligible,source=f'fit_rank_{index+1}'))
        chosen=min((choice for choice in choices if choice['eligible']),key=lambda choice:choice['score']['objective'])
        dump(out/'selection.json',choices);dump(out/'selected_controls.json',dict(choice=chosen,controls=dict(zip(P.NAMES,chosen['vector'])),frozen_before_test=True))
        status(out,'final_held_out_evaluation',chosen=chosen['source'])
        real_test_images=real_images(split['test']);real_test=model(real_test_images)
        test_recipes=recipes(pool,specialist,96,SEED+40);simple_recipes=recipes(pool,simple,96,SEED+40)
        groups=dict(simple_initial=generated(pool,simple_recipes,P.NEUTRAL,fonts),
                    specialist_initial=generated(pool,test_recipes,P.NEUTRAL,fonts),
                    specialist_adjusted=generated(pool,test_recipes,chosen['vector'],fonts))
        group_features={name:features(model,value) for name,value in groups.items()};results={}
        for name,rendered in groups.items():
            values=feedback.score(group_features[name],real_test,rendered)
            values.update(audit_sliced_wasserstein=feedback.audit(group_features[name],real_test),
                images=len(rendered),unique_pixels=len({row.get('pixel_sha256') for row in rendered}),
                unique_strings=len({row.get('text') for row in rendered}),complete_text=sum(row['text_complete'] for row in rendered),
                mean_pairwise_dino_distance=float(np.mean(distance(group_features[name]['dino'],group_features[name]['dino']))),
                font_categories=dict(Counter(row.get('font',{}).get('dominant_category') for row in rendered)),
                mean_weight=float(np.mean([row['font']['median_weight'] for row in rendered if 'font' in row])))
            save_group(out,name,rendered);results[name]=values
        assert all(a['text']==b['text'] for a,b in zip(groups['specialist_initial'],groups['specialist_adjusted']))
        confidence=feedback.bootstrap(group_features['specialist_initial'],group_features['specialist_adjusted'],real_test)
        gallery(out,real_test_images,[groups[name] for name in groups])
        for name,value in group_features.items():np.savez(out/f'{name}_features.npz',**value)
        np.savez(out/'target_test_features.npz',**real_test)
        dump(out/'results.json',dict(groups=results,paired_bootstrap=confidence,selected=chosen,
            candidate_renders=evaluated,neural_updates=0,source_checkpoints_unchanged=all(sha(Path(path))==value for path,value in source_hashes.items()),
            target_splits={key:len(value) for key,value in split.items()},text_pairs_verified=True,
            limitations=['City only','Appearance metrics are proxies, not font accuracy or CAPTCHA recognition accuracy',
                         'Global 15-setting search; no learned neural controller or full Meta-Sim reproduction',
                         'Pilot-only colour and band-slope hooks; renderer/source files unchanged',
                         'Initialization reused cached frozen model predictions; no real answers accessed']))
        assert all(sha(Path(path))==value for path,value in source_hashes.items())
        status(out,'complete',groups=results,paired_bootstrap=confidence,neural_updates=0)


if __name__=='__main__':main()
