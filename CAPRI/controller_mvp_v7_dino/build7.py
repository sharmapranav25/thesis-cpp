"""DINO controller v2: complete text controls, pixel structure and replicated validation."""
from setup7 import *
from perception7 import Comparator,Objective7,WEIGHTS
from dino5 import Objective as DinoObjective,palettes
from build5 import key,select,eligible
from run import load_images,save_rows
import render7 as R
import numpy as np
import copy,time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from PIL import Image

RANDOM_SEED=71019
FIELDS=dict(charset=['digits','upper','lower','upper_digits','lower_digits','mixed','letters'],
    layout=['line','curve','two_layer'],length=[3,4,5,6,7,8],size_absolute=[.45,.6,.75,.9],
    width_absolute=[.65,.85,1.,1.2],spacing_absolute=[-.2,0.,.15,.35],slant_absolute=[-.3,0.,.3],
    size_jitter_absolute=[0.,.15,.3],rotation_absolute=[0.,10.,25.,40.],jitter_absolute=[0.,.1,.2],
    weight_absolute=[0.,.5,1.],bold_absolute=[0,1],curve_absolute=[.05,.15])

def fresh(recipes,count,offset):
    result=[]
    for i in range(count):
        r=copy.deepcopy(recipes[i%len(recipes)]);r['seed']=730003+(offset+i)*7919
        ts=r['text_spec'];ts['text']=R.C.R.T.make_text(np.random.default_rng(r['seed']+101),len(ts['text']),r['description']['content'])
        result.append(r)
    return result

def main():
    assert U.read(ROOT/'checks.json')['passed'];model=Comparator();fonts=U.read(WORK/'learned_controller_20261004/fonts.json');start=time.time()
    codehash={p.name:U.sha(p) for p in [ROOT/'build7.py',ROOT/'render7.py',ROOT/'perception7.py',ROOT/'setup7.py']}
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),name='DINO controller v2 / MVP7',code_sha256=codehash,
        model='Same frozen official DINO ViT-S/8. No VLM, OCR, new learned segmenter or target labels.',weights_sha256=U.sha(WEIGHTS_PATH),
        region='Multiscale RGB local contrast times a soft DINO-attention prior, with tiny-component and long-line downweighting. A heuristic region estimate, not verified target text segmentation.',
        objective_weights=WEIGHTS,contrast_guard='Penalty below 65% of mean target-fit contrast in inferred regions.',
        controls='Absolute charset, layout, size, width, spacing, weight, bold, slant, size variation, curve, rotation, jitter; existing palettes/fonts/effects; generic inner glyph outlines and independently rotated square frames.',
        search='Hybrid bank and fresh random bank seed 71019. Eight fresh-string fit renders per candidate. Accumulating coordinate search, 48 generic font proposals, all 38 effects at levels 0/2/5, then 32 within-bank recombinations. Prior DINO profile also evaluated as a hybrid candidate.',
        validation='Six diverse candidates per bank plus prior DINO profile. Two disjoint target-selection batches of 32, each with 24 fresh synthetic recipe draws. Rank mean objective + 0.25 standard deviation; complete=1, pooled usable>=.9 and each batch>=.875.',
        archive='Up to 8 eligible distinct bank/control pairs within 1.5x best robust validation score; never use diagnostic test to choose.',
        test='Previously exposed v4 diagnostic pixels, first64; 48 fresh synthetic samples per arm. Hybrid neutral, frozen old DINO profile and new selected bank/profile. All policies frozen before diagnostics.',
        caveats='Larger budget and added controls/objective differ from v1; not a backbone-only or equal-budget ablation. Region heuristics may confuse background with glyphs. Framed synthetic masks include frames. No solver training.',
        random_prior='No hybrid descriptions enter the random bank. Only median canvas size comes from fit pixels. Same generic 96 fonts and renderer prior as MVP6.')
    if (ROOT/'protocol.json').exists():assert U.read(ROOT/'protocol.json')['code_sha256']==codehash
    else:U.dump(ROOT/'protocol.json',protocol)
    summary=U.read(ROOT/'summary.json') if (ROOT/'summary.json').exists() else {}
    neutral=R.expand(R.C.R.NEUTRAL)
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:
        for family in FAMILIES:
            if family in summary:continue
            dest=ROOT/family;dest.mkdir(exist_ok=True)
            split=U.read(V3/(family+'_run_01')/'target_splits.json');refs=U.read(V4/'evaluation_pixels'/f'{family}.json')['rows'][:64]
            groups=[{r['pixel_sha256'] for r in x} for x in [split['fit'],split['selection'],refs]]
            assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
            U.dump(dest/'target_splits.json',dict(fit=split['fit'],selection=split['selection'],test=refs))
            realfit=load_images(split['fit']);size=np.median([[im.shape[1],im.shape[0]] for im in realfit],axis=0).astype(int).tolist()
            target=model.extract(realfit);validation=[model.extract(load_images(split['selection'][j:j+32])) for j in [0,32]]
            colors=palettes(realfit,{**target,'focus':target['region_mask']});U.dump(dest/'palettes.json',colors)
            def render(p,recipes):return list(pool.map(R.safe_render,[(r,p) for r in recipes]))
            def features(rows):return model.extract([r.get('image',np.full((size[1],size[0],3),255,np.uint8)) for r in rows])
            calibration=random_recipes(71099,16,size,fonts,400);calfeatures=features(render(neutral,calibration))
            objective=Objective7(target,calfeatures);oldobjective=DinoObjective(target,calfeatures)
            U.dump(dest/'objective.json',dict(std={k:v.tolist() for k,v in objective.std.items()},scale=objective.scale,contrast_floor=objective.contrast_floor))
            previous=V1/('run_03' if family=='City' else family+'_run_01')
            hybridfit=U.read(previous/'fit_recipes.json');hybridsel=U.read(previous/'selection_recipes.json');hybridtest=U.read(V3/(family+'_run_01')/'test_recipes.json')
            banks={'hybrid':dict(fit=fresh(hybridfit,8,0),validation=[fresh(hybridsel,24,100),fresh(hybridsel[12:]+hybridsel[:12],24,200)],test=fresh(hybridtest,48,300))}
            banks['random']=dict(fit=fresh(random_recipes(RANDOM_SEED,8,size,fonts),8,0),
                validation=[fresh(random_recipes(RANDOM_SEED,24,size,fonts,100+j*100),24,100+j*100) for j in range(2)],
                test=fresh(random_recipes(RANDOM_SEED,48,size,fonts,300),48,300))
            for bank in banks.values():
                for recipes in [bank['fit'],*bank['validation'],bank['test']]:
                    for r in recipes:r['size']=size
            U.dump(dest/'recipe_banks.json',banks);oldprofile=R.expand(U.read(V5/family/'selected.json')['dino'])
            allrecords=[];shortlist=[]
            for name,bank in banks.items():
                records=[];seen=set();rng=np.random.default_rng(71515)
                def evaluate(p,label):
                    p=R.expand(p)
                    if key(p) in seen:return
                    seen.add(key(p));rows=render(p,bank['fit']);f=features(rows)
                    rec=dict(index=len(allrecords),bank=name,phase=label,controls=p,score=objective.score(f,target,rows),
                        vector=objective.vector(f).tolist(),render_errors=sum('error' in r for r in rows))
                    records.append(rec);allrecords.append(rec)
                evaluate(neutral,'neutral')
                if name=='hybrid':evaluate(oldprofile,'prior_dino')
                def leader():return min([r for r in records if eligible(r)] or records,key=lambda r:r['score']['objective'])['controls']
                def group(label,options):
                    for p in options:evaluate(p,label)
                    U.dump(ROOT/'status.json',dict(family=family,bank=name,phase=label,candidates=len(allrecords),seconds=time.time()-start))
                    print(family,name,label,len(records),flush=True)
                clean={n:0 for n in R.C.R.R.SAMPLED};p=leader()
                options=[{**p,'background':colors['background'][0]},{**p,'background_pool':colors['background']}]
                for mode in ['single','random','alternating']:
                    for effects in [p['effects'],clean]:options.append({**p,'background':colors['background'][0],'background_pool':None,'text_palette':colors['text'],'effect_palette':colors['effect'],'color_mode':mode,'effects':effects})
                group('palette',options)
                subset=np.random.default_rng(71616).choice(len(fonts),48,replace=False);p=leader()
                group('fonts',[{**p,'font':fonts[i]['key'],'font_pool':None} for i in subset])
                for field,values in FIELDS.items():
                    p=leader();group(field,[{**p,field:v} for v in values])
                p=leader();group('decoration',[{**p,'glyph_outline':outline,'frame_width':frame,'frame_padding':pad} for outline in [0,1,2] for frame,pad in [(0,2),(1,2),(1,4)]])
                # Accumulate multiple independently useful effects instead of keeping only one change per stage.
                for effect in R.C.R.R.SAMPLED:
                    p=leader();group('effect_'+effect,[{**p,'effects':{**p['effects'],effect:level}} for level in [0,2,5]])
                parents=select(records,12,quality=False) if any(eligible(r) for r in records) else sorted(records,key=lambda r:r['score']['objective'])[:12]
                options=[]
                for _ in range(32):
                    a,b=rng.choice(len(parents),2);p=copy.deepcopy(parents[a]['controls']);other=parents[b]['controls']
                    for field in [*FIELDS,'font','glyph_outline','frame_width','text_palette','effects']:
                        if rng.random()<.3:p[field]=copy.deepcopy(other[field])
                    options.append(p)
                group('joint',options)
                chosen=select(records,6,quality=False) if any(eligible(r) for r in records) else sorted(records,key=lambda r:r['score']['objective'])[:6]
                shortlist.extend(chosen)
            shortlist.extend(r for r in allrecords if r['phase']=='prior_dino')
            shortlist=list({(r['bank'],key(r['controls'])):r for r in shortlist}.values())
            U.dump(dest/'history.json',[{k:v for k,v in r.items() if k!='vector'} for r in allrecords])
            np.savez_compressed(dest/'candidate_vectors.npz',vectors=np.asarray([r['vector'] for r in allrecords],np.float32))
            validated=[]
            for candidate in shortlist:
                scores=[];vectors=[]
                for j in range(2):
                    rows=render(candidate['controls'],banks[candidate['bank']]['validation'][j]);f=features(rows)
                    scores.append(objective.score(f,validation[j],rows));vectors.append(objective.vector(f))
                values=[s['objective'] for s in scores];passed=all(s['complete']==1 and s['usable']>=.875 for s in scores) and np.mean([s['usable'] for s in scores])>=.9
                validated.append(dict(bank=candidate['bank'],controls=candidate['controls'],index=candidate['index'],scores=scores,
                    robust_score=float(np.mean(values)+.25*np.std(values)),eligible=bool(passed),vector=np.mean(vectors,axis=0).tolist()))
            valid=[r for r in validated if r['eligible']];ordered=sorted(valid or validated,key=lambda r:r['robust_score']);best=ordered[0]
            archive=[r for r in ordered if r['eligible'] and r['robust_score']<=1.5*max(.05,best['robust_score'])][:8]
            U.dump(dest/'selection.json',[{k:v for k,v in r.items() if k!='vector'} for r in validated])
            frozen={k:v for k,v in best.items() if k!='vector'};frozen['frozen_utc']=datetime.now(timezone.utc).isoformat()
            U.dump(dest/'selected.json',frozen);U.dump(dest/'archive.json',[{k:v for k,v in r.items() if k!='vector'} for r in archive]);digest=U.sha(dest/'selected.json')
            ft=model.extract(load_images(refs));np.savez_compressed(dest/'target_regions.npz',attention=ft['focus'],region=ft['region_mask'])
            scores={};policies={'hybrid':('hybrid',neutral),'dino_v1':('hybrid',oldprofile),'dino_v2':(best['bank'],best['controls'])}
            for arm,(bankname,p) in policies.items():
                rows=render(p,banks[bankname]['test']);f=features(rows)
                score=objective.score(f,ft,rows);score['dino_v1_metric']=oldobjective.score(f,ft,rows)
                truth=[]
                for i,row in enumerate(rows):
                    if 'error' in row:row.update(image=np.full((size[1],size[0],3),255,np.uint8),mask=np.zeros((size[1],size[0]),np.uint8),seed=banks[bankname]['test'][i]['seed'],text='RENDER FAILURE')
                    truth.append(np.asarray(Image.fromarray(row['mask']).resize((256,64),Image.Resampling.NEAREST))>127)
                truth=np.stack(truth)
                for field in ['focus','region_mask']:
                    mask=f[field]>.5;score[field+'_iou']=float(np.mean((mask&truth).sum((1,2))/np.maximum(1,(mask|truth).sum((1,2)))))
                save_rows(dest,arm,rows);scores[arm]=score
            assert U.sha(dest/'selected.json')==digest
            result=dict(family=family,candidates=len(allrecords),fit_renders_per_candidate=8,validation_candidates=len(validated),validation_renders_per_candidate=48,
                selected_bank=best['bank'],search_pass=bool(valid),archive_count=len(archive),scores=scores,generated_per_arm=48,
                selected_sha256=digest,archive_sha256=U.sha(dest/'archive.json'),target_answers_used=False,vlm=False,visual_review='pending')
            U.dump(dest/'results.json',result);summary[family]=result;U.dump(ROOT/'summary.json',summary)
            print(family,'DONE',best['bank'],{a:round(s['objective'],3) for a,s in scores.items()},flush=True)
    assert codehash=={name:U.sha(ROOT/name) for name in codehash}
    U.dump(ROOT/'status.json',dict(phase='complete',families=len(summary),seconds=time.time()-start))

if __name__=='__main__':main()
