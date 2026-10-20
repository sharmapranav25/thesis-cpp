"""All eight current sources: Phase1 directly to frozen font inference."""
import sys,json,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');sys.path.insert(0,str(ROOT))
from feature_bank.core import read,write,new_run,load,pixel_hash,digest
from style_pipeline.font_match import FontMatcher,infer
from style_pipeline.measure import extract_style
from style_pipeline.render import render

def jsonl(p):return [json.loads(x) for x in p.read_text().splitlines()]
out=new_run(Path(__file__).resolve().parent/'dino_recipe_01');torch.set_num_threads(4)
run=ROOT/'runs/style_phase3_font_05';fonts=read(run/'fonts.json');checkpoint=run/'font_matcher.pt';before=digest(checkpoint)
model=FontMatcher(len(fonts)).cuda();model.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict']);model.eval()
families={};variants={};old=read(ROOT/'runs/style_phase1_05/results.json')['families']
for name in old:
    folder=ROOT/'runs'/('style_frozen_'+name+('_02' if name=='darkbay' else '_01'))
    protocol=read(folder/'protocol.json')
    for f,p in protocol['models'].items():assert digest(p)==protocol['model_sha256_before'][f]
    lookup={r['pixel_sha256']:r for r in jsonl(folder/'predictions.jsonl')}
    rows=read(ROOT/'runs/style_phase1_05'/f'{name}_measurements.json')['fit'][:16]
    families[name]=[lookup[r['pixel_sha256']] for r in rows];variants[name]=old[name]['selected']['variant']
lookup={r['pixel_sha256']:r for r in jsonl(ROOT/'runs/style_frozen_inference_01/predictions.jsonl')}
families['roncuevas']=[lookup[r['pixel_sha256']] for r in read(ROOT/'runs/style_phase1_unseen_01/split.json')['fit'][:16]]
for name,folder in [('pymlex','style_phase3_new_family_01'),('sloth','style_phase3_new_family_02')]:families[name]=[r for r in read(ROOT/'runs'/folder/'predictions.json') if r['partition']=='train'][:16]

import time,shutil,cv2
from render_color_locked import render as locked_render
start=time.time();cv2.setNumThreads(1)
shutil.copy2(__file__,out/'script_snapshot.py');shutil.copy2(Path(__file__).parent/'render_color_locked.py',out/'render_color_locked.py')
cache=Path.home()/'.cache/torch/hub';repo=cache/'facebookresearch_dino_main';weights=cache/'checkpoints/dino_deitsmall8_pretrain.pth'
sys.path.insert(0,str(repo));import vision_transformer as vit
dino=vit.vit_small(patch_size=8).cuda().eval();dino.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
for parameter in dino.parameters():parameter.requires_grad=False
dino_hash=digest(weights)
prior=read(ROOT/'runs/style_direct_bank_01/results.json')
candidates=[dict(name='baseline',key=None,gain=1.)]
for key in ('text_height','spacing_gain','glyph_width_gain','line_count_gain','line_width_gain','dot_count_gain','curve_count_gain'):
    for gain in ((.8,1.2) if key in ('text_height','spacing_gain','glyph_width_gain') else (.5,1.5)):
        candidates.append(dict(name=key+'_'+str(gain),key=key,gain=gain))
write(out/'protocol.json',dict(sequence='Phase1 CV + frozen features -> Phase2 existing font selection -> optional frozen DINO parameter search',length='fixed5; count model excluded',candidates=candidates,selection='fit-only one-control-at-a-time; baseline eligible; no changes after development scoring',fit='first4 references; next4 diagnostic; pymlex first2/next2',scope='previously exposed development; no unseen claim',objective='cosine distance family mean CLS + .5*cosine mean attention-weighted patch descriptor + .25*L1 mean attention spatial occupancy',ablation='CLS-only selector from same candidate pool',colors='palettes/gradient unchanged; exact baseline glyph RGB and per-kind operation RGB reused; changed geometry can alter pixel histogram',guard='all glyph visibility>=.55; same text/font/effect/count; reject invalid candidate',dino_sha256=dino_hash,dino_source_sha256=digest(repo/'vision_transformer.py'),training=False,automation='paused'))
for name in ('vision_transformer.py','utils.py','LICENSE'):shutil.copy2(repo/name,out/('dino_'+name))
def describe(images):
    outputs=[]
    for offset in range(0,len(images),32):
        x=torch.tensor(np.stack([np.asarray(im.convert('RGB').resize((224,72)),dtype='float32').transpose(2,0,1)/255 for im in images[offset:offset+32]]),device='cuda')
        x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        with torch.no_grad():
            tokens=dino.get_intermediate_layers(x,n=1)[0];att=dino.get_last_selfattention(x)[:,:,0,1:].mean(1);att=att/att.sum(1,keepdim=True)
            cls=torch.nn.functional.normalize(tokens[:,0],dim=1);weighted=torch.nn.functional.normalize((tokens[:,1:]*att[:,:,None]).sum(1),dim=1)
            grid=torch.nn.functional.adaptive_avg_pool2d(att.reshape(-1,1,9,28),(3,10)).flatten(1);grid=grid/grid.sum(1,keepdim=True)
            outputs.append(torch.cat([cls,weighted,grid],1).cpu().numpy())
    return np.concatenate(outputs)
def cosine(a,b):return float(1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def score(a,b):
    a=a.mean(0);b=b.mean(0);c=cosine(a[:384],b[:384]);w=cosine(a[384:768],b[384:768]);s=float(abs(a[768:]-b[768:]).sum())
    return dict(cls=c,weighted=w,attention=s,total=c+.5*w+.25*s)
def cvdesc(im):
    a=np.asarray(im.convert('L').resize((224,72)));edge=cv2.Canny(a,50,130)/255
    return np.array([edge.mean(),a.std()/255,np.abs(np.diff(a.astype(float),axis=1)).mean()/255])
results={};failures=[];panels=[]
for family,rows in families.items():
    nfit=2 if len(rows)==4 else 4;rows=rows[:nfit+4];fitrows=rows[:nfit];devrows=rows[nfit:]
    assert set(r['pixel_sha256'] for r in fitrows).isdisjoint(r['pixel_sha256'] for r in devrows)
    font=next(f['path'] for f in fonts if f['family']==prior[family]['font']);effect=prior[family]['effect'];variant=variants.get(family,0)
    samples=[]
    for i,r in enumerate(rows):
        im=load(r['path']);assert pixel_hash(im)==r['pixel_sha256'];style=extract_style(im);style['classifier_features']={f:q['image_state'] for f,q in r['predictions'].items()};style['font_effect']=effect
        text=''.join(np.random.default_rng(280000+i).choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),5));seed=281000+i
        baseline,mask,recipe=render(style,text,font,seed,variant)
        style['_locked_operation_colors']={kind:[op['color'] for op in recipe['operations'] if op['kind']==kind] for kind in ('block','line','point','classifier_curve')};style['_locked_glyph_colors']=[g['color'] for g in recipe['glyphs']]
        replay,_,_=locked_render(style,text,font,seed,variant);assert np.array_equal(np.asarray(replay),np.asarray(baseline))
        samples.append(dict(real=im,style=style,text=text,seed=seed,base=baseline,recipe=recipe))
    realfit=describe([s['real'] for s in samples[:nfit]]);realdev=describe([s['real'] for s in samples[nfit:]])
    def candidate_images(candidate,subset):
        images=[];recipes=[]
        for s in subset:
            adjusted=copy.deepcopy(s['style']);key=candidate['key']
            if key:adjusted[key]=adjusted.get(key,1.)*candidate['gain']
            for k in ('background_palette','foreground_palette','gradient_left','gradient_right','gradient_strength'):assert adjusted.get(k)==s['style'].get(k)
            im,mask,recipe=locked_render(adjusted,s['text'],font,s['seed'],variant)
            assert recipe['text']==s['text'] and len(recipe['glyphs'])==5 and recipe['font']==font
            assert [g['color'] for g in recipe['glyphs']]==s['style']['_locked_glyph_colors']
            for op in recipe['operations']:assert op['color'] in s['style']['_locked_operation_colors'][op['kind']]
            images.append(im);recipes.append(recipe)
        return images,recipes
    trial=[]
    for c in candidates:
        try:
            ims,_=candidate_images(c,samples[:nfit]);d=describe(ims);trial.append(dict(candidate=c,score=score(d,realfit)))
        except ValueError as e:failures.append(dict(family=family,candidate=c['name'],partition='fit',error=str(e)))
    chosen=min(trial,key=lambda r:r['score']['total']);cls_chosen=min(trial,key=lambda r:r['score']['cls']);write(out/(family+'_frozen_selection.json'),dict(combined=chosen,cls_only=cls_chosen,trials=trial))
    before=[s['base'] for s in samples[nfit:]];after=[];recipes=[];fallbacks=[]
    for i,s in enumerate(samples[nfit:]):
        try:ims,rs=candidate_images(chosen['candidate'],[s]);after+=ims;recipes+=rs
        except ValueError as e:after.append(s['base']);recipes.append(s['recipe']);fallbacks.append(i);failures.append(dict(family=family,partition='dev',index=i,error=str(e)))
    cls_images=[]
    for s in samples[nfit:]:
        try:ims,_=candidate_images(cls_chosen['candidate'],[s]);cls_images+=ims
        except ValueError:cls_images.append(s['base'])
    bs=score(describe(before),realdev);ats=score(describe(after),realdev);ablation=score(describe(cls_images),realdev)
    targetcv=np.mean([cvdesc(s['real']) for s in samples[nfit:]],0);cvbefore=float(np.linalg.norm(np.mean([cvdesc(im) for im in before],0)-targetcv));cvafter=float(np.linalg.norm(np.mean([cvdesc(im) for im in after],0)-targetcv))
    panel=Image.new('RGB',(1000,430),'#f1f3f5');draw=ImageDraw.Draw(panel)
    for i,(s,b,a,recipe) in enumerate(zip(samples[nfit:],before,after,recipes)):
        for j,im in enumerate([s['real'],b,a]):
            thumb=im.copy();thumb.thumbnail((235,100));panel.paste(thumb,(10+250*i,30+140*j))
        a.save(out/f'{family}_{i:02d}_after.png');b.save(out/f'{family}_{i:02d}_before.png')
    for y,label in [(5,'Real development references'),(145,'Phase1 + Phase2 fonts; fixed length5'),(285,'Optional DINO adjustment: '+chosen['candidate']['name'])]:draw.text((10,y),family+' | '+label,fill='black')
    panel.save(out/(family+'_comparison.png'));panels.append(panel)
    results[family]=dict(selected=chosen['candidate'],cls_selected=cls_chosen['candidate'],before=bs,after=ats,cls_only=ablation,cv_before=cvbefore,cv_after=cvafter,fit_paths=[r['path'] for r in fitrows],development_paths=[r['path'] for r in devrows],fallbacks=fallbacks,recipes=recipes)
    write(out/'results.json',results);write(out/'failures.json',failures);print(family,chosen['candidate']['name'],round(bs['total'],4),round(ats['total'],4),flush=True)
for k in range(2):
    board=Image.new('RGB',(1000,1720),'white')
    for i,p in enumerate(panels[k*4:k*4+4]):board.paste(p,(0,430*i))
    board.save(out/f'bank_{k+1}.png')
assert digest(checkpoint)==read(ROOT/'runs/style_direct_bank_01/protocol.json')['font_model_sha256']
assert digest(weights)==dino_hash
write(out/'status.json',dict(state='complete',seconds=time.time()-start,training=False,count_model_used=False,promoted=False,families=8,final_renders=sum(len(r['recipes']) for r in results.values())))
print('COMPLETE',flush=True)
