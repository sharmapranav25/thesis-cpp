"""Quality-diversity archives from unlabeled target fit/selection pixels only."""
from setup4 import ROOT,V3,V1,U,FAMILIES
import copy,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import controls as C
from perception3 import Perception,Objective,WEIGHTS
from run import load_images

TEXT_KEYS=['size','width','spacing','rotation','jitter','weight','font','text_palette','color_mode','length','jitter_absolute','rotation_absolute','bold_extra','font_pool']

def key(p):return json.dumps(p,sort_keys=True)
def candidates(family):
    old=V3/(family+'_run_01');history=U.read(old/'history.json');best=U.read(old/'frozen_policies.json')['full']
    ranked=[]
    for r in history:
        if 'error' in r or 'controls' not in r:continue
        s=r.get('score') or r.get('scores',{}).get('typography')
        if s:ranked.append((s['objective'],C.expand(r['controls'])))
    unique={}
    for score,p in sorted(ranked,key=lambda x:x[0]):unique.setdefault(key(p),(score,p))
    ordered=list(unique.values());rng=np.random.default_rng(41013)
    pool=[C.expand(best),C.expand(C.R.NEUTRAL),*[p for _,p in ordered[:32]]]
    if len(ordered)>32:pool.extend(ordered[i][1] for i in rng.choice(np.arange(32,len(ordered)),min(16,len(ordered)-32),replace=False))
    # Recombine glyph appearance independently of background/interference.
    for _ in range(32):
        a,b=[pool[int(rng.integers(len(pool)))] for j in range(2)];p=copy.deepcopy(b)
        for field in TEXT_KEYS:p[field]=copy.deepcopy(a[field])
        glyphs=C.R.R.GLYPH_EFFECTS
        p['effects']={n:v for n,v in p['effects'].items() if n not in glyphs}
        p['effects'].update({n:v for n,v in a['effects'].items() if n in glyphs})
        if p['keep_only'] is not None:p['keep_only']=list(dict.fromkeys([n for n in p['keep_only'] if n not in glyphs]+[n for n in glyphs if p['effects'].get(n,0)>0]))
        pool.append(p)
    return list({key(p):p for p in pool}.values())

def mask_quality(rows,features):
    truth=np.stack([np.asarray(Image.fromarray(r['mask']).resize((256,64),Image.Resampling.NEAREST))>127 for r in rows]);pred=features['mask']>.5
    iou=(truth&pred).sum((1,2))/np.maximum(1,(truth|pred).sum((1,2)))
    coverage=pred.sum((1,2))/np.maximum(1,truth.sum((1,2)))
    return dict(iou=float(iou.mean()),coverage=float(np.median(coverage)))

def greedy(records,n=16):
    eligible=[r for r in records if r['eligible']]
    if len(eligible)<n:raise RuntimeError(f'Only {len(eligible)} trustworthy candidates; need {n}, no automatic relaxation')
    best=min(r['score']['objective'] for r in eligible)
    eligible=[r for r in eligible if r['score']['objective']<=2*max(best,.05)]
    if len(eligible)<n:raise RuntimeError('Too few target-like candidates under the fixed quality gate')
    vectors=np.asarray([r['vector'] for r in eligible]);vectors=(vectors-vectors.mean(0))/np.maximum(vectors.std(0),.1)
    chosen=[int(np.argmin([r['score']['objective'] for r in eligible]))]
    while len(chosen)<n:
        dist=np.min(np.mean((vectors[:,None]-vectors[chosen][None])**2,axis=2),axis=1)**.5
        merit=dist/(1+np.asarray([r['score']['objective'] for r in eligible])/max(best,.05));merit[chosen]=-1
        chosen.append(int(merit.argmax()))
    return [eligible[i] for i in chosen]

def main():
    ROOT.mkdir(exist_ok=True);out=ROOT/'collections';out.mkdir(exist_ok=False);model=Perception();start=time.time();summary={}
    U.dump(ROOT/'protocol.json',dict(vlm=False,target_answers_in_controller=False,archive_size=16,families=FAMILIES,
        screening='Complete=100%, usable>=90%, synthetic predicted-mask IoU>=.30, median predicted/true ink area .4..2.5; appearance objective <=2x best eligible. No target masks supplied.',
        diversity='Greedy farthest mean appearance descriptors, quality penalized, after fit-only screening and selection-pixel validation.',
        solver_plan='Three pooled-family arms, equal 12800 training images, 1600 common synthetic validation images, same strings and initialization, 20 fixed epochs, two training seeds. 20% shared generic clean anchors. Target labels only after all models frozen.',
        source_sha256={p.name:U.sha(p) for p in ROOT.glob('*.py')},mask_checkpoint=U.sha(V3/'mask_training/best.pt')))
    with ProcessPoolExecutor(max_workers=8,initializer=C.install) as pool:
        for family in FAMILIES:
            dest=out/family;dest.mkdir();split=U.read(V3/(family+'_run_01')/'target_splits.json')
            real=model.extract_pair(load_images(split['fit']))[1];sel=model.extract_pair(load_images(split['selection']))[1]
            previous=V1/('run_03' if family=='City' else family+'_run_01')
            fit_recipes=U.read(previous/'fit_recipes.json')[:12];sel_recipes=U.read(previous/'selection_recipes.json')[:24]
            options=candidates(family);U.dump(dest/'candidates.json',options)
            def evaluate(p,recipes):
                rows=list(pool.map(C.render,[(recipe,p) for recipe in recipes]));f=model.extract_pair([r['image'] for r in rows])[1];return rows,f
            rows,f=evaluate(options[0],fit_recipes);objective=Objective(real,f);records=[]
            def record(i,p,rows,f,target):
                score=objective.score(f,target,rows);mq=mask_quality(rows,f)
                eligible=score['complete']==1 and score['usable']>=.9 and mq['iou']>=.30 and .4<=mq['coverage']<=2.5
                vector=np.concatenate([f[k].mean(0)/objective.std[k]*WEIGHTS[k]**.5 for k in WEIGHTS])
                return dict(index=i,controls=p,score=score,mask_quality=mq,eligible=bool(eligible),vector=vector.tolist())
            for i,p in enumerate(options):
                try:
                    rows,f=evaluate(p,fit_recipes);records.append(record(i,p,rows,f,real))
                except Exception as e:records.append(dict(index=i,error=repr(e),eligible=False))
                if i%16==0:
                    U.dump(ROOT/'collection_status.json',dict(family=family,phase='fit',done=i+1,total=len(options),seconds=time.time()-start));U.dump(dest/'fit.json',records)
            fit_eligible=[r for r in records if r['eligible']]
            # Diversity shortlist gives selection a choice beyond merely the best appearance scores.
            shortlist=greedy(fit_eligible,min(32,len(fit_eligible)))
            validated=[]
            for r in shortlist:
                rows,f=evaluate(r['controls'],sel_recipes);validated.append(record(r['index'],r['controls'],rows,f,sel))
            chosen=greedy(validated,16);U.dump(dest/'selection.json',validated);U.dump(dest/'fit.json',records)
            U.dump(dest/'archive.json',[r['controls'] for r in chosen]);U.dump(dest/'single.json',chosen[0]['controls'])
            preview=Image.new('RGB',(960,680),'#f5f6f8');d=ImageDraw.Draw(preview);d.text((8,8),family+' — 16 selected schemes; fresh random synthetic text',fill='black')
            preview_rows=[]
            for i,r in enumerate(chosen):
                row=C.render((fit_recipes[0],r['controls'])) if False else list(pool.map(C.render,[(fit_recipes[0],r['controls'])]))[0]
                image=ImageOps.contain(Image.fromarray(row['image']),(228,118),Image.Resampling.NEAREST);x=(i%4)*240;y=30+(i//4)*160
                d.text((x+8,y),str(i+1),fill='black');preview.paste(image,(x+(240-image.width)//2,y+25));preview_rows.append(row)
            preview.save(dest/'preview.png');summary[family]=dict(candidates=len(options),fit_eligible=len(fit_eligible),selection_eligible=sum(r['eligible'] for r in validated),selected=16,archive_sha256=U.sha(dest/'archive.json'))
            U.dump(out/'summary.json',summary);U.dump(ROOT/'collection_status.json',dict(family=family,phase='complete',seconds=time.time()-start));print(family,summary[family],flush=True)
    U.dump(ROOT/'collection_status.json',dict(phase='complete',seconds=time.time()-start))

if __name__=='__main__':main()
