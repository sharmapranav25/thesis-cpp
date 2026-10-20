"""Whole-reference descriptive profiling and small feature-guided render batches."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import collections,json,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2,joblib,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,new_run,load,pixel_hash,digest
from feature_bank.prepare import DATASETS,resolve_image
from feature_bank.regions import regions
from feature_bank.descriptors import extract
from feature_bank.labeled_bricks import FEATURES,sample_operations,compose
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
MODELS=None;CUTS=None

def initialize(cuts):
    global MODELS,CUTS
    cv2.setNumThreads(1);CUTS=cuts
    MODELS={f:joblib.load(ROOT/'runs/feature_bank_exploratory_01'/f'{f}_LR.joblib') for f in FEATURES}

def analyze(item):
    dataset,path,expected,filehash=item
    image=load(path)
    if pixel_hash(image)!=expected or digest(path)!=filehash:raise ValueError('Reference image changed: '+str(path))
    rs=regions(*image.size,scales=((3,10),));xx=np.array([extract(image.crop(r['box']))[0] for r in rs])
    predictions={}
    for f,b in MODELS.items():
        raw=b['model'].decision_function(xx[:,:b['descriptor_length']])
        p=b['calibrator'].predict_proba(raw[:,None])[:,1];cut=CUTS[f]['thresholds']
        states=['uncertain' if cut is None else 'present' if v>=cut[1] else 'absent' if v<=cut[0] else 'uncertain' for v in p]
        predictions[f]=dict(probabilities=p.tolist(),states=states,
            image_state='present' if 'present' in states else 'absent' if all(s=='absent' for s in states) else 'uncertain')
    arr=np.asarray(image)
    return dict(dataset=dataset,path=str(path),pixel_sha256=expected,size=list(image.size),
        median_rgb=np.median(arr.reshape(-1,3),axis=0).tolist(),regions=rs,predictions=predictions)

def freeze_thresholds(out):
    run=ROOT/'runs/feature_bank_exploratory_01';meta=read(run/'descriptor_rows.json')
    lookup={r['id']:i for i,r in enumerate(meta) if r['partition']=='calibration'}
    labels={}
    with (ROOT/'runs/feature_bank_corpus_candidate_01/bricks.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['id'] in lookup:labels[r['id']]=r['labels']
    X=np.load(run/'descriptors.npy',mmap_mode='r');cuts={}
    for f in FEATURES:
        b=joblib.load(run/f'{f}_LR.joblib');ids=[i for id,i in lookup.items() if labels[id][f] in ('present','absent')]
        y=np.array([labels[meta[i]['id']][f]=='present' for i in ids]);raw=b['model'].decision_function(X[ids,:b['descriptor_length']])
        p=b['calibrator'].predict_proba(raw[:,None])[:,1];grid=np.unique(np.quantile(p,np.linspace(0,1,201)))
        lows=[];highs=[]
        for t in grid:
            lo=p<=t;hi=p>=t
            if lo.sum()>=30 and (1-y[lo]).mean()>=.9:lows.append((float(t),int(lo.sum()),float((1-y[lo]).mean())))
            if hi.sum()>=30 and y[hi].mean()>=.9:highs.append((float(t),int(hi.sum()),float(y[hi].mean())))
        pairs=[(lo,hi) for lo in lows for hi in highs if lo[0]<hi[0]]
        best=max(pairs,key=lambda z:(z[0][1]+z[1][1],z[1][0]-z[0][0]),default=None)
        cuts[f]=dict(thresholds=[best[0][0],best[1][0]] if best else None,
            negative_support=best[0][1] if best else 0,positive_support=best[1][1] if best else 0,
            calibration_npv=best[0][2] if best else None,calibration_precision=best[1][2] if best else None,
            original_thresholds=b['thresholds'],model_sha256=digest(run/f'{f}_LR.joblib'))
    write(out/'thresholds.json',cuts)
    print('Synthetic-only threshold repair:',json.dumps(cuts),flush=True)
    return cuts

def generate(out,summaries):
    sys.path.insert(0,str(PROJECT))
    from sourcefree_synth.renderer import default_parameters,render_once,QualityError
    fonts=read(PROJECT/'runs/feature_bank_pilot_01/font_selection.json')[:2]
    for f in fonts:
        if digest(f['path'])!=f['sha256']:raise ValueError('Font changed')
    recipes=[];failures=[]
    for name,summary in summaries.items():
        records=[json.loads(s) for s in (out/(name+'.jsonl')).read_text().splitlines()]
        canvas=Image.new('RGB',(1000,1000),'#edf1f5');draw=ImageDraw.Draw(canvas)
        draw.text((12,8),name+' | feature-guided examples, not verified style recreation',fill='black')
        folder=out/'generated'/name;folder.mkdir(parents=True)
        rng=np.random.default_rng(773+list(DATASETS).index(name))
        for i in range(32):
            source=records[int(rng.integers(len(records)))];w,h=source['size']
            bg=tuple(int(v) for v in source['median_rgb']);dark=np.mean(bg)<128;fg=(230,230,230) if dark else (25,25,25)
            # Only supported positive feature combinations guide rendering.
            enabled=[f for f in FEATURES if source['predictions'][f]['image_state']=='present']
            profile=dict(id='full-reference-descriptive',width=w,height=h,background=[list(bg)],foreground=[list(fg)])
            p=default_parameters(0);p.update(background='flat',dots=0,lines=0,rotation=0,jitter=0,outline=False,shadow=False,multicolor=False,wave_x=0,wave_y=0,blur=0,height=.55,width=.78)
            text=''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),5))
            recipe=dict(dataset=name,index=i,reference_hash=source['pixel_sha256'],enabled=enabled,
                unknown_or_absent_not_rendered=[f for f in FEATURES if f not in enabled],text=text,profile=profile,parameters=p,seed=5800+i,
                density_tilt='generic fixed sampling, not estimated from references',font=fonts[i%2])
            try:
                base,mask,glyphs,meta=render_once(text,p,profile,[fonts[i%2]],5800+i)
                ops=sample_operations(w,h,6800+i,dark);final=compose(base,ops,enabled)
                reference=compose(Image.new('RGB',(w,h),bg),ops,enabled)
                delta=np.max(np.abs(np.asarray(final).astype('int16')-np.asarray(reference).astype('int16')),axis=2)
                visibility=[float((delta[np.asarray(g)>100]>=12).mean()) for g in glyphs]
                accepted=min(visibility)>=.55
                recipe.update(operations=ops,metadata=meta,text_visibility=visibility,quality_pass=accepted)
                final.save(folder/f'{i:02d}.png')
                if not accepted:failures.append(dict(dataset=name,index=i,reason='text visibility below .55'))
                preview=final.copy();preview.thumbnail((230,90));x=10+(i%4)*250;y=35+(i//4)*120
                draw.text((x,y),str(i)+' '+('PASS' if accepted else 'REJECT'),fill='black');canvas.paste(preview,(x,y+18))
            except (QualityError,ValueError) as exc:
                recipe.update(quality_pass=False,error=str(exc));failures.append(dict(dataset=name,index=i,reason=str(exc)))
            recipes.append(recipe)
        canvas.save(out/(name+'_generated.png'))
    write(out/'generation_recipes.json',recipes);write(out/'generation_failures.json',failures)

def main():
    start=time.perf_counter();out=new_run(ROOT/'runs/feature_bank_full_reference_01')
    write(out/'protocol.json',dict(purpose='Whole-reference descriptive profiling and limited feature-guided generation, not accuracy evaluation',
        user_authorized_all_references=True,real_labels_accessed=False,reference_sets_no_longer_untouched=True,
        scope='all historical unique fit+validation images; existing exclusions preserved',
        aggregation='image positive if any positive brick; absent only if all absent; otherwise uncertain; this rule is not image-calibrated',
        density_tilt_estimation=False,unsupported_features=9,workers=4))
    for rel in ['scripts/profile_five_datasets.py','feature_bank/descriptors.py','feature_bank/views.py','feature_bank/regions.py']:
        p=out/'code_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
    cuts=freeze_thresholds(out);items=[];sources=[]
    for name in DATASETS:
        path=PROJECT/'runs'/('external_pixels_outlined_01' if name=='outlined_text_captcha' else 'pixel_statistics_01')/name/'split.json'
        split=read(path);members=split['fit']+split['validation'];seen=set()
        for r in members:
            if r['pixel_sha256'] in seen:raise ValueError('Duplicate reference')
            seen.add(r['pixel_sha256']);items.append((name,resolve_image(PROJECT,name,r),r['pixel_sha256'],r['file_sha256']))
        sources.append(dict(dataset=name,count=len(members),manifest=str(path),sha256=digest(path),exclusions=split['excluded_directories']))
    write(out/'sources.json',sources);summaries={}
    streams={name:(out/(name+'.jsonl')).open('w') for name in DATASETS}
    for name in DATASETS:summaries[name]=dict(images=0,features={f:dict(present=0,absent=0,uncertain=0) for f in FEATURES},combinations={})
    try:
        with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(cuts,)) as pool:
            for i,result in enumerate(pool.map(analyze,items,chunksize=4)):
                name=result['dataset'];streams[name].write(json.dumps(result)+'\n');s=summaries[name];s['images']+=1
                combo=[]
                for f in FEATURES:
                    state=result['predictions'][f]['image_state'];s['features'][f][state]+=1
                    if state=='present':combo.append(f)
                key='+'.join(combo) or 'no_confident_positive';s['combinations'][key]=s['combinations'].get(key,0)+1
                if (i+1)%100==0:
                    print(f'Profiled {i+1}/{len(items)} images',flush=True);write(out/'progress.json',dict(images=i+1,total=len(items),elapsed_seconds=time.perf_counter()-start))
    finally:
        for stream in streams.values():stream.close()
    write(out/'summaries.json',summaries);generate(out,summaries)
    write(out/'status.json',dict(state='COMPLETE_DESCRIPTIVE_ONLY',images=len(items),elapsed_seconds=time.perf_counter()-start,
        accuracy_measured=False,real_calibration_verified=False,style_recreation_verified=False))
    print('Full-reference profiling and generation complete',flush=True)

if __name__=='__main__':main()
