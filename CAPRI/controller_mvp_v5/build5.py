"""Controller 2: frozen DINO attention guides the existing renderer from hybrid recipes."""
from setup5 import ROOT,WORK,V3,V4,V1,U,FAMILIES,WEIGHTS_PATH
from dino5 import DinoComparator,Objective,WEIGHTS,palettes
from run import load_images,save_rows
import controls as C
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import copy,json,time
import numpy as np
from PIL import Image

def key(p):return json.dumps(p,sort_keys=True)
def eligible(r):return 'score' in r and r['score']['complete']==1 and r['score']['usable']>=.9
def select(records,n,quality=True):
    valid=[r for r in records if eligible(r)]
    if not valid:raise RuntimeError('No complete and usable candidate; gates not relaxed')
    best=min(r['score']['objective'] for r in valid)
    if quality:valid=[r for r in valid if r['score']['objective']<=2*max(best,.05)]
    vectors=np.asarray([r['vector'] for r in valid]);vectors=(vectors-vectors.mean(0))/np.maximum(vectors.std(0),.1)
    chosen=[int(np.argmin([r['score']['objective'] for r in valid]))]
    while len(chosen)<min(n,len(valid)):
        dist=np.min(np.mean((vectors[:,None]-vectors[chosen][None])**2,axis=2),axis=1)**.5
        merit=dist/(1+np.asarray([r['score']['objective'] for r in valid])/max(best,.05));merit[chosen]=-1
        chosen.append(int(merit.argmax()))
    return [valid[i] for i in chosen]

def proposals(current,phase,colors,fonts,rng,records):
    out=[]
    def add(**kw):p=copy.deepcopy(current);p.update(kw);out.append(p)
    if phase=='palette':
        bg=colors['background'];clean={n:0 for n in C.R.R.SAMPLED}
        add(background=bg[0]);add(background_pool=bg)
        for mode in ['single','random','alternating']:
            for effects in [current['effects'],clean]:
                add(background=bg[0],background_pool=None,text_palette=colors['text'],effect_palette=colors['effect'],color_mode=mode,effects=effects)
        add(effect_palette=colors['effect'])
    elif phase=='fonts':
        for f in fonts:add(font=f['key'],font_pool=None)
    elif phase=='geometry':
        for k,values in dict(size=[.65,.8,1.,1.15],width=[.7,.85,1.,1.2],spacing=[-.3,-.15,.15,.3],weight=[-.3,0,.3],
            rotation_absolute=[0,8,16,24,36],jitter_absolute=[0,.08,.16,.25],bold_extra=[0,1,2],length=list(range(2,9))).items():
            for v in values:add(**{k:v})
    elif phase=='effects':
        for name in C.R.R.SAMPLED:
            for level in [0,2,5]:add(effects={**current['effects'],name:level})
    elif phase=='joint':
        parents=select(records,12,quality=False)
        for _ in range(48):
            p=copy.deepcopy(parents[int(rng.integers(len(parents)))]['controls'])
            other=parents[int(rng.integers(len(parents)))]['controls']
            for field in ['font','size','width','spacing','weight','rotation_absolute','jitter_absolute','bold_extra']:
                if rng.random()<.5:p[field]=copy.deepcopy(other[field])
            for field in rng.choice(['size','width','spacing','weight'],2,replace=False):
                lo,hi={'size':(.55,1.25),'width':(.55,1.4),'spacing':(-.4,.4),'weight':(-.4,.4)}[field]
                p[field]=float(np.clip(p[field]+rng.normal(0,.08),lo,hi))
            out.append(p)
    return list({key(p):p for p in out}.values())

def write_records(dest,records):
    U.dump(dest/'history.json',[{k:v for k,v in r.items() if k!='vector'} for r in records])
    valid=[r for r in records if 'vector' in r]
    np.savez_compressed(dest/'candidate_vectors.npz',indices=np.asarray([r['index'] for r in valid]),vectors=np.asarray([r['vector'] for r in valid],dtype=np.float32))

def main():
    ROOT.mkdir(exist_ok=True);model=DinoComparator();started=time.time();fonts=U.read(WORK/'learned_controller_20261004/fonts.json')
    source=__import__('pathlib').Path.home()/'.cache/torch/hub/facebookresearch_dino_main'
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),controller='Controller 2: DINO attention',same_goal='Improve target-family text and background resemblance from pixels, starting from the frozen neural hybrid.',
        model='Official ImageNet self-supervised DINO ViT-S/8, frozen; no language encoder, VLM, OCR, or target labels.',weights=str(WEIGHTS_PATH),weights_sha256=U.sha(WEIGHTS_PATH),
        official_source='https://github.com/facebookresearch/dino',upstream_source_sha256=U.sha(source/'vision_transformer.py'),vendor_change='Only utils import made package-relative; architecture unchanged.',
        input='RGB 256x64, ImageNet normalization, 8x32 patch grid, all six last-layer CLS attention heads.',
        representation='Global CLS, six attention-focused patch embeddings, six complementary patch embeddings, attention layout and attention-weighted RGB histograms.',
        attention='Per-head 10th/90th percentile scaling; attention is not calibrated text probability or known character positions. No Controller 1 mask network is used.',
        objective_weights=WEIGHTS,fit='8 shared synthetic recipes; existing unlabeled target fit split',selection='16 different synthetic recipes; existing target selection split; up to 32 diverse fit candidates plus 8 CLS leaders',
        controls='Same existing renderer, font bank and hybrid recipe sources; palettes, fonts, geometry, length 2..8 and all sampled effects. No family-specific constants or added renderer features.',
        validity='Complete=100%, usable>=90%; retain up to16 unique configurations within2x best eligible selection objective. Never pad archive to a requested count.',
        ablation='CLS-only selection from the same DINO-guided candidate pool; this does not isolate attention during proposal generation.',
        test='Previously exposed v4 evaluation pixels (first64 per family); fresh renderer seeds; diagnostic only, not a new unseen-family or accuracy claim.',
        comparison='Hybrid neutral, frozen Controller1 single profile, Controller2 selected, and CLS-only selector. Prior C1 search budget differs; not a pure backbone-only ablation.',
        no_solver_training=True,code_sha256={p.name:U.sha(p) for p in ROOT.glob('*.py')})
    if (ROOT/'protocol.json').exists():assert U.read(ROOT/'protocol.json')['weights_sha256']==protocol['weights_sha256']
    else:U.dump(ROOT/'protocol.json',protocol)
    summaries=U.read(ROOT/'summary.json') if (ROOT/'summary.json').exists() else {}
    with ProcessPoolExecutor(max_workers=8,initializer=C.install) as pool:
        for family in FAMILIES:
            if family in summaries:continue
            dest=ROOT/family;dest.mkdir(exist_ok=True);rng=np.random.default_rng(51031)
            split=U.read(V3/(family+'_run_01')/'target_splits.json');testrefs=U.read(V4/'evaluation_pixels'/f'{family}.json')['rows'][:64]
            fit_ids={r['pixel_sha256'] for r in split['fit']};sel_ids={r['pixel_sha256'] for r in split['selection']};test_ids={r['pixel_sha256'] for r in testrefs}
            assert not fit_ids&sel_ids and not fit_ids&test_ids and not sel_ids&test_ids
            U.dump(dest/'target_splits.json',dict(fit=split['fit'],selection=split['selection'],test=testrefs))
            previous=V1/('run_03' if family=='City' else family+'_run_01');fitrecipes=U.read(previous/'fit_recipes.json')[:8];selrecipes=U.read(previous/'selection_recipes.json')[:16]
            realfit=load_images(split['fit']);target=model.extract(realfit);selection=model.extract(load_images(split['selection']));colors=palettes(realfit,target);U.dump(dest/'attention_palettes.json',colors)
            neutral=C.expand(C.R.NEUTRAL)
            def render(p,recipes):return list(pool.map(C.render,[(r,p) for r in recipes]))
            initial=render(neutral,fitrecipes);basefeatures=model.extract([r['image'] for r in initial]);objective=Objective(target,basefeatures)
            U.dump(dest/'objective.json',dict(std={k:v.tolist() for k,v in objective.std.items()},scale=objective.scale))
            records=[];seen=set()
            def evaluate(p,recipes,real,index,phase):
                rows=render(p,recipes);features=model.extract([r['image'] for r in rows]);s=objective.score(features,real,rows)
                return dict(index=index,phase=phase,controls=p,score=s,vector=objective.vector(features).tolist())
            records.append(evaluate(neutral,fitrecipes,target,0,'baseline'));seen.add(key(neutral));current=neutral
            for phase in ['palette','fonts','geometry','effects','joint']:
                options=proposals(current,phase,colors,fonts,rng,records)
                for p in options:
                    if key(p) in seen:continue
                    seen.add(key(p));i=len(records)
                    try:records.append(evaluate(p,fitrecipes,target,i,phase))
                    except Exception as exc:records.append(dict(index=i,phase=phase,controls=p,error=repr(exc)))
                    if len(records)%16==0:
                        U.dump(ROOT/'status.json',dict(family=family,phase=phase,candidates=len(records),seconds=time.time()-started));print(family,phase,len(records),flush=True)
                candidates=[r for r in records if eligible(r)]
                if candidates:current=min(candidates,key=lambda r:r['score']['objective'])['controls']
                write_records(dest,records)
            shortlist=select(records,32,quality=False)
            clsleaders=sorted([r for r in records if eligible(r)],key=lambda r:r['score']['components']['global_embedding'])[:8]
            shortlist=list({r['index']:r for r in shortlist+clsleaders}.values());validated=[]
            for r in shortlist:
                try:validated.append(evaluate(r['controls'],selrecipes,selection,r['index'],'selection'))
                except Exception as exc:validated.append(dict(index=r['index'],controls=r['controls'],error=repr(exc)))
            chosen=select(validated,16);best=chosen[0];clsbest=min([r for r in validated if eligible(r)],key=lambda r:r['score']['components']['global_embedding'])
            U.dump(dest/'selection.json',[{k:v for k,v in r.items() if k!='vector'} for r in validated])
            U.dump(dest/'archive.json',[r['controls'] for r in chosen]);U.dump(dest/'selected.json',dict(dino=best['controls'],cls_only=clsbest['controls'],selection_indices=dict(dino=best['index'],cls_only=clsbest['index']),frozen_utc=datetime.now(timezone.utc).isoformat()))
            frozen=U.sha(dest/'selected.json');testrecipes=copy.deepcopy(U.read(V3/(family+'_run_01')/'test_recipes.json')[:24])
            for i,r in enumerate(testrecipes):r['seed']=520003+i*7919
            U.dump(dest/'test_recipes.json',testrecipes);realtest=load_images(testrefs);ft=model.extract(realtest)
            np.savez_compressed(dest/'target_attention.npz',focus=ft['focus'],heads=ft['heads'])
            policies=dict(hybrid=neutral,controller1=U.read(V4/'collections'/family/'single.json'),dino=best['controls'],cls_only=clsbest['controls']);scores={}
            for arm,p in policies.items():
                rows=render(p,testrecipes);features=model.extract([r['image'] for r in rows]);save_rows(dest,arm,rows)
                scores[arm]=objective.score(features,ft,rows)
                # Synthetic truth is diagnostic only and never supplied to DINO or search.
                truth=np.stack([np.asarray(Image.fromarray(r['mask']).resize((256,64),Image.Resampling.NEAREST))>127 for r in rows]);focus=features['focus']>.5
                scores[arm]['attention_mask_iou']=float(((truth&focus).sum((1,2))/np.maximum(1,(truth|focus).sum((1,2)))).mean())
                np.savez_compressed(dest/arm/'attention.npz',focus=features['focus'],heads=features['heads'])
            assert U.sha(dest/'selected.json')==frozen
            fontchoices=sorted({str(r['controls']['font'] or 'hybrid') for r in chosen})
            result=dict(family=family,candidates=len(records),selection_candidates=len(validated),retained=len(chosen),font_choices=fontchoices,test_images=len(testrefs),generated_per_arm=len(testrecipes),scores=scores,
                selected_sha256=frozen,archive_sha256=U.sha(dest/'archive.json'),target_answers_used=False,vlm=False,visual_quality_pass=None)
            U.dump(dest/'results.json',result);summaries[family]=result;U.dump(ROOT/'summary.json',summaries)
            print(family,'DONE',len(chosen),'profiles',fontchoices,'scores',{k:round(v['appearance'],3) for k,v in scores.items()},flush=True)
    assert U.sha(WEIGHTS_PATH)==protocol['weights_sha256']
    U.dump(ROOT/'status.json',dict(phase='complete',seconds=time.time()-started,families=len(summaries)))

if __name__=='__main__':main()
