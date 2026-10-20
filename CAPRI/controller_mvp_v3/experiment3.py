"""No VLM. Target images provide pixels only. Synthetic masks never enter search scoring."""
from boot import ROOT,V1,V2,U
import argparse,copy,time,json,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
import run as OLD
from comparison import palette
from perception3 import Perception,Objective,V2Objective,WEIGHTS
from experiment import fixed_candidates,preview
from search3 import candidates,C

def main():
    ap=argparse.ArgumentParser();ap.add_argument('family');ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    family=args.family;out=ROOT/(family+'_run_01');out.mkdir(exist_ok=False);start=time.time()
    def event(stage,**kw):
        record=dict(stage=stage,seconds=round(time.time()-start,1),**kw);U.dump(out/'status.json',record);print(json.dumps(record),flush=True)
    split=U.read(ROOT/(family+'_split.json'));OLD.validate_split(split);U.dump(out/'target_splits.json',split)
    previous=V1/('run_03' if family=='City' else family+'_run_01');v2=V2/(family+'_run_01')
    initial=C.expand(U.read(v2/'frozen_policies.json')['expanded'])
    U.dump(out/'protocol.json',dict(vlm=False,target_labels=False,mask_checkpoint=U.sha(ROOT/'mask_training/best.pt'),hybrid_checkpoint=U.sha(U.HYBRID),
        paired_candidates=96,fit_renders=18,selection_renders=36,test_renders=72,weights=WEIGHTS,
        ablation='Identical paired candidate pixels ranked by old mask/old score, new mask/old score, and new mask/new typography score. Full v3 additionally searches every effect, backgrounds, fonts, layout and palettes; extra search budget is a confound.',
        promotion='Fit proposals; selection objective must improve, text/background each <= 1.05 of v2 under that judge, completeness >= v2, usable >= v2 minus .03.',
        source_sha256={p.name:U.sha(p) for p in ROOT.glob('*.py')},fresh_count=len(split['fresh']),test_policy='Freeze controls before feature extraction on diagnostic/fresh target pixels. No test-driven reruns.'))
    model=Perception();fit_images=OLD.load_images(split['fit']);sel_images=OLD.load_images(split['selection'])
    real=model.extract_pair(fit_images);selection=model.extract_pair(sel_images)
    colors={k:palette(fit_images,real[1],k,count=16 if k=='text' else 8) for k in ['text','background','effect']}
    bgpool=[palette([im],{'mask':real[1]['mask'][i:i+1]},'background',count=1)[0] for i,im in enumerate(fit_images)]
    fontmeta=U.read(U.WORK/'learned_controller_20261004/fonts.json');fonts=[f['key'] for f in fontmeta]
    fit_recipes=U.read(previous/'fit_recipes.json');sel_recipes=U.read(previous/'selection_recipes.json');history=[];rng=np.random.default_rng(96317)
    effects=list(dict.fromkeys([e['name'] for r in fit_recipes for e in r['effects_spec']['effects']]+list(C.R.R.UNDER)))
    effects=[n for n in effects if n in C.R.R.SAMPLED]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=C.install) as pool:
        def evaluate(p,recipes):
            rows=list(pool.map(C.render,[(r,p) for r in recipes]));return rows,model.extract_pair([r['image'] for r in rows])
        rows,features=evaluate(initial,fit_recipes)
        judges={'old':(V2Objective(real[0],features[0]),0),'mask':(V2Objective(real[1],features[1]),1),'typography':(Objective(real[1],features[1]),1)}
        def score(f,target,rows):return {k:j.score(f[i],target[i],rows) for k,(j,i) in judges.items()}
        rows,f=evaluate(initial,sel_recipes);base=score(f,selection,rows)
        def select(proposals,judge,incumbent,current):
            for p in proposals:
                rows,f=evaluate(p,sel_recipes);s=score(f,selection,rows)[judge];b=base[judge]
                eligible=s['text']<=b['text']*1.05 and s['background']<=b['background']*1.05 and s['complete']>=b['complete'] and s['usable']>=b['usable']-.03
                history.append(dict(stage='selection',judge=judge,controls=p,score=s,eligible=eligible))
                if eligible and s['objective']<current['objective']:incumbent=copy.deepcopy(p);current=s
            return incumbent,current
        paired=fixed_candidates(initial,colors,fonts,effects,rng);U.dump(out/'paired_candidates.json',paired);ranked={k:[] for k in judges}
        for i,p in enumerate(paired):
            try:
                rows,f=evaluate(p,fit_recipes);scores=score(f,real,rows)
                for k,s in scores.items():ranked[k].append((s['objective'],p))
                history.append(dict(stage='paired',index=i,controls=p,scores=scores))
            except Exception as exc:history.append(dict(stage='paired',index=i,error=repr(exc)))
            if i%16==0:event('paired_search',completed=i+1,total=len(paired));U.dump(out/'history.json',history)
        policies={'v2':initial};selection_scores={}
        for k in judges:
            ranked[k].sort(key=lambda x:x[0]);policies[k],selection_scores[k]=select([p for _,p in ranked[k][:4]],k,initial,base[k])
        full=copy.deepcopy(policies['typography']);best=selection_scores['typography']
        for phase in ['all_effects','background','font','layout','palette','refine']:
            options=candidates(full,phase,colors,fonts,bgpool,rng);rank=[]
            U.dump(out/(phase+'_candidates.json'),options)
            for i,p in enumerate(options):
                try:
                    rows,f=evaluate(p,fit_recipes);s=judges['typography'][0].score(f[1],real[1],rows);rank.append((s['objective'],p))
                    history.append(dict(stage=phase,index=i,controls=p,score=s))
                except Exception as exc:history.append(dict(stage=phase,index=i,error=repr(exc)))
                if i%24==0:event('full_search',phase=phase,completed=i+1,total=len(options));U.dump(out/'history.json',history)
            rank.sort(key=lambda x:x[0]);full,best=select([p for _,p in rank[:4]],'typography',full,best)
        policies['full']=full;U.dump(out/'frozen_policies.json',policies);digest=U.sha(out/'frozen_policies.json');event('frozen',sha256=digest)
        # Exact v2 synthetic recipe seeds permit paired visual and validity checks.
        recipes=U.read(v2/'test_recipes.json');U.dump(out/'test_recipes.json',recipes)
        groups={};generated={};all_scores={};verification={}
        for k,p in policies.items():
            rows,f=evaluate(p,recipes);groups[k]=rows;generated[k]=f;OLD.save_rows(out,k,rows)
            truth=np.stack([np.asarray(Image.fromarray(r['mask']).resize((256,64),Image.Resampling.NEAREST))>127 for r in rows])
            metrics={}
            for name,feature in zip(['old_mask','new_mask'],f):
                pred=feature['mask']>.5;metrics[name]=float(((pred&truth).sum((1,2))/np.maximum(1,(pred|truth).sum((1,2)))).mean())
            targethashes={r['pixel_sha256'] for g in ['fit','selection','test','fresh'] for r in split[g]}
            for r in rows:
                assert hashlib.sha256(r['image'].tobytes()).hexdigest()==r['pixel_sha256'] and r['pixel_sha256'] not in targethashes
                assert r['text']==r['text_spec']['text']
            verification[k]=dict(count=len(rows),synthetic_mask_iou=metrics,usable=float(np.mean([r['usable'] for r in rows])))
        for testkey in ['test','fresh']:
            if not split[testkey]:continue
            images=OLD.load_images(split[testkey]);target=model.extract_pair(images)
            all_scores[testkey]={k:score(generated[k],target,groups[k]) for k in policies}
            preview(out,images,{'v2':groups['v2'],'mask only':groups['mask'],'typography':groups['typography'],'full v3':groups['full']})
            (out/'preview.png').rename(out/(testkey+'_preview.png'))
            # Pixel-only masks displayed separately; no target ground truth exists.
            maskgroups={}
            for name,features in zip(['old mask','new mask'],target):
                maskgroups[name]=[{'image':np.repeat((m*255).astype(np.uint8)[:,:,None],3,axis=2)} for m in features['mask']]
            preview(out,images,maskgroups);(out/'preview.png').rename(out/(testkey+'_masks.png'))
            np.savez_compressed(out/(testkey+'_features.npz'),**{k+'_'+name:v for k,features in [('old',target[0]),('new',target[1])] for name,v in features.items()})
        assert U.sha(out/'frozen_policies.json')==digest
        U.dump(out/'results.json',dict(scores=all_scores,selection=selection_scores,full_selection=best,baseline_selection=base,verification=verification,frozen_sha256=digest,failures=[r for r in history if 'error' in r],seconds=round(time.time()-start,1),visual_review='pending',fresh_count=len(split['fresh'])))
        U.dump(out/'history.json',history);event('complete')

if __name__=='__main__':main()
