"""Paired comparator ablation, then expanded-control search. Targets supply pixels only."""
from bootstrap import ROOT,V1,U
import argparse,copy,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import run as OLD
import comparison as OLD_C
from perception import Perception,Objective,WEIGHTS
import controls as C

def unique(candidates):return list({json.dumps(p,sort_keys=True):p for p in candidates}.values())
def fixed_candidates(start,colors,fonts,effects,rng):
    result=[C.expand(start),C.expand(C.R.NEUTRAL)]
    for phase,budget in [('palette',8),('geometry',20),('font',24),('effects',26),('refine',16)]:
        options=OLD.variants(start,phase,colors,fonts,effects,rng)[1:]
        rng.shuffle(options);result.extend(C.expand(p) for p in options[:budget])
    result=unique(result)
    while len(result)<96:result=unique(result+[C.expand(p) for p in OLD.variants(start,'refine',colors,fonts,effects,rng)[1:]])
    return result[:96]

def expanded_candidates(start,phase,colors,bgpool,fontpool,effects,rng):
    result=[copy.deepcopy(start)]
    def add(**kw):result.append({**copy.deepcopy(start),**kw})
    if phase=='layout':
        for n in range(2,9):add(length=n)
        for x in [0,.06,.12,.2]:add(jitter_absolute=x,jitter=1.)
        for x in [0,8,16,28]:add(rotation_absolute=x,rotation=1.)
        for x in [0,1,2]:add(bold_extra=x)
        for x in [.65,.8,1.,1.15]:add(size=x)
        for x in [.7,1.,1.3]:add(width=x)
        for x in [-.2,0,.2,.5]:add(spacing=x)
    elif phase=='background':
        # Enumerate each available under-effect alone (plus the existing border),
        # so redundant clutter can be removed together without family templates.
        for name in C.R.R.UNDER:
            if name in C.R.R.SAMPLED:add(keep_only=[name,'border'],effects={name:3,'border':start['effects'].get('border',0)})
        add(keep_only=[])
        for layout in ['random','tiled','overlap_stack','strips','dense']:
            for level in [2,5]:add(block_layout=layout,block_speckle=False,keep_only=['blocks','border'],effects={'blocks':level})
        for level in [1,2,5,8]:add(effects={**start['effects'],'bands':level})
    else:
        for mode in ['single','random','alternating']:
            for count in [3,8,24]:add(text_palette=colors['text'][:count],color_mode=mode)
        add(background_pool=bgpool)
        add(background_pool=bgpool,text_palette=colors['text'],color_mode='random')
        for pool in fontpool:add(font_pool=pool,font=None)
        for _ in range(20):
            p=copy.deepcopy(start)
            for key in rng.choice(['size','width','spacing','jitter_absolute','rotation_absolute'],2,replace=False):
                lo,hi={'size':(.55,1.25),'width':(.55,1.4),'spacing':(-.35,.5),'jitter_absolute':(0,.25),'rotation_absolute':(0,30)}[key]
                old=p[key] if p[key] is not None else (0 if key.endswith('absolute') else 1.)
                p[key]=float(np.clip(old+rng.normal(0,.15*(hi-lo)),lo,hi))
            if rng.random()<.5:p['background_pool']=bgpool
            if rng.random()<.5:p['text_palette']=colors['text'];p['color_mode']='random'
            result.append(p)
    result=unique(result)
    # Fixed budget of 32, including incumbent. Deterministic subsampling, not visual selection.
    if len(result)>32:
        ids=rng.choice(np.arange(1,len(result)),31,replace=False);result=[result[0]]+[result[i] for i in sorted(ids)]
    while len(result)<32:
        p=copy.deepcopy(start);p['size']=float(rng.uniform(.65,1.15));p['width']=float(rng.uniform(.7,1.3));result.append(p)
    return result

def preview(out,real,groups):
    names=list(groups);w=300;n=min(8,len(real));sheet=Image.new('RGB',(w*(1+len(names)),45+n*100),'#f4f5f7');draw=ImageDraw.Draw(sheet)
    for j,name in enumerate(['Target',*names]):draw.text((j*w+8,12),name,fill='black')
    for i in range(n):
        for j,im in enumerate([real[i],*[groups[k][i]['image'] for k in names]]):
            image=ImageOps.contain(Image.fromarray(im),(w-12,88),Image.Resampling.NEAREST)
            sheet.paste(image,(j*w+(w-image.width)//2,40+i*100+(88-image.height)//2))
    sheet.save(out/'preview.png')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--previous',required=True);ap.add_argument('--splits',required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();old=Path(args.previous);out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def event(stage,**kw):U.dump(out/'status.json',dict(stage=stage,seconds=round(time.time()-start,1),**kw))
    split=U.read(args.splits);OLD.validate_split(split);U.dump(out/'target_splits.json',split)
    U.dump(out/'protocol.json',dict(comparator_ablation_candidates=96,expanded_control_candidates=96,fit_renders=18,selection_renders=36,test_renders=72,
        target_counts={k:len(split[k]) for k in ['fit','selection','test']},target_answers=False,weights=WEIGHTS,
        ablation='Old and new comparators rank exactly the same 96 candidate renders. Expanded search adds 96 candidates and additional controls; its gain is not attributable to controls alone.',
        fresh_test='Excluded prior ablation and MVP target splits. CryptBB has only 24 remaining unique images: exploratory evidence, no precision claim.',
        promotion='Selection must not worsen text/background more than 10% versus starting MVP, completeness may not decrease, usable rate may drop at most .03.',
        source_sha256={p.name:U.sha(p) for p in ROOT.glob('*.py')}))
    model=Perception();fit_images=OLD.load_images(split['fit']);sel_images=OLD.load_images(split['selection'])
    real=model.extract(fit_images);selection=model.extract(sel_images);event('perception_ready')
    np.savez_compressed(out/'mask_diagnostics.npz',fit=real['mask_diagnostic'],selection=selection['mask_diagnostic'])
    colors={k:OLD_C.palette(fit_images,real,k,count=24 if k=='text' else 8) for k in ['background','text','effect']}
    bgpool=[OLD_C.palette([im],{'mask':real['mask'][i:i+1]},'background',count=1)[0] for i,im in enumerate(fit_images)]
    old_start=U.read(old/'selected_controls.json');start_policy=C.expand(old_start)
    fonts=U.read(U.WORK/'learned_controller_20261004/fonts.json');choices=[f['key'] for f in fonts]
    pools=[[f['key'] for f in fonts if f['category']==cat][:8] for cat in sorted(set(f['category'] for f in fonts))]
    pools.append(choices)
    fit_recipes=U.read(old/'fit_recipes.json');sel_recipes=U.read(old/'selection_recipes.json')
    effects=list(dict.fromkeys([e['name'] for r in fit_recipes for e in r['effects_spec']['effects']]+list(C.R.R.UNDER)))
    effects=[e for e in effects if e in C.R.R.SAMPLED]
    rng=np.random.default_rng(85041);history=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=C.install) as pool:
        def render_features(p,recipes):
            rows=list(pool.map(C.render,[(r,p) for r in recipes]));f=model.extract([r['image'] for r in rows]);return rows,f
        initial_rows,initial_f=render_features(start_policy,fit_recipes)
        old_objective=OLD_C.Objective(real,initial_f);objective=Objective(real,initial_f)
        judges={'old':old_objective,'new':objective}
        sb_rows,sb_f=render_features(start_policy,sel_recipes)
        sb={k:j.score(sb_f,selection,sb_rows) for k,j in judges.items()}
        policies={k:copy.deepcopy(start_policy) for k in ['old','new']};best=copy.deepcopy(sb)
        def select(candidates,judge,incumbent,inc_score):
            for p in candidates:
                rows,f=render_features(p,sel_recipes);s=judges[judge].score(f,selection,rows)
                eligible=s['text']<=sb[judge]['text']*1.1 and s['background']<=sb[judge]['background']*1.1 and s['complete']>=sb[judge]['complete'] and s['usable']>=sb[judge]['usable']-.03
                history.append(dict(stage='selection',judge=judge,controls=p,score=s,eligible=eligible))
                if eligible and s['objective']<inc_score['objective']:incumbent=copy.deepcopy(p);inc_score=s
            return incumbent,inc_score
        candidates=fixed_candidates(old_start,colors,choices,effects,rng);U.dump(out/'paired_candidates.json',candidates)
        ranked={k:[] for k in judges}
        for i,p in enumerate(candidates):
            try:
                rows,f=render_features(p,fit_recipes);scores={k:j.score(f,real,rows) for k,j in judges.items()}
                for k,s in scores.items():ranked[k].append((s['objective'],p))
                history.append(dict(stage='paired_fit',index=i,controls=p,scores=scores))
            except Exception as exc:history.append(dict(stage='paired_fit',index=i,error=repr(exc)))
            if i%12==0:event('paired_comparator_search',completed=i+1,total=len(candidates));U.dump(out/'history.json',history)
        for k in judges:
            ranked[k].sort(key=lambda x:x[0]);policies[k],best[k]=select([p for _,p in ranked[k][:4]],k,policies[k],best[k])
        U.dump(out/'comparator_policies.json',policies);event('paired_comparator_complete',selection=best)
        expanded=copy.deepcopy(policies['new']);expanded_score=best['new']
        for phase in ['layout','background','variation']:
            ranked_phase=[];candidates=expanded_candidates(expanded,phase,colors,bgpool,pools,effects,rng)
            for i,p in enumerate(candidates):
                try:
                    rows,f=render_features(p,fit_recipes);score=objective.score(f,real,rows);ranked_phase.append((score['objective'],p))
                    history.append(dict(stage=phase,index=i,controls=p,score=score))
                except Exception as exc:history.append(dict(stage=phase,index=i,error=repr(exc)))
                if i%12==0:event('expanded_search',phase=phase,completed=i+1,total=len(candidates))
            ranked_phase.sort(key=lambda x:x[0]);expanded,expanded_score=select([p for _,p in ranked_phase[:4]],'new',expanded,expanded_score)
            U.dump(out/'history.json',history);event('expanded_phase_complete',phase=phase,selection=expanded_score)
        all_policies={'mvp1':start_policy,'old_comparator':policies['old'],'new_comparator':policies['new'],'expanded':expanded}
        U.dump(out/'frozen_policies.json',all_policies);digest=U.sha(out/'frozen_policies.json');event('frozen_before_test',sha256=digest)
        test_images=OLD.load_images(split['test']);target=model.extract(test_images)
        descriptions=U.read(old/'hybrid_descriptions.json')
        test_recipes=OLD.recipes(pool,descriptions,72,85041+300);U.dump(out/'test_recipes.json',test_recipes)
        groups={};scores={};features={}
        for k,p in all_policies.items():
            rows,f=render_features(p,test_recipes);groups[k]=rows;features[k]=f
            scores[k]={name:judge.score(f,target,rows) for name,judge in judges.items()};OLD.save_rows(out,k,rows);event('test_evaluation',arm=k)
        preview(out,test_images,{k:groups[k] for k in ['mvp1','new_comparator','expanded']})
        OLD.gallery(out,test_images,groups['mvp1'],groups['expanded'],target)
        assert U.sha(out/'frozen_policies.json')==digest
        result=dict(test=scores,selection=dict(start=sb,paired=best,expanded=expanded_score),seconds=round(time.time()-start,1),frozen_sha256=digest,visual_review='pending',
                    failures=[r for r in history if 'error' in r],scope='Three arms distinguish scoring change from added search/controls; old comparator paired arm is also saved.')
        U.dump(out/'results.json',result);U.dump(out/'history.json',history)
        np.savez_compressed(out/'test_features.npz',**{'target_'+k:v for k,v in target.items()},**{name+'_'+k:v for name,f in features.items() for k,v in f.items()})
        event('complete',seconds_total=result['seconds'])
if __name__=='__main__':main()
