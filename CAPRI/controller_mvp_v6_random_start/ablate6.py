"""Pixel-only DINO controller: paired hybrid versus generic random recipe banks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'controller_mvp_v5'))
from setup5 import U,V1,V3,V4,WORK,FAMILIES,WEIGHTS_PATH
from dino5 import DinoComparator,Objective,palettes
from build5 import proposals,eligible,select,key,write_records
from run import load_images,save_rows
import controls as C
import numpy as np
import copy,time,argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone

SEEDS=[60101,60202,60303]
COUNTS={'fit':4,'selection':8,'test':24}
BUDGETS={'palette':9,'fonts':48,'geometry':34,'effects':76,'joint':24}

class GenericBank:
    def __init__(self,fonts):
        bank=C.R.bank();self.fonts=[bank.by_key[f['key']] for f in fonts]
        self.by_key={f['key']:f for f in self.fonts}
    def pool(self,split='train',caseless=False):
        return [f for f in self.fonts if f['split']==split and (caseless or not f['caps_only'])]

def random_recipes(seed,n,size,fonts,offset=0):
    """Only generic seed, renderer/font prior and pixel dimensions enter here."""
    bank=GenericBank(fonts);out=[]
    for i in range(n):
        rng=np.random.default_rng(seed+(i+offset)*7919)
        ts=C.R.T.sample_text_spec(rng,bank)
        es=C.R.R.sample_spec_k(rng,int(rng.integers(0,5)),levels=range(1,6))
        out.append(dict(description={'content':ts['charset']},seed=610003+(i+offset)*7919,index=i,
            size=list(size),text_spec=ts,effects_spec=es,origin='generic_renderer_random_prior'))
    return out

def safe_render(item):
    try:return C.render(item)
    except Exception as exc:return {'error':repr(exc),'complete':False,'usable':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=FAMILIES);args=ap.parse_args()
    fonts=U.read(WORK/'learned_controller_20261004/fonts.json');model=DinoComparator();started=time.time()
    neutral=C.expand(C.R.NEUTRAL)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),random_seeds=SEEDS,
        experiment='Replace entire hybrid starting recipe bank with generic random recipes; retain DINO controller.',
        target_input='Existing disjoint fit/selection/diagnostic RGB pixels only. No answers, target masks or boxes.',
        prior='Renderer sample_text_spec on the existing 96 generic fonts; 0..4 uniform effects at levels 1..5. Independent recipe draws, not hybrid perturbations. Canvas dimensions are the median target fit image dimensions.',
        counts=COUNTS,phase_candidate_slots=BUDGETS,total_fit_slots=1+sum(BUDGETS.values()),selection_slots=16,
        fairness='Identical phase slots, font subset, proposal RNG, render seeds, target splits and comparator per arm. Repeated configurations still consume slots; render failures count against validity. Each arm has its own independent recipe bank, including selection/test draws.',
        calibration='Common score normalized once using 16 independent generic random recipes (seed 60999), never hybrid recipes; all arms share it.',
        limits='Whole starting-recipe ablation, not initialization-only optimization on identical fixed latent variables. Existing controller cannot directly change charset/layout/slant/size_jitter. Test pixels were exposed in earlier MVPs; diagnostics, not fresh generalization. No VLM, OCR, solver training or recognition claim.',
        failure='If no candidate meets complete=1 and usable>=.9, report search failure; show lowest penalized objective as failed diagnostic only, never relabel it a successful controller.',
        display='First random seed and first samples fixed in advance; all three random runs reported. Do not pick random seed using test scores.',
        weights_sha256=U.sha(WEIGHTS_PATH),code_sha256=U.sha(Path(__file__)))
    if not (ROOT/'protocol.json').exists():U.dump(ROOT/'protocol.json',protocol)
    else:assert U.read(ROOT/'protocol.json')['code_sha256']==protocol['code_sha256'],'Protocol code changed'
    summaries=U.read(ROOT/'summary.json') if (ROOT/'summary.json').exists() else {}
    with ProcessPoolExecutor(max_workers=8,initializer=C.install) as pool:
        for family in ([args.family] if args.family else FAMILIES):
            if family in summaries:continue
            dest=ROOT/family;dest.mkdir(exist_ok=True)
            split=U.read(V3/(family+'_run_01')/'target_splits.json')
            refs=U.read(V4/'evaluation_pixels'/f'{family}.json')['rows'][:64]
            groups=[{r['pixel_sha256'] for r in rows} for rows in [split['fit'],split['selection'],refs]]
            assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
            U.dump(dest/'target_splits.json',dict(fit=split['fit'],selection=split['selection'],test=refs))
            images=load_images(split['fit']);size=np.median([[im.shape[1],im.shape[0]] for im in images],axis=0).astype(int).tolist()
            targets={'fit':model.extract(images),'selection':model.extract(load_images(split['selection']))}
            colors=palettes(images,targets['fit']);U.dump(dest/'palettes.json',colors)
            def render(p,recipes):return list(pool.map(safe_render,[(r,p) for r in recipes]))
            def features(rows):
                # Failed renders have no visual image; use a fixed blank canvas and retain failure penalty.
                return model.extract([r.get('image',np.full((size[1],size[0],3),255,np.uint8)) for r in rows])
            calibration=random_recipes(60999,16,size,fonts,offset=300)
            objective=Objective(targets['fit'],features(render(neutral,calibration)))
            U.dump(dest/'calibration_recipes.json',calibration)
            U.dump(dest/'objective.json',dict(std={k:v.tolist() for k,v in objective.std.items()},scale=objective.scale))
            previous=V1/('run_03' if family=='City' else family+'_run_01')
            banks={}
            banks['hybrid']={'fit':U.read(previous/'fit_recipes.json')[:4],
                'selection':U.read(previous/'selection_recipes.json')[:8],
                'test':U.read(V3/(family+'_run_01')/'test_recipes.json')[:24]}
            for seed in SEEDS:
                banks[f'random_{seed}']={phase:random_recipes(seed,n,size,fonts,offset=offset)
                    for (phase,n),offset in zip(COUNTS.items(),[0,100,200])}
            for bank in banks.values():
                for phase,recipes in bank.items():
                    for i,r in enumerate(recipes):
                        r['size']=size;r['seed']=610003+(i+{'fit':0,'selection':100,'test':200}[phase])*7919
            frozen={};runs={}
            for arm,bank in banks.items():
                armroot=dest/arm;armroot.mkdir(exist_ok=True);U.dump(armroot/'recipes.json',bank)
                rng=np.random.default_rng(61515);records=[]
                def evaluate(p,phase,index,label):
                    rows=render(p,bank[phase]);f=features(rows)
                    return dict(index=index,phase=label,controls=p,score=objective.score(f,targets[phase],rows),
                        vector=objective.vector(f).tolist(),render_errors=sum('error' in r for r in rows))
                records.append(evaluate(neutral,'fit',0,'baseline'));current=neutral
                for phase,slots in BUDGETS.items():
                    if phase=='joint' and not any(eligible(r) for r in records):
                        options=[] # No valid parents: reproducible local search around penalized leader.
                    else:options=proposals(current,phase,colors,fonts,rng,records)
                    if len(options)>slots:
                        ids=np.random.default_rng(61616).choice(len(options),slots,replace=False)
                        options=[options[i] for i in sorted(ids)]
                    while len(options)<slots:
                        p=copy.deepcopy(current)
                        field=['size','width','spacing','weight'][len(options)%4]
                        lo,hi={'size':(.55,1.25),'width':(.55,1.4),'spacing':(-.4,.4),'weight':(-.4,.4)}[field]
                        p[field]=float(rng.uniform(lo,hi));options.append(p)
                    for p in options:records.append(evaluate(p,'fit',len(records),phase))
                    candidates=[r for r in records if eligible(r)] or records
                    current=min(candidates,key=lambda r:r['score']['objective'])['controls']
                    write_records(armroot,records)
                    print(family,arm,phase,len(records),flush=True)
                    U.dump(ROOT/'status.json',dict(family=family,arm=arm,phase=phase,seconds=time.time()-started))
                eligible_fit=[r for r in records if eligible(r)]
                shortlist=select(records,16,quality=False) if eligible_fit else sorted(records,key=lambda r:r['score']['objective'])[:16]
                while len(shortlist)<16:shortlist.append(shortlist[len(shortlist)%len(shortlist)])
                validated=[evaluate(r['controls'],'selection',r['index'],'selection') for r in shortlist]
                valid=[r for r in validated if eligible(r)]
                best=min(valid or validated,key=lambda r:r['score']['objective'])
                U.dump(armroot/'selection.json',[{k:v for k,v in r.items() if k!='vector'} for r in validated])
                U.dump(armroot/'selected.json',dict(controls=best['controls'],eligible=bool(valid),index=best['index'],
                    frozen_utc=datetime.now(timezone.utc).isoformat()))
                frozen[arm]=U.sha(armroot/'selected.json')
                runs[arm]=dict(search_pass=bool(valid),fit_slots=len(records),unique_fit_controls=len({key(r['controls']) for r in records}),
                    selection_slots=len(validated),selected=best['controls'],selection_score=best['score'])
            # Every arm is frozen before loading diagnostic target pixels.
            targets['test']=model.extract(load_images(refs))
            for arm,bank in banks.items():
                armroot=dest/arm
                for label,p in [('start',neutral),('final',runs[arm]['selected'])]:
                    rows=render(p,bank['test']);runs[arm][label]=objective.score(features(rows),targets['test'],rows)
                    for i,r in enumerate(rows):
                        if 'error' in r:r.update(image=np.full((size[1],size[0],3),255,np.uint8),mask=np.zeros((size[1],size[0]),np.uint8),seed=bank['test'][i]['seed'],text='RENDER FAILURE')
                    save_rows(armroot,label,rows)
                assert U.sha(armroot/'selected.json')==frozen[arm]
                runs[arm]['selected_sha256']=frozen[arm]
            h=runs['hybrid']['final']['objective']
            result=dict(family=family,runs=runs,random_wins=sum(runs[f'random_{s}']['search_pass'] and runs[f'random_{s}']['final']['objective']<h for s in SEEDS),
                target_answers_used=False,vlm=False,seconds=time.time()-started)
            U.dump(dest/'results.json',result);summaries[family]=result;U.dump(ROOT/'summary.json',summaries)
            print(family,'DONE',{a:round(r['final']['objective'],3) for a,r in runs.items()},flush=True)
    U.dump(ROOT/'status.json',dict(phase='complete',families=list(summaries),seconds=time.time()-started))

if __name__=='__main__':main()
