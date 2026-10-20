"""Coverage extension: tune all seven font attributes for every family.

Declared while the original family experiment was running. City removal results
were already inspected. Fixed candidate catalogue; no choices based on test scores.
The existing test sets are reused; this is exploratory, not a new untouched test.
"""
import json,time,sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
import experiment as e

def main():
    model=e.A.Features()
    with ProcessPoolExecutor(max_workers=6,initializer=e.P.install) as pool:
      for family in (sys.argv[1:] or ['City','CryptBB','DarkBay']):
        folder=e.OUT/family;dest=folder/'font_followup';dest.mkdir(exist_ok=False)
        protocol=dict(features=['font.'+h for h in e.FONT],candidates={h:e.candidates('font.'+h) for h in e.FONT},
            rationale='Ensure all seven font attributes receive individual tuning even when other groups dominate the broad screening.',
            evaluation='Original held-out sets reused, not a new untouched test. City ablation results were already inspected before declaring this extension.',
            selection='Top two fitting candidates and baseline per attribute; same selection objective and guards as main experiment. Combine the selected attributes and select versus baseline before test.',
            no_neural_updates=True,no_real_answers=True)
        e.dump(dest/'protocol.json',protocol)
        split=e.A.read(folder/'target_splits.json');recipes={s:e.A.read(folder/f'{s}_recipes.json') for s in ('fit','selection','test')}
        real={}
        for s in ('fit','selection'):
            ims=e.A.real_images(split[s]);real[s]=model(ims);real[s]['shape']=e.shape_features(ims)
        judge=e.Judges(model,real['fit']);cache={};history=[];count=0
        def evaluate(stage,ops):
            nonlocal count
            key=(stage,json.dumps(ops,sort_keys=True))
            if key not in cache:
                rr=list(pool.map(e.render_task,[(r,ops) for r in recipes[stage]]));count+=len(rr)
                f=judge.extract(rr);s=judge.score(f,real[stage],rr)
                cache[key]=dict(ops=ops,rows=rr,features=f,score=s)
                history.append(dict(stage=stage,ops=ops,score=s))
            return cache[key]
        base={s:evaluate(s,{}) for s in ('fit','selection')};selected={}
        existing=e.A.read(folder/'selected_policies.json')['individual']
        for h in e.FONT:
            key='font.'+h
            if key in existing:
                selected[key]=existing[key]
                print(json.dumps(dict(stage='font_selection_reused',family=family,feature=key,ops=existing[key]['ops'])),flush=True)
                continue
            options=[evaluate('fit',{key:v}) for v in e.candidates(key)]
            top=sorted(options,key=lambda v:v['score']['objective'])[:2]
            vals=[base['selection']]+[evaluate('selection',v['ops']) for v in top]
            winner=min([v for v in vals if v is base['selection'] or e.eligible(v['score'],base['selection']['score'])],key=lambda v:v['score']['objective'])
            selected[key]=dict(ops=winner['ops'],score=winner['score'])
            print(json.dumps(dict(stage='font_selection',family=family,feature=key,ops=winner['ops'])),flush=True)
        joint={k:v for result in selected.values() for k,v in result['ops'].items()}
        combined=evaluate('selection',joint)
        if not e.eligible(combined['score'],base['selection']['score']) or combined['score']['objective']>=base['selection']['score']['objective']:combined=base['selection']
        e.dump(dest/'selected_policies.json',dict(individual=selected,combined=combined['ops']))
        ims=e.A.real_images(split['test']);real['test']=model(ims);real['test']['shape']=e.shape_features(ims)
        b=evaluate('test',{});original=e.A.read(folder/'results.json')['baseline']
        assert abs(b['score']['appearance']-original['appearance'])<1e-6
        results=dict(individual={},baseline=b['score'])
        def assess(ops):
            v=evaluate('test',ops);s=v['score'];bs=b['score']
            result=dict(ops=ops,metrics=s,appearance_gain_pct=100*(bs['appearance']-s['appearance'])/bs['appearance'],
                objective_gain=bs['objective']-s['objective'],audit_gain_pct=100*(bs['audit']-s['audit'])/bs['audit'],
                guard_pass=e.eligible(s,bs),**judge.fb.bootstrap(b['features'],v['features'],real['test'],n=200))
            result['seed_block_appearance_gains']=[]
            for block in range(3):
                ids=np.array([i for i,r in enumerate(v['rows']) if r['block']==block])
                before=judge.fb.score(e.sub(b['features'],ids),real['test'],[b['rows'][i] for i in ids])
                after=judge.fb.score(e.sub(v['features'],ids),real['test'],[v['rows'][i] for i in ids])
                result['seed_block_appearance_gains'].append(before['appearance']-after['appearance'])
            result['pixel_changed_fraction']=float(np.mean([a['pixel_sha256']!=bb['pixel_sha256'] for a,bb in zip(v['rows'],b['rows'])]))
            return result,v
        gallery={'baseline':b['rows']}
        for key,policy in selected.items():
            results['individual'][key],v=assess(policy['ops'])
            e.A.save_group(dest,key.replace('.','_'),v['rows'])
        results['combined'],v=assess(combined['ops']);gallery['all_selected_fonts']=v['rows'];e.A.save_group(dest,'combined',v['rows'])
        best=min(selected,key=lambda key:selected[key]['score']['objective'])
        gallery['best_selection_attribute']=evaluate('test',selected[best]['ops'])['rows']
        e.coded_gallery(dest,ims,gallery)
        e.dump(dest/'results.json',results);e.dump(dest/'search_history.json',history);e.dump(dest/'render_counts.json',dict(direct_renders=count))
        print(json.dumps(dict(stage='font_family_complete',family=family,combined=results['combined']['appearance_gain_pct'],renders=count)),flush=True)

if __name__=='__main__':main()
