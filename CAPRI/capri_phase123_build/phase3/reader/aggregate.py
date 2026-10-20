"""Fit-only scheme aggregation; validation predictions are never opened here."""
import argparse, collections, itertools
from pathlib import Path
import numpy as np
from common import *

def aggregate(root, family):
    fit_count=read(root/'data/audit.json')['real_counts']['fit']
    choices=[];comparison={}
    for name in ['crnn','svtr','parseq']:
        candidates=[]
        for aux in [False,True]:
            run=root/(name+('_aux' if aux else '_plain'));status=read(run/'status.json');assert status['state']=='complete'
            hist=read(run/'history.json');best=min(hist,key=lambda r:(r['selection']['cer'],-r['selection']['exact_string_accuracy']))
            comparison[run.name]=dict(selection=best,synthetic=read(run/'synthetic_results.json'),checkpoint_sha256=sha(run/'best.pt'))
            candidates.append((best['selection']['cer'],-best['selection']['exact_string_accuracy'],run))
        choices.append(min(candidates,key=lambda x:x[:2])[2])
    write(root/'comparison.json',comparison)
    # Equal weight per architecture. Auxiliary variant selected strictly by synthetic selection.
    predictions=[];count_checks={};model_summaries={};hashes={};eligible=[]
    for run in choices:
        cal=read(run/'calibration_metrics.json');eligible.append(cal['cer']<=.35 and cal['exact_string_accuracy']>=.20)
        rows=[json.loads(l) for l in (run/'fit_predictions.jsonl').open()];assert len(rows)==fit_count and all(r['id'].startswith('fit_') for r in rows)
        if predictions:assert [(r['id'],r['pixel_sha256']) for r in rows]==[(r['id'],r['pixel_sha256']) for r in predictions[0]]
        predictions.append(rows);hashes[run.name]=sha(run/'best.pt')
        lengths=collections.Counter(r['decoded_length'] for r in rows);combos=collections.Counter(''.join(map(str,r['decoded_groups'])) for r in rows)
        model_summaries[run.name]=dict(length_histogram=dict(lengths),group_combinations=dict(combos),group_prevalence=np.mean([r['decoded_groups'] for r in rows],axis=0).tolist(),mean_decode_confidence=float(np.mean([r['confidence'] for r in rows])),calibration=cal)
    # Check EVERY auxiliary head, even if the plain variant won synthetic selection.
    head_rates=[]
    for name in ['crnn','svtr','parseq']:
        run=root/(name+'_aux');rows=[json.loads(l) for l in (run/'fit_predictions.jsonl').open()]
        pairs=collections.Counter(f"{r['decoded_length']}:{r['head_count']}" for r in rows);rate=float(np.mean([r['decoded_length']!=r['head_count'] for r in rows]));head_rates.append(rate)
        count_checks[name]=dict(disagreement_fraction=rate,decoded_vs_head=dict(pairs),mean_head_probabilities=np.mean([r['count_probabilities'] for r in rows],axis=0).tolist(),mean_group_probabilities=np.mean([r['group_probabilities'] for r in rows],axis=0).tolist(),decoded_vs_group_head_disagreement=float(np.mean([r['decoded_groups']!=(np.asarray(r['group_probabilities'])>=.5).astype(int).tolist() for r in rows])))
    length_disagreement=float(np.mean([len({rows[i]['decoded_length'] for rows in predictions})>1 for i in range(fit_count)]))
    group_disagreement=float(np.mean([len({tuple(rows[i]['decoded_groups']) for rows in predictions})>1 for i in range(fit_count)]))
    text_disagreement=float(np.mean([len({rows[i]['prediction'] for rows in predictions})>1 for i in range(fit_count)]))
    # A length/group JOINT avoids inventing impossible combinations (3 groups at length 2).
    cells=[(n,''.join(map(str,bits))) for n in range(2,9) for bits in itertools.product([0,1],repeat=3) if 0<sum(bits)<=n]
    lookup={cell:i for i,cell in enumerate(cells)};vectors=np.zeros((fit_count,len(cells)));invalid=0
    for rows in predictions:
        for i,r in enumerate(rows):
            cell=(r['decoded_length'],''.join(map(str,r['decoded_groups'])))
            if cell in lookup:vectors[i,lookup[cell]]+=1/len(predictions)
            else:invalid+=1;vectors[i]+=np.ones(len(cells))/len(cells)/len(predictions)
    observed=vectors.mean(0);prior=np.ones(len(cells))/len(cells)
    uncertainty=max(length_disagreement,group_disagreement,float(np.mean(head_rates)),invalid/(fit_count*len(predictions)))
    prior_mass=min(.8,.2+.6*uncertainty);sampling=(1-prior_mass)*observed+prior_mass*prior
    rng=np.random.default_rng(731);boot=np.stack([vectors[rng.integers(fit_count,size=fit_count)].mean(0) for _ in range(500)])
    rows=[dict(length=n,groups=g,observed_weight=float(observed[j]),sampling_weight=float(sampling[j]),bootstrap_95_fit_sampling_interval=np.quantile(boot[:,j],[.025,.975]).tolist()) for j,(n,g) in enumerate(cells)]
    scheme=dict(schema=1,stage='Official CAPRI Phase 3',family=family,source_partition='fit',images=fit_count,model_selection='Best plain/aux per architecture by synthetic selection CER; equal architecture weights',models=hashes,model_summaries=model_summaries,auxiliary_crosschecks=count_checks,disagreement=dict(length=length_disagreement,groups=group_disagreement,text=text_disagreement),invalid_or_out_of_support_predictions=invalid,joint_distribution=rows,uncertainty_prior_mass=prior_mass,uncertainty_rule='0.2 + 0.6 * max(length disagreement, group disagreement, mean count-head disagreement, invalid fraction), capped .8; engineering safeguard, not calibrated real probability',group_order=GROUPS,alphabet=CHARSET,alphabet_policy='All 62 characters retained. Zero predictions do not establish exclusion.',length_policy='Weighted 2..8 range, never forced fixed length; training support limitation disclosed; decode support extends to 12',bootstrap_limit='Intervals reflect sampling variability conditional on these biased models, not real correctness or model uncertainty',real_accuracy=None,validation_used_for_scheme=False,synthetic_reading_gate=dict(passed=all(eligible),criterion='Each selected reader calibration CER <= .35 and exact >= .20; engineering minimum, not real validation'),shortcut_limit=read(root/'data/shortcut_audit.json')['claim'],status='estimated_with_uncertainty' if all(eligible) else 'blocked_inadequate_synthetic_reading')
    write(root/'inferred_scheme.json',scheme)
    with (root/'disagreement_examples.jsonl').open('w') as f:
        for i in range(fit_count):
            if len({rr[i]['prediction'] for rr in predictions})>1:f.write(json.dumps(dict(id=predictions[0][i]['id'],pixel_sha256=predictions[0][i]['pixel_sha256'],models={run.name:rr[i]['prediction'] for run,rr in zip(choices,predictions)}))+'\n')
    return scheme

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);a=p.parse_args();print(json.dumps(aggregate(a.root,a.family),indent=2))
