"""Analytic retrieval/persistence checks; no trained font accuracy claim."""
import copy
import shutil
import string
import time
from pathlib import Path
import torch
from .common import write,now,sha
from .retrieval import FontIndex,IDENTITY_FIELDS
from .specimens import plan_specimens


def check_retrieval(run):
    torch.set_num_threads(2);checks={}
    contract={k:'a'*64 for k in IDENTITY_FIELDS[:4]};contract.update(weights='ema',diagnostic=True)
    records=[]
    for i,(family,digest) in enumerate([('alpha','1'*64),('alpha','1'*64),('beta','2'*64),('beta','3'*64)]):
        records.append(dict(prototype_id='prototype_'+str(i),font_key=digest[:20],font_sha256=digest,
                            family_group=family,style='regular',split='train',specimen_id='library/'+str(i)))
    prototypes=torch.zeros(4,128);prototypes[0,0]=1;prototypes[1,2]=1;prototypes[2,1]=1;prototypes[3,1]=2
    query=torch.zeros(2,128);query[0,0]=1;query[1,1]=1
    index=FontIndex(prototypes,records,contract)
    found=index.search(query,contract,topk=2)
    assert found['per_query'][0]['candidates'][0]['family_group']=='alpha'
    assert found['per_query'][1]['candidates'][0]['family_group']=='beta'
    assert all(x['best_match_weight_fraction']==.5 for x in found['aggregate'])
    assert not found['calibrated'] and found['status']=='uncalibrated_candidates'
    assert found['per_query'][0]['candidates'][0]['similarity']==1.
    weighted=index.search(query,contract,quality=[.1,.9])
    assert weighted['aggregate'][0]['family_group']=='beta'
    assert abs(weighted['aggregate'][0]['best_match_weight_fraction']-.9)<1e-6
    absent=index.search(query[:1],contract,excluded_families=['alpha'])
    assert absent['aggregate'][0]['family_group']=='beta' and absent['aggregate'][0]['mean_similarity']==0.
    assert not absent['calibrated']  # Never turn nearest available into true identity.
    reverse=FontIndex(prototypes.flip(0),list(reversed(records)),contract)
    assert reverse.search(query,contract)==index.search(query,contract)
    query_reversed=index.search(query.flip(0),contract,topk=2)
    assert query_reversed['aggregate']==found['aggregate']
    assert [r['candidates'] for r in query_reversed['per_query']]==[r['candidates'] for r in reversed(found['per_query'])]
    assert index.search(query,contract,quality=[0,0])['status']=='no_visual_evidence'
    assert index.search(torch.empty(0,128),contract)['status']=='no_visual_evidence'
    assert index.search(query,contract,excluded_families=['alpha','beta'])['status']=='empty_candidate_bank'
    tie=torch.zeros(1,128);tie[0,3]=1
    assert index.search(tie,contract,topk=1)['aggregate'][0]['best_match_weight_fraction']==.5
    checks['known_neighbours_styles_alias_family_mixture_quality_and_absence']=True
    checks['order_invariance_exact_ties_empty_and_zero_evidence']=True
    def rejects(fn,kind=ValueError):
        try:fn()
        except kind:return
        raise AssertionError('Invalid request was accepted')
    rejects(lambda:index.search(query,dict(contract,weights='raw')))
    rejects(lambda:index.search(query,dict(contract,checkpoint_sha256='b'*64)))
    rejects(lambda:index.search(torch.zeros(1,128),contract))
    rejects(lambda:index.search(torch.full((1,128),float('nan')),contract))
    rejects(lambda:index.search(query,contract,quality=[-1,1]))
    rejects(lambda:FontIndex(prototypes,records[:-1],contract))
    repeated=copy.deepcopy(records);repeated[-1]['prototype_id']=repeated[0]['prototype_id']
    rejects(lambda:FontIndex(prototypes,repeated,contract))
    crossed=copy.deepcopy(records);crossed[1]['split']='test'
    rejects(lambda:FontIndex(prototypes,crossed,contract))
    checks['identity_zero_nonfinite_duplicates_and_cross_split_refused']=True
    destination=Path(run)/('retrieval_smoke_'+str(time.time_ns()))
    receipt=index.save(destination/'index')
    rejects(lambda:index.save(destination/'index'),FileExistsError)
    rejects(lambda:FontIndex.load(destination/'index',contract))
    restored=FontIndex.load(destination/'index',contract,allow_diagnostic=True)
    assert restored.search(query,contract)==found
    shutil.copytree(destination/'index',destination/'corruption_probe')
    with (destination/'corruption_probe'/'embeddings.npy').open('ab') as f:f.write(b'corrupt')
    rejects(lambda:FontIndex.load(destination/'corruption_probe',contract,allow_diagnostic=True))
    checks['atomic_round_trip_hash_tamper_overwrite_and_diagnostic_guard']=True
    base=dict(key='1'*20,sha256='1'*64,split_group='alpha',split='train',style='regular',
              chars=string.ascii_letters+string.digits,axes=[dict(tag='wght',min=100.,default=400.,max=900.)])
    plan=plan_specimens([base]);other=dict(base,key='2'*20,sha256='2'*64,split_group='beta')
    second=plan_specimens([other])
    assert [p['text'] for p in plan]==[p['text'] for p in second]
    assert len(plan)==36 and len({p['specimen_id'] for p in plan})==36
    assert set(''.join(p['text'] for p in plan))==set(base['chars'])
    assert all(len(p['text'])==6 for p in plan)
    assert {p['axes']['wght'] for p in plan}=={100.,400.,900.}
    rejects(lambda:plan_specimens([dict(base,split='test')]))
    checks['train_only_specimens_full_coverage_actual_axes_family_independent_content']=True
    result=dict(created_at=now(),passed=True,checks=checks,index_receipt=receipt,path=str(destination),
                source_hashes={n:sha(Path(__file__).with_name(n)) for n in ('retrieval.py','retrieval_checks.py','specimens.py')},
                limitation='Analytic embedding vectors, not model/font accuracy. No model weights or corpus images used. Production reference rendering/extraction, calibrated quality/rejection and retrieval evaluation remain required.')
    write(Path(run)/'retrieval_checks.json',result)
    return result
