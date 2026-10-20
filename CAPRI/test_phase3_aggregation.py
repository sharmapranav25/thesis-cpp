"""Ensure scheme computation tolerates no validation access and retains uncertainty."""
import sys,json
from pathlib import Path
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base/'phase3_reader'))
from aggregate import aggregate
from common import write,read
root=base/'phase3_aggregation_fixture_01';root.mkdir(exist_ok=False)
write(root/'data/shortcut_audit.json',dict(claim='synthetic fixture'))
for model,pred in zip(['crnn','svtr','parseq'],['12','AB','ab']):
    for suffix in ['plain','aux']:
        out=root/(model+'_'+suffix);out.mkdir()
        write(out/'status.json',dict(state='complete'));write(out/'history.json',[dict(selection=dict(cer=.1 if suffix=='plain' else .2,exact_string_accuracy=.5))]);write(out/'synthetic_results.json',{});write(out/'calibration_metrics.json',dict(cer=.1,exact_string_accuracy=.5));(out/'best.pt').write_bytes(b'fixture-not-a-checkpoint')
        with (out/'fit_predictions.jsonl').open('w') as f:
            for i in range(1599):
                gs=[int(model=='crnn'),int(model=='parseq'),int(model=='svtr')]
                row=dict(id=f'fit_{i:04d}',pixel_sha256=str(i),prediction=pred,decoded_length=2,decoded_groups=gs,confidence=.99,head_count=3,count_probabilities=[0,1,0,0,0,0,0],group_probabilities=gs)
                f.write(json.dumps(row)+'\n')
        (out/'validation_predictions.jsonl').write_bytes(b'POISON: validation must never be read')
def guard(event,args):
    if event=='open' and 'validation' in str(args[0]):raise AssertionError('Forbidden validation read')
sys.addaudithook(guard)
scheme=aggregate(root)
assert scheme['uncertainty_prior_mass']==.8 and scheme['disagreement']['groups']==1
assert all(c['sampling_weight']>0 for c in scheme['joint_distribution'])
assert abs(sum(c['sampling_weight'] for c in scheme['joint_distribution'])-1)<1e-10
assert all(not(c['length']==2 and c['groups']=='111') for c in scheme['joint_distribution'])
assert all(name.endswith('_plain') for name in scheme['models'])
print('PASS: validation-file access forbidden; synthetic-only variant selection; head disagreement retains broad prior; all feasible cells positive.')
