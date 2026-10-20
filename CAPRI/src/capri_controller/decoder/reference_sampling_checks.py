"""CPU sampling/resume invariants plus a bounded read of existing TRAIN data."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,read,write,now,sha
from .reference_sampling import ReferenceSampler
from .reference_data import TrainReferenceDataset
from .data import CorpusDataset,ShapeBatchPlan,exact_shape_collate


class ToyReferences:
    snapshot_hash='known-train-reference-fixture'
    def __init__(self):
        self.rows=[dict(record=dict(split='train',family_group=f,font_key=f+'/'+str(j//2),specimen_id=f+'/'+str(j)),
                        shape=(16,32 if j%2==0 else 48)) for f in ('alpha','beta','gamma') for j in range(4)]
    def __getitem__(self,i):
        row=self.rows[i];h,w=row['shape'];r=row['record']
        return dict(inputs=dict(pixels=torch.full((3,h,w),float(i)),valid_pixels=torch.ones(h,w,dtype=torch.bool)),
                    library_labels=dict(r),audit=dict(stem=r['specimen_id']))


def check_reference_sampling(run):
    if torch.cuda.is_available():raise RuntimeError('Hide CUDA for reference sampling checks')
    torch.set_num_threads(2);checks={};dataset=ToyReferences();mapping=dict(alpha=7,beta=42,gamma=99)
    sampler=ReferenceSampler(dataset,mapping,918)
    scene=dict(targets=[dict(font_family=torch.tensor([7,42])),dict(font_family=torch.tensor([99,7]))],
               audit=[dict(stem='scene_a',font_keys=['alpha/0','beta/0']),dict(stem='scene_b',font_keys=['gamma/1','alpha/1'])])
    state=random.getstate();numpy_state=np.random.get_state();torch_state=torch.get_rng_state().clone()
    choice=sampler.select(scene,epoch=1,max_references=2,mode='family')
    # Unrelated choices or prefetch cannot advance an implicit sampler cursor.
    sampler.select(scene,epoch=3,max_references=3,mode='font_file')
    assert choice==sampler.select(scene,epoch=1,max_references=2,mode='family')
    restored=ReferenceSampler(dataset,mapping,918)
    assert choice==restored.select(scene,epoch=1,max_references=2,mode='family')
    assert state==random.getstate() and torch.equal(torch_state,torch.get_rng_state())
    after=np.random.get_state();assert numpy_state[0]==after[0] and np.array_equal(numpy_state[1],after[1]) and numpy_state[2:]==after[2:]
    checks['retry_resume_prefetch_invariant_without_global_rng_mutation']=True
    permuted=copy.deepcopy(scene)
    permuted['targets'].reverse();permuted['audit'].reverse()
    for t,a in zip(permuted['targets'],permuted['audit']):t['font_family']=t['font_family'].flip(0);a['font_keys'].reverse()
    assert choice==sampler.select(permuted,1,2,'family')
    assert len({tuple(r['specimen_id'] for r in sampler.select(scene,e,2,'family')['selected']) for e in range(1,10)})>1
    checks['scene_order_invariant_and_epoch_specific_choices']=True
    assert choice['references']==2 and len(choice['omitted_families'])==1
    exact=sampler.select(scene,1,8,'font_file')
    assert exact['references']==3 and not exact['omitted_families']
    observed=set(k for a in scene['audit'] for k in a['font_keys'])
    assert all(r['font_key'] in observed for r in exact['selected'])
    checks['bounded_family_coverage_and_explicit_exact_file_mode']=True
    batches=sampler.materialize(json.loads(json.dumps(exact)))
    assert sum(len(b['library_labels']) for b in batches)==3
    actual_shapes=[]
    for batch in batches:
        assert set(batch['inputs'])=={'pixels','valid_pixels'}
        shape=list(batch['inputs']['pixels'].shape[-2:]);actual_shapes.append(shape)
        assert all(next(r['shape'] for r in exact['selected'] if r['specimen_id']==label['specimen_id'])==shape
                   for label in batch['library_labels'])
    checks['json_roundtrip_exact_shape_batches_no_extra_padding']=True
    def refuses(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Invalid reference selection accepted')
    bad=copy.deepcopy(choice);bad['selected'][0]['index']+=1;refuses(lambda:sampler.materialize(bad))
    refuses(lambda:ReferenceSampler(dataset,mapping,919).materialize(choice))
    wrong=copy.deepcopy(scene);wrong['targets'][0]['font_family'][0]=0
    refuses(lambda:sampler.select(wrong,1,2,'family'))
    wrong=copy.deepcopy(scene);wrong['audit'][0]['font_keys'][0]='unknown'
    refuses(lambda:sampler.select(wrong,1,2,'family'))
    bad=ToyReferences();bad.rows[0]['record']['split']='dev';refuses(lambda:ReferenceSampler(bad,mapping,918))
    bad=ToyReferences();bad.rows[1]['record']['specimen_id']=bad.rows[0]['record']['specimen_id']
    refuses(lambda:ReferenceSampler(bad,mapping,918))
    refuses(lambda:sampler.select(scene,0,2,'family'));refuses(lambda:sampler.select(scene,1,0,'family'))
    refuses(lambda:sampler.select(scene,1,2,'implicit'))
    checks['heldout_mismatched_ids_unknown_fonts_mutated_plan_changed_contract_refused']=True
    # Read existing data only: one TRAIN corpus shard and the verified reference
    # snapshot. No GPU, generation, DEV/calibration/test read or optimization.
    source=ROOT/read(ROOT/'configs/local.json')['decoder_run']
    references=TrainReferenceDataset(source);corpus=None
    try:
        corpus=CorpusDataset(source,split='train',allow_partial=True,max_shards=1)
        actual=ReferenceSampler(references,corpus.family_indices,918)
        plan=ShapeBatchPlan(corpus,batch_size=4,epoch=1,seed=918)
        decoded=0;examples=[]
        for indices in plan.batches[:3]:
            batch=exact_shape_collate([corpus[i] for i in indices])
            for mode in ('family','font_file'):
                selection=actual.select(batch,1,4,mode);reference_batches=actual.materialize(selection)
                assert sum(len(b['library_labels']) for b in reference_batches)==selection['references']
                assert len({r['family_group'] for r in selection['selected']})==selection['references']
                observed_keys={k for a in batch['audit'] for k in a['font_keys']}
                if mode=='font_file':assert all(r['font_key'] in observed_keys for r in selection['selected'])
                decoded+=selection['references']
                examples.append(dict(mode=mode,reference_count=selection['references'],subbatches=len(reference_batches),
                    selected_families=[r['family_group'] for r in selection['selected']],omitted_families=selection['omitted_families']))
        expected=read(source/'reference_data_checks.json')['snapshot_hash']
        assert references.snapshot_hash==expected
        checks['actual_train_reference_snapshot_and_corpus_ids_verified']=True
        evidence=dict(train_references=len(references),reference_snapshot=references.snapshot_hash,
            diagnostic_corpus_images=len(corpus),decoded_reference_selections=decoded,examples=examples,
            sampling_contract=actual.contract)
    finally:
        references.close()
        if corpus is not None:corpus.close()
    result=dict(created_at=now(),passed=True,device='cpu',checks=checks,actual_train_io=evidence,
                source_sha256=sha(Path(__file__).with_name('reference_sampling.py')),
                limitation='Sampling and bounded existing-data I/O checks only. No production reference frequency/budget, GPU two-forward readiness, phase launch or trained-quality claim.')
    write(Path(run)/'reference_sampling_checks.json',result);return result
