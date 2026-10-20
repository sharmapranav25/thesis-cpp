"""Fault-injected CPU checks, isolated from real decoder training."""
import copy
import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import read,write,sha,now
from .phase_engine import run_phase,_validation
from .checkpoints import rng_state,execution_state


class ToyPixels:
    snapshot_hash='diagnostic-only-eight-pixel-images-v1'
    rows=[dict(stem=f'toy_{i:07d}',shape=(1,2)) for i in range(8)]
    def __len__(self):return 8
    def __getitem__(self,index):
        return dict(inputs=dict(pixels=torch.tensor([[[index/8.,1.]]]),valid_pixels=torch.ones(1,2,dtype=torch.bool)),
                    targets=dict(value=index/16.),audit=dict(index=index))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(2,1);self.dropout=torch.nn.Dropout(.2)
    def forward(self,pixels,valid_pixels):return self.linear(self.dropout(pixels.flatten(1))).flatten()
    def resume_configuration(self):return dict(diagnostic='phase-engine-toy-v1')


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def check_phase_engine(run):
    if torch.cuda.is_available():raise RuntimeError('Hide CUDA for CPU phase-engine diagnostics')
    torch.set_num_threads(2)
    root=Path(run)/('phase_engine_smoke_'+str(time.time_ns()));root.mkdir()
    def construct(seed=97):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        m=ToyModel();o=torch.optim.AdamW(m.parameters(),lr=.01,foreach=False)
        return m,o,torch.optim.lr_scheduler.StepLR(o,step_size=3,gamma=.8),torch.amp.GradScaler('cpu',init_scale=16.,growth_interval=2)
    def evaluate(m,epoch):
        assert not m.training and not torch.is_grad_enabled()
        # Validation must not advance any training RNG stream.
        random.random();np.random.rand();torch.rand(4)
        return dict(diagnostic_epoch=epoch,parameter_sum=float(sum(v.sum() for v in m.parameters())))
    seen=[]
    def loss(m,batch):
        seen.extend(x['index'] for x in batch['audit'])
        target=torch.tensor([t['value'] for t in batch['targets']])
        scale=.8+.1*random.random()+.1*np.random.rand()
        return (m(**batch['inputs'])-target).square().mean()*scale
    settings=dict(objective='toy-mse-stochastic-v1',evaluation='toy-v1',scheduler='StepLR-3-gamma0.8')
    def execute(name,parts,**changes):
        kwargs=dict(run=root/name,phase='toy',model=parts[0],dataset=ToyPixels(),optimizer=parts[1],scheduler=parts[2],
                    scaler=parts[3],compute_loss=loss,evaluate=evaluate,protocol=settings,epochs=2,batch_size=2,
                    seed=731,workers=0,recovery_every=2,diagnostic=True)
        kwargs.update(changes);return run_phase(**kwargs)
    def state(name,epoch=2):
        path=root/name/'phase_engine_diagnostics/toy'/f'epoch_{epoch:04d}'/'checkpoint.pt'
        return torch.load(path,map_location='cpu',weights_only=False)
    full=construct();execute('full',full)
    expected_seen=seen.copy();assert len(expected_seen)==16 and all(expected_seen[:8].count(i)==1 for i in range(8))
    assert all(expected_seen[8:].count(i)==1 for i in range(8))
    uninterrupted=state('full');seen.clear()
    parts=construct()
    try:execute('resumed',parts,interrupt_after=2)
    except RuntimeError as exc:assert 'Injected' in str(exc)
    else:raise AssertionError('Fault injection did not stop')
    execute('resumed',construct(991))
    resumed=state('resumed')
    fields=('raw','ema','optimizer','scheduler','scaler','sampler_state','rng','execution','metrics','optimizer_layout')
    assert all(equal(uninterrupted[k],resumed[k]) for k in fields)
    assert seen==expected_seen
    checks=dict(exact_cpu_raw_ema_optimizer_scheduler_scaler_rng_resume=True,consumed_cursor_no_repetition=True,
                each_image_once_per_epoch=True,every_epoch_full_checkpoint=True)
    assert resumed['scaler']['scale']>16.
    for at in (4,6):
        seen.clear();name=f'interrupted_{at}'
        try:execute(name,construct(),interrupt_after=at)
        except RuntimeError as exc:assert 'Injected' in str(exc)
        else:raise AssertionError('Boundary interruption did not stop')
        execute(name,construct(542))
        assert seen==expected_seen and all(equal(uninterrupted[k],state(name)[k]) for k in fields)
    checks['final_batch_and_later_epoch_recovery_exact']=True
    for name in ('full','resumed'):
        folder=root/name/'phase_engine_diagnostics/toy'
        assert len(list(folder.glob('epoch_*/checkpoint.pt')))==2
        assert len(list(folder.glob('epoch_*/validation_ema.json')))==2
    # Fail once after the epoch has been safely saved, then retry only validation.
    calls=[]
    def failed_eval(m,epoch):
        calls.append(epoch);evaluate(m,epoch);raise RuntimeError('Injected validation failure')
    seen.clear();parts=construct()
    try:execute('validation_retry',parts,evaluate=failed_eval)
    except RuntimeError as exc:assert 'validation failure' in str(exc)
    else:raise AssertionError('Validation fault did not stop')
    folder=root/'validation_retry/phase_engine_diagnostics/toy'
    checkpoint=folder/'epoch_0001/checkpoint.pt';before=sha(checkpoint)
    assert len(seen)==8 and calls==[1]
    execute('validation_retry',construct(43))
    assert sha(checkpoint)==before and seen==expected_seen
    assert all(equal(uninterrupted[k],state('validation_retry')[k]) for k in fields)
    checks['failed_validation_retried_without_retraining_or_overwrite']=True
    # Completed reruns do no optimization and preserve epoch bytes.
    seen.clear();final_hash=sha(folder/'epoch_0002/checkpoint.pt')
    execute('validation_retry',construct())
    assert not seen and sha(folder/'epoch_0002/checkpoint.pt')==final_hash
    checks['completed_phase_idempotent']=True
    # Check evaluation finally restores raw weights, modes, precision and RNG on failure.
    before_raw=copy.deepcopy(parts[0].state_dict());before_rng=rng_state();before_execution=execution_state(parts[0])
    fake=root/'validation_finally/epoch_0001';fake.mkdir(parents=True);write(fake/'receipt.json',dict(sha256='x'*64))
    alternate={k:v+1 for k,v in before_raw.items()}
    try:_validation(fake.parent,1,parts[0],alternate,failed_eval)
    except RuntimeError:pass
    assert equal(before_raw,parts[0].state_dict()) and equal(before_rng,rng_state())
    assert equal(before_execution,execution_state(parts[0]))
    checks['validation_failure_restores_raw_modes_and_rng']=True
    # Frozen protocol prevents a changed optimizer/epoch budget from resuming.
    for changes in (dict(epochs=3),dict(protocol=dict(settings,objective='changed'))):
        try:execute('full',construct(),**changes)
        except ValueError as exc:assert 'protocol mutation' in str(exc)
        else:raise AssertionError('Protocol mutation accepted')
    changed=construct();changed[1].param_groups[0]['lr']=.02
    try:execute('full',changed)
    except ValueError as exc:assert 'protocol mutation' in str(exc)
    else:raise AssertionError('Optimizer mutation accepted')
    checks['phase_objective_budget_optimizer_mutation_refused']=True
    # Same-batch AMP overflow retry must match a clean lower-scale run across
    # full epochs, including checkpoints, schedules, EMA and RNG continuation.
    low=construct();low=list(low);low[3]=torch.amp.GradScaler('cpu',init_scale=8.,growth_interval=2)
    execute('overflow_reference',low)
    retried=construct();fired=[]
    def overflow_once(grad):
        if not fired:fired.append(True);return torch.full_like(grad,float('inf'))
        return grad
    hook=retried[0].linear.weight.register_hook(overflow_once)
    execute('overflow_retry',retried);hook.remove()
    reference_state=state('overflow_reference');retry_state=state('overflow_retry')
    assert all(equal(reference_state[k],retry_state[k]) for k in fields)
    retry_log=(root/'overflow_retry/phase_engine_diagnostics/toy/amp_retries.jsonl').read_text().splitlines()
    assert len(retry_log)==1 and retry_state['metrics']['phase_runtime']['global_step']==uninterrupted['metrics']['phase_runtime']['global_step']
    checks['amp_retry_exact_full_epoch_raw_ema_optimizer_scheduler_scaler_rng']=True
    broken=construct();before=copy.deepcopy(broken[0].state_dict());before_schedule=copy.deepcopy(broken[2].state_dict())
    hook=broken[0].linear.weight.register_hook(lambda grad:torch.full_like(grad,float('inf')))
    try:execute('overflow_persistent',broken)
    except FloatingPointError as exc:assert 'Persistent' in str(exc)
    else:raise AssertionError('Persistent overflow accepted')
    hook.remove();failed_root=root/'overflow_persistent/phase_engine_diagnostics/toy'
    assert read(failed_root/'progress.json')['optimizer_updates']==0
    assert not list(failed_root.glob('epoch_*')) and not broken[1].state
    assert equal(before,broken[0].state_dict()) and equal(before_schedule,broken[2].state_dict())
    assert len((failed_root/'amp_retries.jsonl').read_text().splitlines())==8
    checks['persistent_amp_failure_bounded_no_optimizer_cursor_or_epoch_advance']=True
    # Epoch-aware preparation is outside the retry closure and reconstructs
    # exactly from checkpointed epoch/step/cursor, without new sampler state.
    class Preparation:
        contract=dict(name='deterministic-toy-preparation-v1')
        def __init__(self):self.seen=[]
        def __call__(self,batch,context):
            self.seen.append(dict(context))
            return dict(batch,prepared_scale=1+context['epoch']/10+context['optimizer_update']/100)
    class PreparedLoss:
        contract=dict(name='prepared-toy-loss-v1')
        def __call__(self,m,batch):return loss(m,batch)*batch['prepared_scale']
    prep_full=Preparation();execute('prepared_full',construct(),compute_loss=PreparedLoss(),batch_preparer=prep_full)
    prep_before=Preparation()
    try:execute('prepared_resume',construct(),compute_loss=PreparedLoss(),batch_preparer=prep_before,interrupt_after=2)
    except RuntimeError as exc:assert 'Injected' in str(exc)
    prep_after=Preparation();execute('prepared_resume',construct(19),compute_loss=PreparedLoss(),batch_preparer=prep_after)
    assert prep_full.seen==prep_before.seen+prep_after.seen
    assert all(equal(state('prepared_full')[k],state('prepared_resume')[k]) for k in fields)
    assert {r['epoch'] for r in prep_full.seen}=={1,2}
    assert [r['optimizer_update'] for r in prep_full.seen]==list(range(1,len(prep_full.seen)+1))
    checks['epoch_aware_preparation_and_complete_resume_exact']=True
    low=construct();low=list(low);low[3]=torch.amp.GradScaler('cpu',init_scale=8.,growth_interval=2)
    prep_low=Preparation();execute('prepared_low',low,compute_loss=PreparedLoss(),batch_preparer=prep_low)
    retried=construct();fired.clear();hook=retried[0].linear.weight.register_hook(overflow_once)
    prep_retry=Preparation();execute('prepared_retry',retried,compute_loss=PreparedLoss(),batch_preparer=prep_retry);hook.remove()
    assert prep_low.seen==prep_retry.seen
    assert all(equal(state('prepared_low')[k],state('prepared_retry')[k]) for k in fields)
    checks['amp_retry_reuses_prepared_batch_once']=True
    changed=Preparation();changed.contract=dict(name='changed-preparation')
    try:execute('prepared_full',construct(),compute_loss=PreparedLoss(),batch_preparer=changed)
    except ValueError as exc:assert 'protocol mutation' in str(exc)
    else:raise AssertionError('Changed preparation accepted')
    changed_loss=PreparedLoss();changed_loss.contract=dict(name='changed-loss')
    try:execute('prepared_full',construct(),compute_loss=changed_loss,batch_preparer=Preparation())
    except ValueError as exc:assert 'protocol mutation' in str(exc)
    else:raise AssertionError('Changed objective contract accepted')
    checks['preparation_and_objective_contract_mutation_refused']=True
    # Tamper only diagnostic artifacts; real checkpoints are never touched.
    with (folder/'epoch_0002/checkpoint.pt').open('ab') as f:f.write(b'diagnostic-corruption')
    try:execute('validation_retry',construct())
    except ValueError as exc:assert 'hash mismatch' in str(exc)
    else:raise AssertionError('Corrupt checkpoint accepted')
    checks['corrupt_checkpoint_refused']=True
    result=dict(created_at=now(),passed=True,device='cpu',checks=checks,root=str(root),
                engine_sha256=sha(Path(__file__).with_name('phase_engine.py')),
                limitation='Toy CPU full-pass engine checks only. No new actual CAPRI phase, GPU readiness, specialist sampling, teacher selection or font training claimed.')
    write(Path(run)/'phase_engine_checks.json',result);return result
