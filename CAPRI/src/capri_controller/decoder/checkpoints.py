"""One atomic full checkpoint per epoch, with receipts and exact RNG continuation.

Each checkpoint contains raw and EMA weights plus optimizer/scheduler/scaler,
sampler and RNG states. A failed/incomplete write never becomes a saved epoch.
No epoch pruning is performed by this module.
"""
import copy
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .common import write, read, sha, now


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def execution_state(model):
    return dict(trainable_parameters=[name for name,p in model.named_parameters() if p.requires_grad],
                module_training={name:module.training for name,module in model.named_modules()},
                model_configuration=model.resume_configuration() if hasattr(model,'resume_configuration') else None,
                precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                               cudnn_tf32=torch.backends.cudnn.allow_tf32,
                               cudnn_benchmark=torch.backends.cudnn.benchmark,
                               cudnn_deterministic=torch.backends.cudnn.deterministic,
                               float32_matmul=torch.get_float32_matmul_precision(),
                               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                               deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled()))


def validate_execution(saved,model):
    current=execution_state(model)
    if saved['trainable_parameters']!=current['trainable_parameters']:
        raise ValueError('Resume trainable-parameter policy mismatch; construct matching optimizer groups first')
    if saved['model_configuration']!=current['model_configuration']:
        raise ValueError('Resume model configuration/pretraining provenance mismatch')
    if set(saved['module_training'])!=set(current['module_training']):
        raise ValueError('Resume module layout mismatch')


def restore_execution(saved,model):
    for name,module in model.named_modules():module.training=saved['module_training'][name]
    p=saved['precision']
    torch.set_float32_matmul_precision(p['float32_matmul'])
    torch.backends.cuda.matmul.allow_tf32=p['matmul_tf32']
    torch.backends.cudnn.allow_tf32=p['cudnn_tf32']
    torch.backends.cudnn.benchmark=p['cudnn_benchmark']
    torch.backends.cudnn.deterministic=p['cudnn_deterministic']
    torch.use_deterministic_algorithms(p['deterministic_algorithms'],warn_only=p['deterministic_warn_only'])


def optimizer_layout(model,optimizer):
    names={id(p):name for name,p in model.named_parameters()}
    groups=[[names.get(id(p)) for p in group['params']] for group in optimizer.param_groups]
    flat=[name for group in groups for name in group]
    if None in flat or len(flat)!=len(set(flat)):
        raise ValueError('Optimizer parameters must be unique parameters of this model')
    return groups


def require_finite(value,path='state'):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError('Nonfinite checkpoint tensor: '+path)
    elif isinstance(value,dict):
        for key,item in value.items():require_finite(item,path+'/'+str(key))
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):require_finite(item,path+'/'+str(index))
    elif isinstance(value,float) and not np.isfinite(value):
        raise ValueError('Nonfinite checkpoint scalar: '+path)


def save_epoch(root,epoch,model,ema,optimizer,scheduler,scaler,sampler_state,protocol_hash,metrics,reserve_bytes=10*2**30):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    destination=root/f'epoch_{epoch:04d}'
    if destination.exists():
        raise FileExistsError('Preserving completed epoch: '+str(destination))
    raw=model.state_dict()
    ema_state=ema.state_dict() if hasattr(ema,'state_dict') else ema
    if set(raw)!=set(ema_state) or any(raw[k].shape!=ema_state[k].shape for k in raw):
        raise ValueError('Raw/EMA architecture mismatch')
    opt_state=optimizer.state_dict()
    require_finite(dict(raw=raw,ema=ema_state,optimizer=opt_state,scaler=scaler.state_dict() if scaler else None,
                        scheduler=scheduler.state_dict() if scheduler else None,metrics=metrics))
    estimated=6*sum(t.numel()*t.element_size() for t in raw.values())
    if shutil.disk_usage(root).free < estimated+reserve_bytes:
        raise OSError('Insufficient checkpoint space; existing epochs are preserved')
    temporary=root/f'epoch_{epoch:04d}.writing.{os.getpid()}.{time.time_ns()}'
    temporary.mkdir()
    state=dict(schema='capri-decoder-resume-v2',epoch=epoch,raw=raw,ema=ema_state,
               optimizer=opt_state,scheduler=scheduler.state_dict() if scheduler else None,
               scaler=scaler.state_dict() if scaler else None,sampler_state=sampler_state,rng=rng_state(),
               protocol_hash=protocol_hash,metrics=metrics,created_at=now(),execution=execution_state(model),
               optimizer_layout=optimizer_layout(model,optimizer))
    path=temporary/'checkpoint.pt'
    with path.open('wb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    digest=sha(path)
    # Files produced here are trusted local training state, not downloaded pickles.
    checked=torch.load(path,map_location='cpu',weights_only=False)
    assert checked['epoch']==epoch and checked['protocol_hash']==protocol_hash
    assert set(checked['raw'])==set(raw) and set(checked['ema'])==set(ema_state)
    receipt=dict(epoch=epoch,sha256=digest,bytes=path.stat().st_size,protocol_hash=protocol_hash,
                 contains=['raw','ema','optimizer','scheduler','scaler','sampler_state','rng','metrics','execution','optimizer_layout'],verified_at=now())
    write(temporary/'receipt.json',receipt)
    temporary.rename(destination)
    write(root/'latest.json',dict(epoch=epoch,path=str(destination/'checkpoint.pt'),sha256=digest))
    return destination/'checkpoint.pt'


def load_resume(path,model,optimizer,scheduler=None,scaler=None,expected_protocol_hash=None):
    path=Path(path)
    receipt=read(path.parent/'receipt.json')
    if sha(path)!=receipt['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if expected_protocol_hash is not None and state['protocol_hash']!=expected_protocol_hash:
        raise ValueError('Resume protocol mismatch')
    if state['schema'] not in ('capri-decoder-resume-v1','capri-decoder-resume-v2'):
        raise ValueError('Unsupported checkpoint schema')
    if (scheduler is None)!=(state['scheduler'] is None) or (scaler is None)!=(state['scaler'] is None):
        raise ValueError('Resume scheduler/scaler presence mismatch')
    if state['schema']=='capri-decoder-resume-v2':
        validate_execution(state['execution'],model)
        if state['optimizer_layout']!=optimizer_layout(model,optimizer):
            raise ValueError('Resume optimizer parameter order/group mismatch')
    model.load_state_dict(state['raw'],strict=True);optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    if scaler is not None:
        scaler.load_state_dict(state['scaler'])
    if state['schema']=='capri-decoder-resume-v2':restore_execution(state['execution'],model)
    restore_rng(state['rng'])
    return state


def check_resume(run):
    dest=Path(run)/('checkpoint_smoke_'+str(time.time_ns()))
    torch.manual_seed(731);np.random.seed(731);random.seed(731)
    model=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.GELU(),torch.nn.Linear(8,2))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.StepLR(opt,1,gamma=.9)
    def step(m,o,s):
        x=torch.randn(7,5);target=torch.randn(7,2)
        loss=(m(x)-target).square().mean();o.zero_grad();loss.backward();o.step();s.step()
        return float(loss)
    step(model,opt,sched)
    ema=copy.deepcopy(model.state_dict())
    path=save_epoch(dest,1,model,ema,opt,sched,None,dict(epoch=2,cursor=0,seed=731),'smoke-v1',{},reserve_bytes=0)
    loss1=step(model,opt,sched)
    reference=copy.deepcopy(model.state_dict())
    model2=copy.deepcopy(model);opt2=torch.optim.AdamW(model2.parameters(),lr=.001)
    sched2=torch.optim.lr_scheduler.StepLR(opt2,1,gamma=.9)
    state=load_resume(path,model2,opt2,sched2,expected_protocol_hash='smoke-v1')
    loss2=step(model2,opt2,sched2)
    assert loss1==loss2 and all(torch.equal(reference[k],v) for k,v in model2.state_dict().items())
    assert state['sampler_state']==dict(epoch=2,cursor=0,seed=731)
    assert all(torch.equal(ema[k],state['ema'][k]) for k in ema)
    try:
        save_epoch(dest,1,model,ema,opt,sched,None,{},'smoke-v1',{},reserve_bytes=0)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Completed epoch overwritten')
    result=dict(passed=True,exact_cpu_resume=True,epoch_overwrite_refused=True,raw_and_ema_preserved=True,
                path=str(path),limitation='CPU infrastructure test; CUDA/scaler and actual decoder sampler must also be tested')
    write(Path(run)/'checkpoint_checks.json',result)
    return result


def check_cuda_resume(run):
    """Tiny CUDA/AMP infrastructure diagnostic, never a CAPRI training run."""
    from .data import CorpusDataset,ShapeBatchPlan
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.manual_seed(977);np.random.seed(977);random.seed(977)
    torch.backends.cuda.matmul.allow_tf32=False
    model=torch.nn.Sequential(torch.nn.Linear(5,16),torch.nn.GELU(),torch.nn.Linear(16,2)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.StepLR(opt,1,gamma=.9)
    scaler=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=2)
    def step(m,o,s,scale):
        x=torch.randn(7,5,device='cuda');target=torch.randn(7,2,device='cuda')
        o.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=(m(x)-target).square().mean()
        scale.scale(loss).backward();scale.step(o);scale.update();s.step()
        return float(loss)
    step(model,opt,sched,scaler)
    ds=CorpusDataset(run,allow_partial=True,max_shards=1)
    plan=ShapeBatchPlan(ds,epoch=3);sampler=copy.deepcopy(plan.state);sampler['consumed_batches']=17
    ema=copy.deepcopy(model.state_dict())
    dest=Path(run)/('checkpoint_cuda_smoke_'+str(time.time_ns()))
    path=save_epoch(dest,1,model,ema,opt,sched,scaler,sampler,'cuda-amp-smoke-v1',{},reserve_bytes=0)
    expected_loss=step(model,opt,sched,scaler)
    expected_weights=copy.deepcopy(model.state_dict());expected_scale=scaler.state_dict()
    m2=copy.deepcopy(model);o2=torch.optim.AdamW(m2.parameters(),lr=.001)
    s2=torch.optim.lr_scheduler.StepLR(o2,1,gamma=.9);scale2=torch.amp.GradScaler('cuda')
    state=load_resume(path,m2,o2,s2,scale2,expected_protocol_hash='cuda-amp-smoke-v1')
    restored_batches=list(ShapeBatchPlan.from_state(ds,state['sampler_state']))
    assert restored_batches==plan.batches[17:]
    actual_loss=step(m2,o2,s2,scale2)
    assert expected_loss==actual_loss
    assert all(torch.equal(expected_weights[k],v) for k,v in m2.state_dict().items())
    assert expected_scale==scale2.state_dict()
    assert all(torch.equal(ema[k].cpu(),state['ema'][k]) for k in ema)
    bad_ema=copy.deepcopy(ema);next(iter(bad_ema.values())).flatten()[0]=float('nan')
    try:save_epoch(dest,2,model,bad_ema,opt,sched,scaler,sampler,'cuda-amp-smoke-v1',{},reserve_bytes=0)
    except ValueError as exc:assert 'Nonfinite' in str(exc)
    else:raise AssertionError('Nonfinite EMA accepted')
    optimizer_tensor=next(iter(opt.state.values()))['exp_avg']
    optimizer_tensor.flatten()[0]=float('nan')
    try:save_epoch(dest,2,model,ema,opt,sched,scaler,sampler,'cuda-amp-smoke-v1',{},reserve_bytes=0)
    except ValueError as exc:assert 'Nonfinite' in str(exc)
    else:raise AssertionError('Nonfinite optimizer accepted')
    assert not (dest/'epoch_0002').exists()
    ds.close()
    result=dict(created_at=now(),passed=True,exact_cuda_amp_continuation=True,
                exact_batch_plan_resume=True,raw_ema_scaler_rng_restored=True,
                nonfinite_ema_and_optimizer_refused=True,path=str(path),device=torch.cuda.get_device_name(),
                limitation='Tiny random-tensor infrastructure test. Actual decoder gradients/matching/AMP/memory and end-to-end continuation still require its implementation and approved pretrained weights.')
    write(Path(run)/'checkpoint_cuda_checks.json',result)
    return result
