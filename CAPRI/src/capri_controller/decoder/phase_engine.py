"""Resumable multi-epoch engine for subsequent decoder phases.

Independent of the immutable, already-running localization trainer. Callers
provide the initialized model, phase objective, optimizer, and DEV evaluator.
"""
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .common import ROOT,read,write,sha,now,run_path
from .data import ShapeBatchPlan,exact_shape_collate
from .checkpoints import (save_epoch,load_resume,require_finite,rng_state,restore_rng,
                          execution_state,restore_execution)
from .optimization import finite_update,MAX_ATTEMPTS


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def move(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    return value


def _latest(root,recovery,protocol_hash):
    """Recover from published checkpoint pointers, including a failed validation."""
    found=[]
    for folder,complete in ((root,True),(recovery,False)):
        if not (folder/'latest.json').exists():continue
        pointer=read(folder/'latest.json');path=Path(pointer['path'])
        if not path.resolve().is_relative_to(folder.resolve()):raise ValueError('Checkpoint pointer escapes phase')
        receipt=read(path.parent/'receipt.json')
        if sha(path)!=pointer['sha256'] or pointer['sha256']!=receipt['sha256']:raise ValueError('Phase checkpoint hash mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['protocol_hash']!=protocol_hash:raise ValueError('Phase resume protocol mismatch')
        progress=state['metrics']['phase_runtime']
        if progress['epoch_complete']!=complete:raise ValueError('Recovery/epoch checkpoint kind mismatch')
        expected_number=progress['epoch'] if complete else progress['global_step']
        if state['epoch']!=expected_number or receipt['epoch']!=expected_number or pointer['epoch']!=expected_number:
            raise ValueError('Checkpoint runtime/number mismatch')
        found.append((progress['global_step'],complete,path,progress))
        del state
    return max(found,key=lambda row:(row[0],row[1])) if found else None


def _validation(root,epoch,model,ema,evaluate):
    destination=root/f'epoch_{epoch:04d}'/'validation_ema.json'
    checkpoint=read(destination.parent/'receipt.json')
    if destination.exists():
        saved=read(destination)
        if digest(saved['payload'])!=saved['payload_sha256'] or saved['payload']['checkpoint_sha256']!=checkpoint['sha256']:
            raise ValueError('Phase validation identity/hash mismatch')
        return saved['payload']['metrics']
    raw={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    rng=rng_state();execution=execution_state(model)
    try:
        model.load_state_dict(ema,strict=True);model.eval()
        with torch.no_grad():metrics=evaluate(model,epoch)
        require_finite(metrics)
        payload=dict(epoch=epoch,weights='ema',checkpoint_sha256=checkpoint['sha256'],metrics=metrics)
        write(destination,dict(payload=payload,payload_sha256=digest(payload)))
        return metrics
    finally:
        model.load_state_dict(raw,strict=True);restore_execution(execution,model);restore_rng(rng)


def run_phase(run,phase,model,dataset,optimizer,scheduler,scaler,compute_loss,evaluate,protocol,
              epochs,batch_size,seed,workers=4,recovery_every=5000,diagnostic=False,interrupt_after=None,batch_preparer=None):
    """Full-corpus passes; save every epoch before validation, never prune.

    compute_loss(model, batch) returns a scalar. Images and validity are the only
    model inputs; synthetic targets remain in the caller's separate loss path.
    The caller freezes objective/evaluation/source identities and scheduler
    formulas in protocol, and holds a phase-specific StageLock. Dataset reads
    must be deterministic by index: worker RNG is not a data augmentation state.
    interrupt_after is a diagnostic fault-injection hook, forbidden in production.
    Optional batch_preparer(batch, context) runs once before bounded AMP retries.
    It must be deterministic without advancing RNG/sampling cursors, expose a
    JSON contract, and retain the original scene inputs/targets/audit fields.
    """
    run=run_path(run)
    if not re.fullmatch('[a-z0-9_]+',phase):raise ValueError('Safe explicit phase name required')
    if epochs<1 or recovery_every<1 or workers<0 or (interrupt_after is not None and not diagnostic):raise ValueError('Invalid phase limits')
    root=run/('phase_engine_diagnostics' if diagnostic else 'training')/phase
    recovery=run/('phase_engine_recovery_diagnostics' if diagnostic else 'training_recovery')/phase
    root.mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device
    if not diagnostic and (device.type!='cuda' or scaler is None):raise ValueError('Production decoder phases require CUDA/AMP')
    preparation=None if batch_preparer is None else json.loads(json.dumps(batch_preparer.contract,sort_keys=True,allow_nan=False))
    objective_contract=json.loads(json.dumps(getattr(compute_loss,'contract',None),sort_keys=True,allow_nan=False))
    if batch_preparer is not None and (not callable(batch_preparer) or not isinstance(preparation,dict) or not preparation):
        raise ValueError('Batch preparation requires an explicit nonempty contract')
    frozen=dict(schema='capri-phase-engine-v1',phase=phase,diagnostic=diagnostic,epochs=epochs,
                batch_size=batch_size,seed=seed,workers=workers,recovery_every=recovery_every,
                snapshot_hash=dataset.snapshot_hash,images_per_epoch=len(dataset),settings=protocol,
                engine_sha256=sha(__file__),model_configuration=model.resume_configuration(),
                optimization_sha256=sha(Path(__file__).with_name('optimization.py')),
                amp_retry=dict(max_attempts=MAX_ATTEMPTS,same_batch_rng=True,scale_multiplier=.5,growth_tracker_reset=True),
                trainable_parameters=[k for k,p in model.named_parameters() if p.requires_grad],
                optimizer_type=type(optimizer).__module__+'.'+type(optimizer).__qualname__,
                optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
                scheduler_type=None if scheduler is None else type(scheduler).__module__+'.'+type(scheduler).__qualname__,
                initial_scheduler_state=None if scheduler is None else scheduler.state_dict(),
                batch_preparation=preparation,
                objective_contract=objective_contract,
                initial_scaler_state=None if scaler is None else scaler.state_dict())
    # Normalize tuples (e.g. Adam betas) to their persisted JSON representation.
    frozen=json.loads(json.dumps(frozen,sort_keys=True,allow_nan=False))
    if (root/'protocol.json').exists():
        if read(root/'protocol.json')!=frozen:raise ValueError('Refusing phase protocol mutation')
    else:write(root/'protocol.json',frozen)
    protocol_hash=digest(frozen);model.train();ema=copy.deepcopy(model.state_dict())
    step=0;start_epoch=1;cursor=0;images=0;loss_sum=0.
    def report(**fields):write(root/'progress.json',dict(updated_at=now(),phase=phase,pid=os.getpid(),**fields))
    latest=_latest(root,recovery,protocol_hash)
    if latest:
        _,complete,path,progress=latest
        restored=load_resume(path,model,optimizer,scheduler,scaler,protocol_hash)
        ema=move(restored['ema'],device);step=progress['global_step'];start_epoch=progress['epoch']
        plan=ShapeBatchPlan.from_state(dataset,restored['sampler_state'])
        cursor=plan.consumed_batches;images=progress['images'];loss_sum=progress['loss_sum']
        if plan.state['epoch']!=start_epoch or plan.state['seed']!=seed or plan.state['batch_size']!=batch_size:
            raise ValueError('Checkpoint sampler/phase mismatch')
        if not 1<=start_epoch<=epochs or images!=sum(map(len,plan.batches[:cursor])):
            raise ValueError('Checkpoint consumed samples/epoch mismatch')
        for previous in range(1,start_epoch):
            saved=read(root/f'epoch_{previous:04d}'/'validation_ema.json')
            receipt=read(root/f'epoch_{previous:04d}'/'receipt.json')
            if digest(saved['payload'])!=saved['payload_sha256'] or saved['payload']['checkpoint_sha256']!=receipt['sha256']:
                raise ValueError('Prior epoch validation identity/hash mismatch')
        if complete:
            if cursor!=len(plan.batches) or images!=len(dataset):raise ValueError('Incomplete saved epoch')
            try:_validation(root,start_epoch,model,ema,evaluate)
            except Exception as exc:
                report(state='failed_validation',epoch=start_epoch,optimizer_updates=step,error=repr(exc));raise
            start_epoch+=1;cursor=0;images=0;loss_sum=0.
        del restored
    try:
        for epoch in range(start_epoch,epochs+1):
            plan=ShapeBatchPlan(dataset,batch_size,epoch,seed,consumed_batches=cursor)
            generator=torch.Generator().manual_seed(seed+epoch)
            kwargs=dict(dataset=dataset,batch_sampler=plan,collate_fn=exact_shape_collate,num_workers=workers,
                        pin_memory=device.type=='cuda',generator=generator)
            if workers:kwargs['prefetch_factor']=2
            for index,batch in enumerate(DataLoader(**kwargs),start=cursor):
                if digest(getattr(compute_loss,'contract',None))!=digest(objective_contract):
                    raise ValueError('Objective contract changed during phase')
                if batch_preparer is not None:
                    if digest(batch_preparer.contract)!=digest(preparation):raise ValueError('Batch preparation contract changed')
                    original=batch
                    batch=batch_preparer(batch,dict(epoch=epoch,optimizer_update=step+1,consumed_batches=index))
                    if any(batch.get(k) is not original[k] for k in ('inputs','targets','audit')):
                        raise ValueError('Batch preparation must preserve original scene fields')
                def forward_loss():
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                        return compute_loss(model,batch)
                def retry_log(row):
                    row.update(created_at=now(),epoch=epoch,optimizer_update=step+1,consumed_batches=index,
                               batch_stems=[r.get('stem') for r in batch.get('audit',[])])
                    with (root/'amp_retries.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
                loss,norm,retries=finite_update(model,optimizer,scaler,forward_loss,retry_log)
                if scheduler is not None:scheduler.step()
                step+=1;n=len(batch['inputs']['pixels']);images+=n;loss_sum+=float(loss.detach())*n
                with torch.no_grad():
                    decay=min(.999,1-1/(step+1))
                    for k,v in model.state_dict().items():
                        if v.is_floating_point():ema[k].lerp_(v,1-decay)
                        else:ema[k].copy_(v)
                plan.state['consumed_batches']=index+1;plan.consumed_batches=index+1
                progress=dict(epoch=epoch,global_step=step,images=images,loss_sum=loss_sum,epoch_complete=False)
                if step%25==1 or index+1==len(plan.batches):
                    report(state='training',epoch=epoch,epochs=epochs,optimizer_updates=step,batches=index+1,
                           total_batches=len(plan.batches),images=images,total_images=len(dataset),loss=float(loss.detach()),
                           gradient_norm=float(norm),mean_loss=loss_sum/images,amp_retries=retries,
                           amp_scale=None if scaler is None else scaler.get_scale())
                if step%recovery_every==0:
                    save_epoch(recovery,step,model,ema,optimizer,scheduler,scaler,plan.state,protocol_hash,
                               dict(phase_runtime=progress),reserve_bytes=0 if diagnostic else 10*2**30)
                if interrupt_after==step:raise RuntimeError('Injected diagnostic interruption')
            if images!=len(dataset):raise AssertionError('Full-epoch sample count mismatch')
            progress=dict(epoch=epoch,global_step=step,images=images,loss_sum=loss_sum,epoch_complete=True)
            path=save_epoch(root,epoch,model,ema,optimizer,scheduler,scaler,plan.state,protocol_hash,
                            dict(phase_runtime=progress,mean_loss=loss_sum/images),reserve_bytes=0 if diagnostic else 10*2**30)
            report(state='validating',epoch=epoch,optimizer_updates=step,checkpoint=str(path))
            _validation(root,epoch,model,ema,evaluate)
            cursor=0;images=0;loss_sum=0.
        report(state='phase_complete',epochs=epochs,optimizer_updates=step)
        return dict(state='phase_complete',epochs=epochs,optimizer_updates=step,root=str(root))
    except Exception as exc:
        report(state='failed',optimizer_updates=step,error=repr(exc));raise
