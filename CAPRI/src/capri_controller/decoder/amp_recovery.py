"""Explicit runtime amendment for bounded same-batch AMP overflow recovery.

Original phase code/protocol and all checkpoints remain immutable. No failed
update advances the optimizer, scheduler, EMA or consumed-sample cursor.
"""
import copy
import json
from pathlib import Path
import torch
from .common import read,write,sha,now,ROOT
from .checkpoints import rng_state,restore_rng

MAX_ATTEMPTS=9


def safe_step(model,opt,sched,scaler,ema,batch,weights,step,objective,update_ema,on_retry=None):
    device=next(model.parameters()).device
    inputs={k:v.to(device,non_blocking=True) for k,v in batch['inputs'].items()}
    before_rng=rng_state();buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            restore_rng(before_rng)
            with torch.no_grad():
                for k,v in model.named_buffers():v.copy_(buffers[k])
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            pred=model(**inputs);loss,parts=objective(pred,batch['targets'],weights)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite forward loss; not treated as AMP overflow')
        scaler.scale(loss).backward();scaler.unscale_(opt)
        try:
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        except RuntimeError as exc:
            if 'non-finite' not in str(exc):raise
            invalid=[name for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            if not invalid:raise FloatingPointError('Nonfinite gradient norm with finite gradient elements') from exc
            previous=scaler.get_scale();reduced=previous*.5
            if attempt+1==MAX_ATTEMPTS or reduced<1e-8:
                raise FloatingPointError('Persistent nonfinite gradients after bounded AMP retries: '+','.join(invalid[:8])) from exc
            # Explicit scale update resets per-optimizer unscale/inf bookkeeping.
            scaler.update(new_scale=reduced)
            state=scaler.state_dict();state['_growth_tracker']=0;scaler.load_state_dict(state)
            if on_retry:on_retry(dict(created_at=now(),optimizer_update=step,attempt=attempt+1,
                previous_scale=previous,retry_scale=reduced,nonfinite_parameters=invalid,
                batch_stems=[r.get('stem') for r in batch.get('audit',[])],same_batch=True))
            continue
        previous=scaler.get_scale();scaler.step(opt);scaler.update()
        if scaler.get_scale()<previous:raise FloatingPointError('Unexpected AMP skip after finite gradients')
        sched.step();update_ema(model,ema,step)
        counts=torch.tensor([int(t['count']) for t in batch['targets']],device=device)
        return dict(loss=float(loss.detach()),gradient_norm=float(norm),n=len(counts),
                    count_correct=int(((pred['count_logits'].argmax(-1)+2)==counts).sum()),
                    tasks={k:float(v.detach()) for k,v in parts.items()},lr=opt.param_groups[0]['lr'],amp_retries=attempt)
    raise AssertionError('Unreachable AMP retry state')


def resume(run,workers=4):
    from . import training
    amendment=read(run/'amp_recovery/amendment.json')
    if not amendment['passed'] or amendment['recovery_source_sha256']!=sha(__file__):raise ValueError('Unverified AMP recovery amendment')
    if amendment['original_sources']!=training.sources() or amendment['original_protocol_sha256']!=sha(run/'training/baseline_localization/protocol.json'):
        raise ValueError('Original training code/protocol differs from AMP amendment')
    for path,digest in amendment['original_sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen baseline source changed')
    identity=dict(amendment_sha256=sha(run/'amp_recovery/amendment.json'),source_sha256=sha(__file__),
                  policy='same batch/RNG, halve scale/reset growth tracker, maximum nine attempts; no skipped optimizer updates')
    def retry_log(row):
        with (run/'amp_recovery/retries.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(amp_retry=row)),flush=True)
    original_update=training.update;original_save=training.save_epoch;original_load=training.load_resume
    def amended_update(model,opt,sched,scaler,ema,batch,weights,step):
        return safe_step(model,opt,sched,scaler,ema,batch,weights,step,training.objective,training.update_ema,retry_log)
    def amended_save(root,epoch,model,ema,opt,sched,scaler,sampler,protocol_hash,metrics,**kwargs):
        return original_save(root,epoch,model,ema,opt,sched,scaler,sampler,protocol_hash,
                             dict(metrics,execution_amendment=identity),**kwargs)
    def amended_load(*args,**kwargs):
        restored=original_load(*args,**kwargs);prior=restored['metrics'].get('execution_amendment')
        if prior is not None and prior!=identity:raise ValueError('Checkpoint AMP recovery identity mismatch')
        if prior is None and restored['sampler_state']['consumed_batches']>amendment['starting_update']:
            raise ValueError('Later checkpoint missing AMP amendment identity')
        return restored
    training.update=amended_update;training.save_epoch=amended_save;training.load_resume=amended_load
    write(run/'amp_recovery/active.json',dict(started_at=now(),identity=identity))
    try:return training.train_baseline(run,workers)
    except Exception as exc:
        training.report(run/'training/baseline_localization',state='failed',error=repr(exc),execution_amendment=identity)
        raise
    finally:training.update=original_update;training.save_epoch=original_save;training.load_resume=original_load
