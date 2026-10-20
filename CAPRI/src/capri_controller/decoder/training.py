"""Real, resumable CAPRI training. The first stage learns count and localization.

No real targets, calibration/test reads, or controller/reader model writes.
Later joint/specialist stages have their own frozen protocols and checkpoints.
"""
import copy
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import ROOT, read, write, sha, now
from .model import CapriDecoder, align_patch_grid
from .data import CorpusDataset, ShapeBatchPlan, exact_shape_collate
from .objectives import instance_losses
from .checkpoints import save_epoch, load_resume, rng_state, restore_rng
from .metrics import count_statistics, localization_statistics

SEED=731
BATCH_SIZE=8
LOCAL_WEIGHTS=dict(count_ce=2.,objectness=1.,box_l1=5.,box_giou=2.,mask_bce=1.,mask_dice=1.,
                   cardinality=.5,count_consistency=.25,visibility=.25)
LOCAL_OUTPUTS=('presence_logits','boxes','mask_logits','count_logits','visibility_logits')
SECONDARY=('font_reduce.','font_spatial.','font_project.','font_style.','font_attributes.',
           'effect_presence.','effect_intensity.','composition.','rows.','character_colour.','background_under.')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def sources():
    here=Path(__file__).parent
    config=read(ROOT/'configs/local.json')
    result={str(p.relative_to(ROOT)):sha(p) for name in ('training.py','model.py','attention.py','appearance.py',
                'data.py','objectives.py','checkpoints.py','metrics.py') for p in [here/name]}
    vendor=Path(config['decoder_dinov2_source'])
    result.update({str(p.relative_to(ROOT)):sha(p) for p in vendor.rglob('*.py')})
    return result


def initialize():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False


def construct():
    c=read(ROOT/'configs/local.json')
    if c.get('decoder_backbone')!='dinov2_vits14_reg':raise ValueError('Explicit DINOv2 configuration required')
    m=CapriDecoder(c['decoder_dinov2_source'],c['decoder_dinov2_checkpoint'],backbone=c['decoder_backbone']).cuda()
    # These heads remain present and saved, but receive no artificial warmup loss.
    for name,p in m.named_parameters():
        if name.startswith(SECONDARY):p.requires_grad_(False)
    m.train()
    return m


def optimizer(model,total_steps):
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,foreach=False)
    warmup=min(500,max(1,total_steps//20))
    def rate(step):
        if step<warmup:return (step+1)/warmup
        progress=min(1.,(step-warmup)/max(1,total_steps-warmup))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,rate)
    scaler=torch.amp.GradScaler('cuda',init_scale=16.)
    return opt,sched,scaler


def objective(pred,targets,weights):
    selected={k:pred[k] for k in LOCAL_OUTPUTS if k in pred}
    losses,_=instance_losses(selected,targets)
    if set(losses)!=set(weights):raise ValueError('Localization loss contract mismatch')
    loss=sum(losses[k]*weights[k] for k in weights)
    for auxiliary in pred['auxiliary']:
        more,_=instance_losses(auxiliary,targets)
        loss=loss+.25*sum(more[k]*weights[k] for k in more)
    return loss,losses


@torch.no_grad()
def update_ema(model,ema,step):
    decay=min(.999,1-1/(step+1))
    for k,v in model.state_dict().items():
        if v.is_floating_point():ema[k].lerp_(v,1-decay)
        else:ema[k].copy_(v)


def update(model,opt,sched,scaler,ema,batch,weights,step):
    inputs={k:v.cuda(non_blocking=True) for k,v in batch['inputs'].items()}
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
        pred=model(**inputs);loss,parts=objective(pred,batch['targets'],weights)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    scaler.scale(loss).backward();scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    previous=scaler.get_scale();scaler.step(opt);scaler.update()
    if scaler.get_scale()<previous:raise FloatingPointError('AMP skipped optimizer update; not counted as progress')
    sched.step();update_ema(model,ema,step)
    counts=torch.tensor([int(t['count']) for t in batch['targets']],device='cuda')
    return dict(loss=float(loss.detach()),gradient_norm=float(norm),n=len(counts),
                count_correct=int(((pred['count_logits'].argmax(-1)+2)==counts).sum()),
                tasks={k:float(v.detach()) for k,v in parts.items()},lr=opt.param_groups[0]['lr'])


def loader(dataset,plan,workers):
    # Dedicated generator: loader construction/prefetch cannot change model RNG.
    generator=torch.Generator().manual_seed(SEED+plan.state['epoch'])
    kwargs=dict(dataset=dataset,batch_sampler=plan,collate_fn=exact_shape_collate,num_workers=workers,
                pin_memory=True,generator=generator)
    if workers:kwargs.update(prefetch_factor=2,persistent_workers=False)
    return DataLoader(**kwargs)


def report(root,**values):
    value=dict(updated_at=now(),pid=os.getpid(),**values)
    write(root/'progress.json',value)
    with (root/'progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


@torch.no_grad()
def evaluate(model,ema,dataset,workers,root):
    raw={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    preserved_rng=rng_state()
    model.load_state_dict(ema,strict=True);model.eval()
    logits=[];counts=[];presence=[];boxes=[];truth=[]
    plan=ShapeBatchPlan(dataset,BATCH_SIZE,epoch=0,shuffle=False)
    try:
        for i,batch in enumerate(loader(dataset,plan,workers)):
            inputs={k:v.cuda(non_blocking=True) for k,v in batch['inputs'].items()}
            with torch.autocast('cuda',dtype=torch.float16):pred=model(**inputs)
            for key in ('count_logits','presence_logits','boxes'):
                if not torch.isfinite(pred[key]).all():raise FloatingPointError('Nonfinite validation output')
            logits.append(pred['count_logits'].float().cpu());presence.append(pred['presence_logits'].float().cpu())
            boxes.append(pred['boxes'].float().cpu());counts.extend(int(t['count']) for t in batch['targets'])
            truth.extend(t['boxes_visible'] for t in batch['targets'])
            if i%100==0:report(root,state='validating',validation_batches=i+1,validation_total=len(plan),images=len(counts),weights='ema')
        if len(counts)!=len(dataset):raise AssertionError('Incomplete validation')
        return dict(weights='ema',split='dev',count=count_statistics(torch.cat(logits),counts,torch.cat(presence),.5),
                    localization=localization_statistics(torch.cat(boxes),torch.cat(presence),truth,.5),
                    thresholds='fixed diagnostic 0.5 presence and 0.5 IoU; not calibrated',snapshot_hash=dataset.snapshot_hash)
    finally:
        model.load_state_dict(raw,strict=True);model.train();restore_rng(preserved_rng)


def check_training(run):
    """Actual pretrained recovery plus real-task update and maximum-canvas capacity."""
    from .model_resume_checks import _check
    recovery=_check(run,pretrained=True)
    gc.collect();torch.cuda.empty_cache();initialize()
    model=construct();dataset=CorpusDataset(run,allow_partial=True,max_shards=1)
    plan=ShapeBatchPlan(dataset,BATCH_SIZE,epoch=1)
    loader_rng=rng_state()
    worker_loader=loader(dataset,plan,2);worker_iterator=iter(worker_loader)
    worker_batch=next(worker_iterator)
    assert [x['stem'] for x in worker_batch['audit']]==[dataset.rows[i]['stem'] for i in plan.batches[0]]
    assert torch.equal(loader_rng['torch'],torch.get_rng_state())
    del worker_iterator,worker_loader,worker_batch
    opt,sched,scaler=optimizer(model,len(plan));ema=copy.deepcopy(model.state_dict())
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();task_rows=[]
    for i,indices in enumerate(plan.batches[:16]):
        batch=exact_shape_collate([dataset[index] for index in indices])
        result=update(model,opt,sched,scaler,ema,batch,LOCAL_WEIGHTS,i+1)
        task_rows.append(result)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    # Analytic physical-coordinate alignment on a linear ramp, including boundaries.
    yy,xx=torch.meshgrid((torch.arange(10)+.5)*14,(torch.arange(55)+.5)*14,indexing='ij')
    ramp=torch.stack((xx,yy))[None].cuda().float()
    aligned=align_patch_grid(ramp,(128,768),14)
    y,x=torch.meshgrid((torch.arange(8)+.5)*16,(torch.arange(48)+.5)*16,indexing='ij')
    torch.testing.assert_close(aligned,torch.stack((x,y))[None].cuda(),atol=1e-4,rtol=1e-5)
    opt.zero_grad(set_to_none=True)
    pixels=torch.rand(BATCH_SIZE,3,128,768,device='cuda');valid=torch.ones(BATCH_SIZE,128,768,device='cuda',dtype=torch.bool)
    with torch.autocast('cuda',dtype=torch.float16):
        pred=model(pixels,valid)
        probe=sum(p['mask_logits'].float().square().mean()+p['count_logits'].float().square().mean()+
                  p['boxes'].float().square().mean()+p['presence_logits'].float().square().mean()
                  for p in [pred]+pred['auxiliary'])
    probe.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/2**30
    if peak>12:raise RuntimeError('Batch exceeds conservative 12 GiB allocation ceiling')
    # Exact trainer state/cursor round-trip, including its LambdaLR and EMA.
    state=dict(plan.state,consumed_batches=16)
    root=run/('training_smoke_'+str(time.time_ns()))
    path=save_epoch(root,1,model,ema,opt,sched,scaler,state,'training-readiness-v1',task_rows[-1])
    restored=load_resume(path,model,opt,sched,scaler,'training-readiness-v1')
    assert list(ShapeBatchPlan.from_state(dataset,restored['sampler_state']))==plan.batches[16:]
    assert all(torch.equal(v.cpu(),restored['raw'][k]) for k,v in model.state_dict().items())
    # Freeze data-derived scaling on TRAIN only, with bounded adjustment to the
    # priority weights. These are initial-scale estimates, not optimized weights.
    means={k:sum(r['tasks'][k] for r in task_rows)/len(task_rows) for k in LOCAL_WEIGHTS}
    weights={k:v/max(.25,min(4.,means[k])) for k,v in LOCAL_WEIGHTS.items()}
    result=dict(created_at=now(),passed=True,backbone=model.resume_configuration(),parameters=model.parameter_counts(),
                batch_size=BATCH_SIZE,train_pilot_batches=len(task_rows),train_pilot_snapshot=dataset.snapshot_hash,
                real_task_images_per_second=sum(r['n'] for r in task_rows)/elapsed,
                real_task_seconds=elapsed,maximum_canvas_peak_allocated_GiB=peak,
                physical_patch_alignment_checked=True,trainer_checkpoint=str(path),pretrained_recovery=recovery['path'],
                initial_task_means=means,priority_weights=LOCAL_WEIGHTS,frozen_weights=weights,sources=sources(),
                limitation='16 TRAIN diagnostic updates reset before actual training; first-stage readiness only. No held-out tuning.')
    write(run/'dinov2_training_readiness.json',result)
    provenance=read(run/'pretrained/dinov2/provenance.json');provenance['strict_load_verified']=True
    provenance['verified_weight_sha256']=model.pretrained_sha256
    write(run/'pretrained/dinov2/provenance.json',provenance)
    dataset.close()
    return result


def train_baseline(run,workers=4):
    """One full 450k-image count/localization warmup epoch, then a saved phase boundary."""
    readiness=read(run/'dinov2_training_readiness.json')
    if not readiness['passed'] or readiness['sources']!=sources():raise ValueError('Current-source training readiness required')
    initialize();root=run/'training/baseline_localization';root.mkdir(parents=True,exist_ok=True)
    report(root,state='loading_verified_datasets')
    train=CorpusDataset(run);dev=CorpusDataset(run,split='dev',purpose='selection')
    model=construct();plan=ShapeBatchPlan(train,BATCH_SIZE,epoch=1)
    protocol=dict(schema='capri-baseline-localization-v1',created_at=readiness['created_at'],
                  phase='baseline_localization',epochs=1,images_per_epoch=len(train),seed=SEED,
                  batch_size=BATCH_SIZE,sampler='all samples exactly once, exact shape, paired units',
                  train_snapshot=train.snapshot_hash,dev_snapshot=dev.snapshot_hash,corpus_hash=train.protocol_hash,
                  model=model.resume_configuration(),sources=sources(),weights=readiness['frozen_weights'],
                  priority='count/localization first; font and effects deferred to joint baseline and specialists',
                  optimizer=dict(name='AdamW',lr=1e-4,weight_decay=.01,gradient_clip=1.,warmup_steps=min(500,max(1,len(plan)//20)),
                                 schedule='cosine to 10% base LR',total_steps=len(plan),amp='float16, initial scale 16'),
                  ema='min(0.999, 1-1/(step+1)); step counts actual optimizer updates',
                  selection='one warmup epoch; final raw/EMA preserved; EMA measured on all 20000 dev images; no final selection yet',
                  recovery='full state after 100 updates and every 5000 updates; preserve all recovery snapshots separately from epoch checkpoints',
                  next='joint font/effect baseline, measured TRAIN TAG specialists, assigned-output distillation, separate intensity phase',
                  reserve_bytes=10*2**30,workers=workers)
    if (root/'protocol.json').exists():
        if read(root/'protocol.json')!=protocol:raise ValueError('Existing phase protocol differs; refusing mutation')
    else:write(root/'protocol.json',protocol)
    protocol_hash=digest(protocol);opt,sched,scaler=optimizer(model,len(plan));ema=copy.deepcopy(model.state_dict())
    if (root/'latest.json').exists():
        latest=read(root/'latest.json');saved=load_resume(latest['path'],model,opt,sched,scaler,protocol_hash)
        if saved['epoch']==1:
            if not (root/'epoch_0001/validation_ema.json').exists():
                ema={k:v.cuda() for k,v in saved['ema'].items()}
                validation=evaluate(model,ema,dev,workers,root)
                write(root/'epoch_0001/validation_ema.json',validation)
            report(root,state='phase_complete',epoch=1,checkpoint=latest['path'],remaining=protocol['next'])
            return dict(state='phase_complete',epoch=1,checkpoint=latest['path'],remaining=protocol['next'])
        raise ValueError('Unexpected saved epoch')
    total_images=0;correct=0;sum_loss=0.;elapsed_before=0.
    recovery_root=run/'training_recovery/baseline_localization'
    if (recovery_root/'latest.json').exists():
        saved=load_resume(read(recovery_root/'latest.json')['path'],model,opt,sched,scaler,protocol_hash)
        plan=ShapeBatchPlan.from_state(train,saved['sampler_state'])
        ema={k:v.cuda() for k,v in saved['ema'].items()}
        total_images=saved['metrics']['images'];correct=saved['metrics']['correct']
        sum_loss=saved['metrics']['sum_loss'];elapsed_before=saved['metrics']['elapsed_seconds']
        if saved['epoch']!=plan.consumed_batches:raise ValueError('Recovery update/cursor mismatch')
        del saved
    start=time.perf_counter()
    for index,batch in enumerate(loader(train,plan,workers),start=plan.consumed_batches):
        result=update(model,opt,sched,scaler,ema,batch,protocol['weights'],index+1)
        total_images+=result['n'];correct+=result['count_correct'];sum_loss+=result['loss']*result['n']
        plan.consumed_batches=index+1;plan.state['consumed_batches']=index+1
        if index%25==0 or index+1==len(plan.batches):
            elapsed=time.perf_counter()-start+elapsed_before
            report(root,state='training',phase='baseline_localization',epoch=1,epochs=1,batches=index+1,total_batches=len(plan.batches),
                   images=total_images,total_images=len(train),optimizer_updates=index+1,loss=result['loss'],
                   mean_loss=sum_loss/total_images,online_train_count_accuracy=correct/total_images,
                   lr=result['lr'],gradient_norm=result['gradient_norm'],amp_scale=scaler.get_scale(),
                   images_per_second=total_images/elapsed,elapsed_seconds=elapsed,weights='raw',
                   estimated_remaining_epoch_seconds=(len(train)-total_images)*elapsed/total_images)
        if index+1==100 or (index+1)%5000==0:
            recovery_metrics=dict(images=total_images,correct=correct,sum_loss=sum_loss,
                                  elapsed_seconds=time.perf_counter()-start+elapsed_before)
            path=save_epoch(recovery_root,index+1,model,ema,opt,sched,scaler,plan.state,protocol_hash,recovery_metrics)
            print(json.dumps(dict(recovery_saved=str(path),optimizer_updates=index+1,images=total_images)),flush=True)
    if total_images!=len(train):raise AssertionError('Incomplete training epoch')
    training_metrics=dict(images=total_images,optimizer_updates=len(plan.batches),mean_loss=sum_loss/total_images,
                          online_count_accuracy=correct/total_images)
    # Save the full epoch immediately, before validation can fail. Validation is
    # a separate immutable report; it never overwrites a completed checkpoint.
    path=save_epoch(root,1,model,ema,opt,sched,scaler,plan.state,protocol_hash,training_metrics)
    report(root,state='epoch_saved',epoch=1,checkpoint=str(path),images=total_images)
    validation=evaluate(model,ema,dev,workers,root)
    write(root/'epoch_0001/validation_ema.json',validation)
    report(root,state='phase_complete',epoch=1,checkpoint=str(path),validation=validation,remaining=protocol['next'])
    train.close();dev.close()
    return dict(state='phase_complete',epoch=1,checkpoint=str(path),validation=validation,remaining=protocol['next'])
