"""User-authorized six-hour CAPRI pilot, bounded by an eight-hour supervisor."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from torch.utils.data import DataLoader
from .common import ROOT,read,write,sha,now,StageLock
from . import training,pilot_losses
from .data import CorpusDataset,ShapeBatchPlan,exact_shape_collate
from .timed_pilot_data import subset,audit,PilotSubset
from .timed_pilot_eval import evaluate
from .reference_data import TrainReferenceDataset
from .dev_gallery_data import DevGalleryDataset
from .reference_sampling import ReferenceSampler
from .joint_phase import JointPreparation,JointObjective
from .specialist_tasks import task_manifest
from .optimization import finite_update
from .checkpoints import save_epoch,require_finite

PILOT='timed_pilot_20261008'


def report(root,**fields):
    value=dict(updated_at=now(),pid=os.getpid(),**fields);write(root/'progress.json',value)
    with (root/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def fresh_model(checkpoint):
    training.initialize();model=training.construct();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['raw'],strict=True);del saved
    model.requires_grad_(True);model.set_dino_trainable(0);model.train();return model


def objective(families,weights):
    tasks=task_manifest()
    # Pilot priorities are fixed before DEV is read; no DEV-derived scales.
    scales={k:1. for k in tasks}
    return JointObjective(families,tuple(tasks),dict(count=4.,localization=4.,font=1.,appearance=.5,
        composition=.5,effect_presence=1.,effect_intensity=.5),scales,
        dict(instance_threshold=.5,visibility_threshold=.25,library_iou_threshold=.5),.1,.25,
        {k:v for k,v in weights.items() if k!='visibility'})


def run(run):
    import psutil
    root=run/'diagnostics'/PILOT;root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists() or (root/'completion.json').exists():raise ValueError('Existing pilot preserved; no implicit restart')
    for proc in psutil.process_iter(['cmdline']):
        if 'train-baseline' in (proc.info['cmdline'] or []):raise RuntimeError('Old trainer still running')
    authorization=read(root/'authorization.json');started=authorization['started_unix'];train_end=started+5.5*3600
    eval_end=started+7.5*3600
    with StageLock(root,'worker'):
        try:
            report(root,state='preparing',target_hours=6,hard_limit_hours=8)
            checkpoint=Path(read(run/'training_recovery/baseline_localization/latest.json')['path']);receipt=read(checkpoint.parent/'receipt.json')
            if receipt['epoch']!=20000 or sha(checkpoint)!=receipt['sha256']:raise ValueError('Expected verified step-20000 initialization')
            original=read(run/'training/baseline_localization/protocol.json')
            for path,digest in original['sources'].items():
                if sha(ROOT/path)!=digest:raise ValueError('Frozen baseline source changed')
            checks=pilot_losses.check();write(root/'loss_equivalence.json',dict(passed=True,checks=checks))
            train_full=CorpusDataset(run);dev_full=CorpusDataset(run,split='dev',purpose='selection')
            train=subset(train_full,20000,918);dev=subset(dev_full,2000,918)
            train_audit=audit(train,root/'train_subset.json');audit(dev,root/'dev_subset.json')
            if len(train_audit['distributions']['count'])!=7 or len(train_audit['distributions']['effect'])!=37:
                raise ValueError('Pilot subset lacks count/effect coverage')
            refs=TrainReferenceDataset(run);dev_refs=DevGalleryDataset(run)
            family_indices={f:i for f,i in train.family_indices.items() if f in {r['record']['family_group'] for r in refs.rows}}
            sampler=ReferenceSampler(refs,family_indices,918);prepare=JointPreparation(sampler,4,2,'family')
            loss_fn=objective(family_indices,original['weights']);model=fresh_model(checkpoint)
            plan=ShapeBatchPlan(train,8,epoch=1,seed=918)
            ready_batch=exact_shape_collate([train[i] for i in max(plan.batches,key=lambda b:len(b)*train.rows[b[0]]['shape'][0]*train.rows[b[0]]['shape'][1])])
            prepared=prepare(ready_batch,dict(epoch=1,optimizer_update=1,consumed_batches=0))
            # Real-model original/optimized joint losses and parameter gradients
            # must agree before the optimized path is used for this pilot.
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16):old=loss_fn(model,prepared)
            old.backward();torch.cuda.synchronize();original_seconds=time.perf_counter()-start
            old_grads={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
            require_finite(old_grads)
            old_value=float(old.detach());model.zero_grad(set_to_none=True);del old
            pilot_losses.install();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16):new=loss_fn(model,prepared)
            new.backward();torch.cuda.synchronize();optimized_seconds=time.perf_counter()-start
            torch.testing.assert_close(torch.tensor(float(new.detach())),torch.tensor(old_value),atol=2e-4,rtol=2e-4)
            max_abs=0.;max_relative_l2=0.
            for k,p in model.named_parameters():
                if k in old_grads:
                    require_finite(p.grad,k)
                    difference=(p.grad.detach().cpu()-old_grads[k]).float()
                    max_abs=max(max_abs,float(difference.abs().max()))
                    relative=float(difference.norm()/old_grads[k].float().norm().clamp_min(1e-5));max_relative_l2=max(max_relative_l2,relative)
                    if relative>.03 and float(difference.abs().max())>.003:raise ValueError('Material gradient mismatch: '+k)
            peak=torch.cuda.max_memory_allocated()/2**30
            if peak>13.5:raise RuntimeError('Pilot exceeds GPU allocation reserve')
            write(root/'gpu_readiness.json',dict(passed=True,original_seconds=original_seconds,optimized_seconds=optimized_seconds,
                loss=old_value,max_absolute_gradient_difference=max_abs,max_relative_gradient_l2=max_relative_l2,
                peak_allocated_GiB=peak,batch_size=8,reference_limit=2,limitation='One maximum-area diagnostic batch; timings are not steady-state throughput.'))
            del new,old_grads,prepared;model.zero_grad(set_to_none=True)
            # A tiny fixed-data learning gate, then reset exactly to initial raw
            # weights. This probe does not count as pilot epochs.
            tiny_batches=[exact_shape_collate([train[i] for i in ids]) for ids in plan.batches[:4]]
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,foreach=True)
            scaler=torch.amp.GradScaler('cuda',init_scale=16.);losses=[]
            report(root,state='tiny_learning_gate',steps=96)
            for step in range(96):
                batch=tiny_batches[step%4]
                def forward():
                    inputs={k:v.cuda() for k,v in batch['inputs'].items()}
                    with torch.autocast('cuda',dtype=torch.float16):return training.objective(model(**inputs),batch['targets'],original['weights'])[0]
                value,_,_=finite_update(model,opt,scaler,forward);losses.append(float(value))
            before=sum(losses[:16])/16;after=sum(losses[-16:])/16;gate=dict(initial_mean_loss=before,final_mean_loss=after,relative_improvement=1-after/before,
                passed=after<before*.9,steps=96,images=sum(len(x['targets']) for x in tiny_batches),reset_before_pilot=True)
            write(root/'tiny_learning_gate.json',gate)
            del opt,scaler,tiny_batches,model;torch.cuda.empty_cache()
            if not gate['passed']:
                write(root/'completion.json',dict(state='failed_learning_gate',gate=gate,completed_at=now()));report(root,state='failed_learning_gate');return
            model=fresh_model(checkpoint);pilot_losses.install()
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=.01,foreach=True)
            max_epochs=8;total_steps=sum(len(ShapeBatchPlan(train,8,epoch=e,seed=918)) for e in range(1,max_epochs+1))
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(1.,(step+1)/100)*(.2+.8*.5*(1+math.cos(math.pi*min(step,total_steps)/total_steps))))
            scaler=torch.amp.GradScaler('cuda',init_scale=16.);ema=copy.deepcopy(model.state_dict())
            protocol=dict(schema='capri-six-hour-pilot-v1',created_at=now(),authorization=authorization,
                initialization=dict(checkpoint=str(checkpoint),sha256=receipt['sha256'],weights='raw',optimizer='new joint phase; warmup optimizer not reused'),
                train=train.snapshot_hash,dev=dev.snapshot_hash,train_images=len(train),dev_images=len(dev),max_epochs=max_epochs,batch_size=8,
                reference=prepare.contract,objective=loss_fn.contract,train_stop_unix=train_end,evaluation_stop_unix=eval_end,
                optimizer=dict(name='AdamW',lr=5e-5,weight_decay=.01,foreach=True,clip=1.,schedule='100-update warmup, cosine to20% over8 pilot epochs'),
                model=model.resume_configuration(),sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
                criteria=dict(count_accuracy_gain=.02,localization_f1_max_drop=.02,font_top1_gain=.01,
                    action='Report candidate promise only if count gains>=2pp, localization F1 drops<=2pp and end-to-end unseen-family font top1 gains>=1pp. Otherwise report mixed/inconclusive; no automatic full run.'),
                recovery_every=500,checkpoints='raw/EMA/full resumable state every epoch and500updates; final partial epoch separately saved',
                scope='Diagnostic joint baseline only; no specialists/distillation/final calibration/test, no active model replacement. Original full run remains stopped.')
            write(root/'protocol.json',protocol);ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
            report(root,state='initial_dev_evaluation')
            initial=evaluate(model,dev,refs,dev_refs,root/'initial_dev.json',eval_end,4)
            step=0;history=[];reason='max_epochs';last_path=None
            for epoch in range(1,max_epochs+1):
                if time.time()>train_end:reason='training_time_budget';break
                model.train();plan=ShapeBatchPlan(train,8,epoch=epoch,seed=918);images=0;loss_sum=0.;epoch_start=time.time()
                generator=torch.Generator().manual_seed(918+epoch)
                loader=DataLoader(train,batch_sampler=plan,collate_fn=exact_shape_collate,num_workers=4,pin_memory=True,prefetch_factor=2,generator=generator)
                complete=True
                for index,batch in enumerate(loader):
                    if time.time()>train_end:complete=False;reason='training_time_budget';break
                    prepared=prepare(batch,dict(epoch=epoch,optimizer_update=step+1,consumed_batches=index))
                    def forward():
                        with torch.autocast('cuda',dtype=torch.float16):return loss_fn(model,prepared)
                    def retry(value):
                        with (root/'amp_retries.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(value,epoch=epoch,step=step+1))+'\n')
                    value,norm,retries=finite_update(model,opt,scaler,forward,retry);scheduler.step();step+=1
                    with torch.no_grad():
                        current=model.state_dict();floating=[k for k,v in current.items() if v.is_floating_point()]
                        torch._foreach_lerp_([ema[k] for k in floating],[current[k] for k in floating],1-min(.999,1-1/(step+1)))
                        for k,v in current.items():
                            if not v.is_floating_point():ema[k].copy_(v)
                    images+=len(batch['targets']);loss_sum+=float(value)*len(batch['targets']);plan.consumed_batches=index+1;plan.state['consumed_batches']=index+1
                    runtime=dict(epoch=epoch,global_step=step,images=images,loss_sum=loss_sum,epoch_complete=False,deadline=train_end)
                    if step%25==1:report(root,state='training',epoch=epoch,step=step,images=images,epoch_images=len(train),loss=float(value),gradient_norm=float(norm),
                        amp_scale=scaler.get_scale(),images_per_second=images/(time.time()-epoch_start),remaining_training_seconds=max(0,train_end-time.time()))
                    if step%500==0:save_epoch(root/'recoveries',step,model,ema,opt,scheduler,scaler,plan.state,ph,dict(pilot_runtime=runtime))
                if not images:break
                runtime=dict(epoch=epoch,global_step=step,images=images,loss_sum=loss_sum,epoch_complete=complete,deadline=train_end)
                folder=root/'epochs' if complete else root/'final_partial'
                last_path=save_epoch(folder,epoch,model,ema,opt,scheduler,scaler,plan.state,ph,dict(pilot_runtime=runtime,mean_loss=loss_sum/images))
                report(root,state='evaluating',epoch=epoch,step=step,checkpoint=str(last_path),epoch_complete=complete)
                raw={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.load_state_dict(ema,strict=True)
                metrics=evaluate(model,dev,refs,dev_refs,root/f'dev_epoch_{epoch:02d}.json',eval_end,4)
                model.load_state_dict(raw,strict=True);del raw
                history.append(dict(epoch=epoch,step=step,epoch_complete=complete,checkpoint=str(last_path),count_accuracy=metrics['count']['accuracy'],
                    localization_f1=metrics['localization']['f1'],font_top1=metrics['font']['end_to_end_top1']))
                write(root/'learning_curve.json',history)
                # Abort demonstrable count/localization regression instead of
                # consuming the whole budget. Fixed guard, no real labels.
                if metrics['count']['accuracy']<initial['count']['accuracy']-.10 or (metrics['localization']['f1'] or 0)<(initial['localization']['f1'] or 0)-.10:
                    reason='quality_regression_guard';break
                if not complete:break
            last=history[-1] if history else None
            promising=bool(last and last['count_accuracy']>=initial['count']['accuracy']+.02 and
                (last['localization_f1'] or 0)>=(initial['localization']['f1'] or 0)-.02 and last['font_top1']>=initial['font']['end_to_end_top1']+.01)
            result=dict(completed_at=now(),state='complete',reason=reason,elapsed_hours=(time.time()-started)/3600,
                completed_epochs=sum(r['epoch_complete'] for r in history),optimizer_updates=step,initial_count_accuracy=initial['count']['accuracy'],
                initial_localization_f1=initial['localization']['f1'],initial_font_top1=initial['font']['end_to_end_top1'],last=last,
                verdict='promising_pilot_requires_larger_validation' if promising else 'mixed_or_inconclusive_do_not_scale_automatically',
                training_continues=False,limitation='Small synthetic development run, no final test or real-target performance claim; no automatic full training.')
            write(root/'completion.json',result);report(root,**result)
            train.close();dev.close();refs.close();dev_refs.close()
        except Exception as exc:
            report(root,state='failed',error=repr(exc));raise


def supervise(run):
    """Independent wall-clock cap, including setup, training and evaluation."""
    import psutil
    root=run/'diagnostics'/PILOT;root.mkdir(parents=True,exist_ok=True)
    with StageLock(root,'supervisor'):
        if (root/'authorization.json').exists():raise ValueError('Existing timed pilot preserved; inspect before retry')
        started=time.time();write(root/'authorization.json',dict(created_at=now(),started_unix=started,target_hours=6,hard_hours=8,
            instruction='User: Do a dummy run that finished in6/8hrs',supersedes_stop='This bounded pilot only. Original full training and general heartbeat stay stopped.'))
        with (root/'stdout.log').open('w') as out,(root/'stderr.log').open('w') as err:
            child=subprocess.Popen([sys.executable,'-B',str(ROOT/'capri.py'),'decoder','timed-pilot-worker','--run',str(run)],cwd=ROOT,
                stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            proc=psutil.Process(child.pid)
            write(root/'supervisor.json',dict(pid=os.getpid(),worker_pid=child.pid,worker_created=proc.create_time(),started_unix=started,hard_deadline_unix=started+8*3600))
            while child.poll() is None:
                if time.time()>started+8*3600 or (root/'STOP').exists():
                    processes=proc.children(recursive=True)+[proc]
                    for p in processes:
                        try:p.terminate()
                        except psutil.NoSuchProcess:pass
                    _,alive=psutil.wait_procs(processes,timeout=5)
                    for p in alive:p.kill()
                    write(root/'hard_stop.json',dict(stopped_at=now(),reason='eight_hour_cap_or_stop_file',saved_checkpoints_preserved=True));break
                time.sleep(5)
            write(root/'supervisor_result.json',dict(ended_at=now(),exit_code=child.wait(),elapsed_hours=(time.time()-started)/3600))
