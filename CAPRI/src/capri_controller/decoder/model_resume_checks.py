"""Full-core CUDA checkpoint recovery diagnostic, outside actual training phases."""
import copy
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import read, write, sha, now, ROOT
from .model import CapriDecoder
from .data import CorpusDataset, ShapeBatchPlan, exact_shape_collate
from .objectives import instance_losses
from .checkpoints import save_epoch, load_resume, execution_state, restore_execution, rng_state, restore_rng


def clone_cpu(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:clone_cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [clone_cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(clone_cpu(v) for v in value)
    return copy.deepcopy(value)


def compare_tensors(a,b,atol,rtol,strict=True):
    errors=[]; exact=True; close=True
    def visit(x,y):
        nonlocal exact,close
        if torch.is_tensor(x):
            y=y.detach().cpu()
            if strict:torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
            close=close and torch.allclose(x,y,atol=atol,rtol=rtol)
            exact=exact and torch.equal(x,y)
            if x.numel(): errors.append(float((x.float()-y.float()).abs().max()))
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for key in x:visit(x[key],y[key])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for u,v in zip(x,y):visit(u,v)
        else: assert x==y
    visit(a,b)
    return dict(bit_exact=exact,within_tolerance=close,max_absolute_difference=max(errors,default=0.),atol=atol,rtol=rtol)


def _check(run, pretrained=False):
    config=read(ROOT/'configs/local.json')
    source=config['decoder_dinov2_source' if pretrained else 'decoder_dinov3_source']
    def construct():
        return CapriDecoder(source,checkpoint=config['decoder_dinov2_checkpoint'],backbone='dinov2_vits14_reg') if pretrained else CapriDecoder(source,diagnostic_random=True)
    torch.manual_seed(731);np.random.seed(731);random.seed(731);torch.set_num_threads(4)
    model=construct().cuda()
    original_execution=execution_state(model)
    dataset=None
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model.set_dino_trainable(4);model.train()
        dataset=CorpusDataset(run,allow_partial=True,max_shards=1)
        plan=ShapeBatchPlan(dataset,batch_size=2,epoch=1,seed=731)
        batches=[exact_shape_collate([dataset[i] for i in indices]) for indices in plan.batches[:2]]
        for batch in batches:batch['inputs']={k:v.cuda() for k,v in batch['inputs'].items()}
        def make_optimizer(m):
            o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5,foreach=False)
            s=torch.optim.lr_scheduler.StepLR(o,step_size=1,gamma=.9)
            scale=torch.amp.GradScaler('cuda',init_scale=16,growth_interval=2)
            return o,s,scale
        opt,sched,scaler=make_optimizer(model)
        ema={k:v.detach().clone() for k,v in model.state_dict().items()}
        def step(m,o,s,sc,e,batch):
            draws=(random.random(),float(np.random.random()))
            inp=dict(batch['inputs'])
            inp['pixels']=(inp['pixels']+torch.rand_like(inp['pixels'])*.002*(1+sum(draws))).clamp(0,1)
            o.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                pred=m(**inp);losses,_=instance_losses(pred,batch['targets'])
                loss=sum(losses.values())
                for aux in pred['auxiliary']:
                    more,_=instance_losses(aux,batch['targets']);loss=loss+.2*sum(more.values())
                # These small probe terms exercise all exported branches; they are
                # not the training loss manifest or meaningful teacher supervision.
                loss=loss+.01*(pred['font_embeddings'][...,0].mean()+pred['colour_rgb'].mean()+pred['background_under'].mean())
            previous=sc.get_scale();sc.scale(loss).backward();sc.unscale_(o)
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            sc.step(o);sc.update()
            assert sc.get_scale()>=previous,'Diagnostic optimizer update was skipped'
            s.step()
            with torch.no_grad():
                for key,value in m.state_dict().items():
                    if value.is_floating_point():e[key].lerp_(value,.01)
                    else:e[key].copy_(value)
            return dict(loss=float(loss.detach()),gradient_norm=float(norm),rng_draws=draws)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        first=step(model,opt,sched,scaler,ema,batches[0])
        assert opt.state and model.dino.blocks[-1].norm1.weight in opt.state
        sampler=copy.deepcopy(plan.state);sampler['consumed_batches']=1
        protocol=dict(diagnostic=True,corpus_snapshot=dataset.snapshot_hash,
                      configuration=model.resume_configuration(),
                      sources={name:sha(Path(__file__).with_name(name)) for name in ('model.py','attention.py','appearance.py','data.py','objectives.py','checkpoints.py','model_resume_checks.py')})
        protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        destination=run/('model_resume_smoke_'+str(time.time_ns()))
        write(destination/'protocol.json',protocol)
        checkpoint=save_epoch(destination,1,model,ema,opt,sched,scaler,sampler,protocol_hash,first)
        live=dict(raw=clone_cpu(model.state_dict()),ema=clone_cpu(ema),optimizer=clone_cpu(opt.state_dict()),
                  scheduler=copy.deepcopy(sched.state_dict()),scaler=copy.deepcopy(scaler.state_dict()),rng=rng_state())
        expected=step(model,opt,sched,scaler,ema,batches[1])
        expected_raw=clone_cpu(model.state_dict());expected_ema=clone_cpu(ema)
        expected_opt=clone_cpu(opt.state_dict());expected_sched=copy.deepcopy(sched.state_dict());expected_scale=scaler.state_dict()
        # Repeat the continuation entirely from live memory, without loading a
        # file or reconstructing a model, to measure native CUDA backward drift.
        # AdamW keeps its step counters on CPU; load_state_dict can retain those
        # tensor references. Preserve the comparison snapshot during replay.
        model.load_state_dict(live['raw']);opt.load_state_dict(clone_cpu(live['optimizer']))
        sched.load_state_dict(live['scheduler']);scaler.load_state_dict(live['scaler'])
        ema={k:v.cuda() for k,v in live['ema'].items()};restore_rng(live['rng'])
        control=step(model,opt,sched,scaler,ema,batches[1])
        control_raw=compare_tensors(expected_raw,model.state_dict(),2e-6,2e-5,strict=False)
        control_opt=compare_tensors(expected_opt,opt.state_dict(),1e-6,2e-4,strict=False)
        assert expected['rng_draws']==control['rng_draws'] and abs(expected['loss']-control['loss'])<=1e-5
        # Deliberately construct a frozen backbone first: policy mismatch must be
        # caught before the checkpoint mutates model or optimizer tensors.
        resumed=construct().cuda()
        ro,rs,rc=make_optimizer(resumed)
        try:load_resume(checkpoint,resumed,ro,rs,rc,protocol_hash)
        except ValueError as exc:assert 'trainable-parameter' in str(exc)
        else:raise AssertionError('Frozen/partial DINO mismatch accepted')
        resumed.set_dino_trainable(4);resumed.eval();ro,rs,rc=make_optimizer(resumed)
        # Same tensors, reversed order: optimizer state cannot safely be zipped
        # onto this layout even when parameter counts coincide.
        wrong=torch.optim.AdamW(list(reversed([p for p in resumed.parameters() if p.requires_grad])),lr=1e-5)
        try:load_resume(checkpoint,resumed,wrong,rs,rc,protocol_hash)
        except ValueError as exc:assert 'parameter order/group' in str(exc)
        else:raise AssertionError('Reordered optimizer accepted')
        torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
        state=load_resume(checkpoint,resumed,ro,rs,rc,protocol_hash)
        assert resumed.training and not resumed.dino.training
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
        restored_plan=ShapeBatchPlan.from_state(dataset,state['sampler_state'])
        assert list(restored_plan)==plan.batches[1:]
        resumed_ema={k:v.cuda() for k,v in state['ema'].items()}
        before_ema=compare_tensors(state['ema'],resumed_ema,0,0)
        compare_tensors(live['raw'],resumed.state_dict(),0,0)
        compare_tensors(live['optimizer'],ro.state_dict(),0,0)
        compare_tensors(live['ema'],resumed_ema,0,0)
        assert live['scheduler']==rs.state_dict() and live['scaler']==rc.state_dict()
        del state
        actual=step(resumed,ro,rs,rc,resumed_ema,batches[1])
        assert expected['rng_draws']==actual['rng_draws']
        assert abs(expected['loss']-actual['loss'])<=1e-5
        raw=compare_tensors(expected_raw,resumed.state_dict(),2e-6,2e-5,strict=False)
        moving=compare_tensors(expected_ema,resumed_ema,2e-7,2e-5,strict=False)
        optimizer=compare_tensors(expected_opt,ro.state_dict(),1e-6,2e-4,strict=False)
        # Do not call non-bit-exact CUDA continuation an exact restart. The
        # exact pre-step state tests above prove persistence; this bound compares
        # subsequent differences against the measured no-file control replay.
        bound=max(2e-6,10*control_raw['max_absolute_difference'])
        assert raw['max_absolute_difference']<=bound,'Restart drift exceeds measured native replay variation'
        assert expected_sched==rs.state_dict() and expected_scale==rc.state_dict()
        result=dict(created_at=now(),passed=True,diagnostic_only=True,actual_training_epochs=0,
                    path=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
                    model_parameters=model.parameter_counts(),raw_comparison=raw,ema_comparison=moving,
                    optimizer_comparison=optimizer,initial_ema_exact=before_ema['bit_exact'],
                    pre_step_raw_ema_optimizer_exact=True,live_memory_replay_raw=control_raw,
                    live_memory_replay_optimizer=control_opt,native_variation_bound=bound,
                    loss_before_restart=expected['loss'],loss_after_restart=actual['loss'],
                    sampler_rng_scheduler_scaler_modes_precision_restored=True,
                    wrong_unfreezing_and_optimizer_order_refused=True,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.perf_counter()-start,
                    limitation='Random DINO initialization, two logical diagnostic updates with the second replayed twice, one diagnostic shard, not baseline/specialist training. CUDA grid-sampling backward can be nondeterministic; numerical tolerances are explicit. Final heads/weights/optimizer schedule still need their readiness checks.')
        result['pretrained_backbone']=pretrained
        if pretrained:result['limitation']='Pretrained DINOv2, two diagnostic updates and two replays on one TRAIN shard; numerical CUDA drift is explicitly bounded. Not a training epoch.'
        write(run/('dinov2_resume_checks.json' if pretrained else 'model_resume_checks.json'),result)
        return result
    finally:
        restore_execution(original_execution,model)
        if dataset is not None:dataset.close()


def check_model_resume(run):
    if not torch.cuda.is_available():raise RuntimeError('Full model CUDA restart check requires CUDA')
    return _check(run)
