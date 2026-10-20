"""Bounded, non-optimizing timing of the stopped CAPRI model on TRAIN pixels."""
import collections
import contextlib
import statistics
import time
from pathlib import Path
import torch
from torch.profiler import profile,ProfilerActivity,record_function
from .common import read,write,sha,now,StageLock
from . import training
from .data import CorpusDataset,ShapeBatchPlan,exact_shape_collate


def move(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device) for v in value]
    return value


def measure(run):
    import psutil
    if not (run/'user_stop.json').exists():raise ValueError('This diagnostic requires an explicitly stopped run')
    for proc in psutil.process_iter(['pid','name','cmdline']):
        cmd=proc.info['cmdline'] or []
        if 'train-baseline' in cmd:raise RuntimeError('Training process still present')
    root=run/'diagnostics'/('performance_'+str(time.time_ns()));root.mkdir(parents=True)
    with StageLock(root,'profile'):
        write(root/'progress.json',dict(state='loading',created_at=now(),optimizer_steps=0))
        frozen=read(run/'training/baseline_localization/protocol.json')
        for path,digest in frozen['sources'].items():
            if sha(path)!=digest:raise ValueError('Frozen source changed: '+path)
        checkpoint=Path(read(run/'training_recovery/baseline_localization/latest.json')['path'])
        if checkpoint.is_dir():checkpoint=checkpoint/'checkpoint.pt'
        receipt=read(checkpoint.parent/'receipt.json')
        if sha(checkpoint)!=receipt['sha256']:raise ValueError('Checkpoint checksum mismatch')
        training.initialize();model=training.construct()
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state['raw'],strict=True);del state
        dataset=CorpusDataset(run,allow_partial=True,max_shards=1)
        plan=ShapeBatchPlan(dataset,8,epoch=1)
        full=[b for b in plan.batches if len(b)==8]
        full.sort(key=lambda b:dataset.rows[b[0]]['shape'][0]*dataset.rows[b[0]]['shape'][1])
        selected=[full[round(i*(len(full)-1)/3)] for i in range(4)]
        rows=[];batches=[]
        def timed(label,fn):
            torch.cuda.synchronize();start=time.perf_counter()
            with record_function(label):value=fn()
            torch.cuda.synchronize();return value,(time.perf_counter()-start)*1000
        def trial(batch,gpu_targets=False):
            model.zero_grad(set_to_none=True)
            values={};inputs,values['input_transfer_ms']=timed('probe/transfer',lambda:move(batch['inputs'],'cuda'))
            targets=batch['targets']
            if gpu_targets:targets,values['target_transfer_ms']=timed('probe/target_transfer',lambda:move(targets,'cuda'))
            with torch.autocast('cuda',dtype=torch.float16):
                pred,values['forward_ms']=timed('probe/forward',lambda:model(**inputs))
                (loss,parts),values['objective_ms']=timed('probe/objective',lambda:training.objective(pred,targets,frozen['weights']))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite diagnostic loss')
            _,values['backward_ms']=timed('probe/backward',lambda:(loss*128).backward())
            values['loss']=float(loss.detach());values['total_ms']=sum(v for k,v in values.items() if k.endswith('_ms'))
            del pred,loss,parts,inputs
            return values
        for indices in selected:
            start=time.perf_counter();batch=exact_shape_collate([dataset[i] for i in indices]);decode_ms=(time.perf_counter()-start)*1000
            batches.append(batch)
            for _ in range(2):trial(batch)
            measured=[trial(batch) for _ in range(3)]
            relocated=[trial(batch,True) for _ in range(3)]
            row=dict(shape=list(batch['inputs']['pixels'].shape),stems=[x['stem'] for x in batch['audit']],
                synchronous_decode_ms=decode_ms,baseline=measured,targets_transferred_once=relocated,
                medians={k:statistics.median(x[k] for x in measured) for k in measured[0]},
                gpu_target_medians={k:statistics.median(x[k] for x in relocated) for k in relocated[0]})
            rows.append(row);write(root/'progress.json',dict(state='timing',completed_shapes=len(rows),rows=rows,optimizer_steps=0))
            print('profile shape '+str(row['shape'])+' '+str(row['medians']),flush=True)
        # Named module scopes plus operator-level CPU/CUDA accounting. These
        # scopes only affect this disposable model/process, never frozen code.
        def wrapped(fn,name):
            def call(*args,**kwargs):
                with record_function(name):return fn(*args,**kwargs)
            return call
        for name,module in model.named_modules():
            if name in ('cnn','font_reduce','font_spatial','font_project','font_style') or name in [f'memory.{i}' for i in range(2)]+[f'decoder.{i}' for i in range(4)]:
                module.forward=wrapped(module.forward,'module/'+name)
        model.dino.forward_features=wrapped(model.dino.forward_features,'module/dino')
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as prof:
            trial(batches[2])
        events=prof.key_averages()
        def event(e):return dict(name=e.key,calls=e.count,cpu_total_ms=e.cpu_time_total/1000,
            cpu_self_ms=e.self_cpu_time_total/1000,device_total_ms=e.device_time_total/1000,
            device_self_ms=e.self_device_time_total/1000)
        stats=[event(e) for e in events]
        result=dict(created_at=now(),source_sha256=sha(__file__),checkpoint=str(checkpoint),checkpoint_sha256=receipt['sha256'],
            frozen_training_sources_unchanged=True,parameters=model.parameter_counts(),optimizer_steps=0,
            gpu=torch.cuda.get_device_name(),pytorch=torch.__version__,rows=rows,
            scopes=[e for e in stats if e['name'].startswith(('probe/','module/'))],
            top_cpu=sorted(stats,key=lambda e:e['cpu_self_ms'],reverse=True)[:35],
            top_device=sorted(stats,key=lambda e:e['device_self_ms'],reverse=True)[:35],
            limitation='Four shapes from the first 1000 TRAIN images; synchronized cached-batch forward/loss/backward only. No optimizer updates, EMA, concurrent loader throughput or full-epoch performance claim. Profiler adds overhead. Target-transfer variant preserves objective but is diagnostic only.')
        write(root/'report.json',result)
        (root/'operators_cpu.txt').write_text(events.table(sort_by='self_cpu_time_total',row_limit=50),encoding='utf8')
        (root/'operators_cuda.txt').write_text(events.table(sort_by='self_cuda_time_total',row_limit=50),encoding='utf8')
        dataset.close();model.zero_grad(set_to_none=True)
        write(root/'progress.json',dict(state='complete',report=str(root/'report.json'),optimizer_steps=0))
        return dict(report=str(root/'report.json'),optimizer_steps=0)
