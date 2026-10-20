"""Synthetic DEV evidence for joint heads and later specialist selection.

Task losses retain their evidence units. Fixed-batch font contrastive loss is
not font retrieval accuracy; a separate gallery evaluation must supply that.
This module never selects a checkpoint, fits thresholds or reads real labels.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .common import sha
from .data import ShapeBatchPlan,exact_shape_collate,EFFECT_HEADS
from .appearance import ATTRIBUTE_DIMS
from .specialist_tasks import task_losses,task_manifest,COMPOSITION
from .metrics import count_statistics,localization_statistics
from .checkpoints import rng_state,restore_rng,execution_state,restore_execution


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def contract(dataset,batch_size=8,instance_threshold=.5,iou_threshold=.5,binary_threshold=.5):
    if dataset.split!='dev':raise ValueError('Joint selection evaluator accepts synthetic DEV only')
    if not 0<instance_threshold<1 or not 0<iou_threshold<=1 or not 0<binary_threshold<1:
        raise ValueError('Invalid fixed evaluation thresholds')
    plan=ShapeBatchPlan(dataset,batch_size,epoch=0,seed=731,shuffle=False)
    here=Path(__file__).parent
    value=dict(schema='capri-joint-dev-v1',split='dev',purpose='selection',dataset_sha256=dataset.snapshot_hash,
                images=len(dataset),batch_size=batch_size,batch_plan_sha256=digest(plan.batches),
                instance_threshold=instance_threshold,iou_threshold=iou_threshold,binary_threshold=binary_threshold,
                task_directions={k:'lower' for k in task_manifest()},
                sources={name:sha(here/name) for name in ('joint_evaluation.py','data.py','specialist_tasks.py',
                                                        'objectives.py','appearance.py','metrics.py')},
                font_contrastive='Eligible-anchor weighted mean over this exact batch plan; not retrieval accuracy',
                selection='Metrics only; guard limits and teacher routing are separate frozen decisions')
    return dict(value,contract_sha256=digest(value))


class JointMetrics:
    def __init__(self,instance_threshold=.5,iou_threshold=.5,binary_threshold=.5):
        if not 0<instance_threshold<1 or not 0<iou_threshold<=1 or not 0<binary_threshold<1:
            raise ValueError('Invalid evaluation thresholds')
        self.instance_threshold=instance_threshold;self.iou_threshold=iou_threshold;self.binary_threshold=binary_threshold
        self.sums={k:0. for k in task_manifest()};self.support={k:0 for k in task_manifest()}
        self.measures={};self.binary={};self.logits=[];self.counts=[];self.presence=[];self.boxes=[];self.truth=[]

    def mean(self,name,values):
        values=values.detach().float()
        if not torch.isfinite(values).all():raise ValueError('Nonfinite evaluation metric '+name)
        total,n=self.measures.get(name,(0.,0));self.measures[name]=(total+float(values.sum()),n+values.numel())

    def classification(self,name,probability,truth):
        p=probability.detach().float();y=truth.to(p.device).bool()
        if not torch.isfinite(p).all() or ((p<0)|(p>1)).any() or p.shape!=y.shape:
            raise ValueError('Invalid binary evaluation predictions')
        positive=p>=self.binary_threshold
        values=[int((positive&y).sum()),int((positive&~y).sum()),int((~positive&y).sum()),int((~positive&~y).sum())]
        previous=self.binary.setdefault(name,[0,0,0,0])
        for i,v in enumerate(values):previous[i]+=v
        self.mean(name+'/brier',(p-y.float()).square())

    @torch.no_grad()
    def add(self,pred,targets):
        losses,support,assignments=task_losses(pred,targets)
        for name,value in losses.items():
            n=support[name]
            if type(n) is not int or n<0 or not math.isfinite(float(value)):raise ValueError('Invalid task metric evidence')
            self.sums[name]+=float(value)*n;self.support[name]+=n
        self.logits.append(pred['count_logits'].detach().float().cpu())
        self.presence.append(pred['presence_logits'].detach().float().cpu());self.boxes.append(pred['boxes'].detach().float().cpu())
        self.counts.extend(int(t['count']) for t in targets);self.truth.extend(t['boxes_visible'].detach().cpu() for t in targets)
        device=pred['count_logits'].device
        presence=torch.stack([t['effect_presence'] for t in targets]).to(device)
        intensity=torch.stack([t['effect_intensity'] for t in targets]).to(device)
        for i,name in enumerate(EFFECT_HEADS):
            self.classification('presence/'+name,pred['effect_presence_logits'][:,i].float().sigmoid(),presence[:,i])
            valid=presence[:,i].bool()
            self.mean('intensity/'+name+'/mae',(pred['intensity_predictions'][:,i].float()[valid]-intensity[:,i][valid]).abs())
        composition=torch.stack([t['composition'] for t in targets]).to(device)
        for i,name in enumerate(COMPOSITION):
            self.classification('composition/'+name,pred['composition_logits'][:,i].float().sigmoid(),composition[:,i])
        rows=torch.stack([t['rows'] for t in targets]).to(device)
        self.mean('rows/accuracy',(pred['row_logits'].argmax(-1)+1==rows).float())
        for b,(t,(pi,ti)) in enumerate(zip(targets,assignments)):
            for name in ATTRIBUTE_DIMS:
                meta=t['font_attributes'];valid=meta['valid'][name].to(device)[ti]
                p=pred['font_attribute_logits'][name][b,pi][valid].float();y=meta['values'][name].to(device)[ti][valid]
                if name in ('category','stroke'):self.mean('font_'+name+'/accuracy',(p.argmax(-1)==y).float())
                elif name=='italic':self.classification('font_italic',p[:,0].sigmoid(),y)
                else:
                    estimate=p[:,0].sigmoid() if name=='weight' else F.softplus(p[:,0])
                    self.mean('font_'+name+'/mae',(estimate-y).abs())
            y=t['colour_rgb'].to(device)[ti];valid=torch.isfinite(y).all(-1)
            self.mean('colour_rgb/mae',(pred['colour_rgb'][b,pi][valid].float()-y[valid]).abs().mean(-1))
            coverage=t['valid_mask'].to(device).float();truth=t['background_under'].to(device).float()/coverage.clamp_min(1e-8)[None]
            error=((pred['background_under'][b].float()-truth).abs()*coverage[None]).sum()/(3*coverage.sum())
            self.mean('background_under/mae',error[None])

    def finish(self,expected_images):
        if expected_images<1 or len(self.counts)!=expected_images:raise ValueError('Incomplete DEV metric coverage')
        metrics={name:dict(value=self.sums[name]/n if n else None,n=n,direction='lower',kind='task_loss')
                 for name,n in self.support.items()}
        for name,(total,n) in self.measures.items():
            metrics[name]=dict(value=total/n if n else None,n=n,direction='higher' if name.endswith('/accuracy') else 'lower')
        binary={}
        for name,(tp,fp,fn,tn) in self.binary.items():
            n=tp+fp+fn+tn;positives=tp+fn;negatives=tn+fp
            balanced=(tp/positives+tn/negatives)/2 if positives and negatives else None
            binary[name]=dict(tp=tp,fp=fp,fn=fn,tn=tn,threshold=self.binary_threshold)
            metrics[name+'/accuracy']=dict(value=(tp+tn)/n if n else None,n=n,direction='higher')
            metrics[name+'/balanced_accuracy']=dict(value=balanced,n=n,direction='higher')
        count=count_statistics(torch.cat(self.logits),self.counts,torch.cat(self.presence),self.instance_threshold)
        location=localization_statistics(torch.cat(self.boxes),torch.cat(self.presence),self.truth,self.instance_threshold,self.iou_threshold)
        metrics['count_accuracy']=dict(value=count['correct']/count['n'],n=count['n'],direction='higher')
        metrics['count_mae']=dict(value=count['mae'],n=count['n'],direction='lower')
        metrics['localization_f1']=dict(value=location['f1'],n=location['true_positive']+location['false_negative'],direction='higher')
        return dict(metrics=metrics,count=count,localization=location,binary_confusions=binary,images=expected_images,
                    limitations=['No font retrieval accuracy/gallery metric is supplied by contrastive loss.',
                                 'Unsupported metrics are null with zero evidence; balanced accuracy is null if either class is absent.',
                                 'Task loss scales/units differ; never average these into a selection score implicitly.',
                                 'Thresholds are fixed diagnostics, not fitted or calibrated.'])


def evaluate_joint(model,dataset,frozen_contract,workers=4,on_progress=None):
    """Evaluate current loaded weights; phase engine owns raw/EMA switching."""
    current=contract(dataset,**{k:frozen_contract[k] for k in ('batch_size','instance_threshold','iou_threshold','binary_threshold')})
    if current!=frozen_contract:raise ValueError('Joint evaluation protocol/data/source mismatch')
    if workers<0:raise ValueError('Invalid evaluator worker count')
    device=next(model.parameters()).device;modes=execution_state(model);rng=rng_state()
    meter=JointMetrics(**{k:current[k] for k in ('instance_threshold','iou_threshold','binary_threshold')})
    plan=ShapeBatchPlan(dataset,current['batch_size'],epoch=0,seed=731,shuffle=False)
    seen=set();generator=torch.Generator().manual_seed(731)
    try:
        model.eval()
        loader=DataLoader(dataset,batch_sampler=plan,collate_fn=exact_shape_collate,num_workers=workers,
                          generator=generator,pin_memory=device.type=='cuda')
        with torch.no_grad():
            for i,batch in enumerate(loader):
                if set(batch['inputs'])!={'pixels','valid_pixels'}:raise ValueError('Only pixel inputs permitted')
                for row in batch['audit']:
                    if row['stem'] in seen:raise ValueError('Repeated DEV sample')
                    seen.add(row['stem'])
                inputs={k:v.to(device,non_blocking=True) for k,v in batch['inputs'].items()}
                with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):pred=model(**inputs)
                meter.add(pred,batch['targets'])
                if on_progress and (i%100==0 or i+1==len(plan)):
                    on_progress(dict(state='validating',validation_batches=i+1,validation_total=len(plan),images=len(seen)))
        if seen!={row['stem'] for row in dataset.rows}:raise ValueError('DEV sample identity mismatch')
        report=meter.finish(len(dataset))
        report.update(split='dev',purpose='selection',dataset_sha256=dataset.snapshot_hash,evaluation_protocol_sha256=digest(current))
        return report
    finally:restore_execution(modes,model);restore_rng(rng)
