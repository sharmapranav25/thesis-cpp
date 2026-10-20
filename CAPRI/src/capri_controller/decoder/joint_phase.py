"""Reference preparation and weighted objective for the later joint phase.

These callable adapters connect to run_phase; production initialization, GPU
readiness, budgets, DEV selection and a phase lock remain the launcher's job.
"""
import math
from pathlib import Path
import torch
from .common import sha
from .font_extraction import extraction_policy
from .font_training import joint_task_forward
from .specialist_tasks import task_manifest,task_losses,specialist_objective,COUNT_TASKS,LOCAL_TASKS
from .objectives import instance_losses
from .reference_sampling import digest


def device_copy(value,device):
    if torch.is_tensor(value):return value.to(device,non_blocking=True)
    if isinstance(value,dict):return {k:device_copy(v,device) for k,v in value.items()}
    if isinstance(value,list):return [device_copy(v,device) for v in value]
    if isinstance(value,tuple):return tuple(device_copy(v,device) for v in value)
    return value


class JointPreparation:
    def __init__(self,sampler,every_updates,max_references,mode):
        if type(every_updates) is not int or every_updates<1 or type(max_references) is not int or max_references<1:
            raise ValueError('Explicit positive reference frequency and budget required')
        if mode not in ('family','font_file'):raise ValueError('Explicit reference pairing mode required')
        self.sampler=sampler
        self.contract=dict(schema='capri-joint-preparation-v1',sampling=sampler.contract,every_updates=every_updates,
            max_references=max_references,mode=mode,source_sha256=sha(__file__),schedule='first update then every_updates globally')

    def __call__(self,batch,context):
        if set(context)!=set(('epoch','optimizer_update','consumed_batches')) or any(type(v) is not int for v in context.values()):
            raise ValueError('Explicit integer phase context required')
        if context['epoch']<1 or context['optimizer_update']<1 or context['consumed_batches']<0:
            raise ValueError('Invalid phase context')
        selection=None;references=[]
        if (context['optimizer_update']-1)%self.contract['every_updates']==0:
            selection=self.sampler.select(batch,context['epoch'],self.contract['max_references'],self.contract['mode'])
            references=self.sampler.materialize(selection)
        return dict(batch,reference_batches=references,reference_selection=selection,phase_context=dict(context))


class JointObjective:
    def __init__(self,family_indices,active_tasks,group_weights,task_scales,reference_policy,temperature,
                 auxiliary_weight,auxiliary_task_weights):
        tasks=tuple(sorted(set(active_tasks)))
        if not tasks or any(k not in task_manifest() for k in tasks):raise ValueError('Explicit known active tasks required')
        if set(task_scales)!=set(tasks) or any(not math.isfinite(v) or v<=0 for v in task_scales.values()):
            raise ValueError('Explicit positive active-task scales required')
        groups={task_manifest()[k] for k in tasks}
        if set(group_weights)!=groups or any(not math.isfinite(v) or v<0 for v in group_weights.values()) or not any(group_weights.values()):
            raise ValueError('Explicit active-group weights required')
        if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(auxiliary_weight) or auxiliary_weight<0:
            raise ValueError('Invalid font temperature or auxiliary coefficient')
        required=set(COUNT_TASKS+LOCAL_TASKS)-{'visibility'}
        if set(auxiliary_task_weights)!=required or any(not math.isfinite(v) or v<0 for v in auxiliary_task_weights.values()):
            raise ValueError('Explicit complete auxiliary localization weights required')
        self.family_indices=dict(family_indices)
        self.contract=dict(schema='capri-joint-objective-v1',active_tasks=list(tasks),group_weights=dict(group_weights),
            family_indices_sha256=digest(self.family_indices),
            task_scales=dict(task_scales),reference_policy=extraction_policy(reference_policy),temperature=temperature,
            auxiliary_weight=auxiliary_weight,auxiliary_task_weights=dict(auxiliary_task_weights),
            sources={name:sha(Path(__file__).with_name(name)) for name in ('joint_phase.py','font_training.py',
                'reference_sampling.py','reference_data.py','font_extraction.py','specialist_tasks.py','affinity.py',
                'objectives.py','appearance.py','data.py')})

    def __call__(self,model,batch):
        if 'reference_batches' not in batch or 'phase_context' not in batch:
            raise ValueError('Joint objective requires epoch-aware batch preparation')
        if set(batch['inputs'])!={'pixels','valid_pixels'}:raise ValueError('Pixel-only scene input required')
        scene=device_copy(dict(inputs=batch['inputs'],targets=batch['targets']),next(model.parameters()).device)
        references=device_copy(batch['reference_batches'],next(model.parameters()).device)
        c=self.contract
        if references:
            result=joint_task_forward(model,scene,references,self.family_indices,c['reference_policy'],c['temperature'])
            pred=result['scene_prediction'];losses=result['losses'];support=result['support']
        else:
            pred=model(**scene['inputs']);losses,support,_=task_losses(pred,scene['targets'])
        loss=specialist_objective(losses,support,c['active_tasks'],(),c['group_weights'],c['task_scales'])
        if c['auxiliary_weight']:
            if not pred.get('auxiliary'):raise ValueError('Missing auxiliary localization predictions')
            for auxiliary in pred['auxiliary']:
                more,_=instance_losses(auxiliary,scene['targets'])
                if set(more)!=set(c['auxiliary_task_weights']):raise ValueError('Auxiliary localization task mismatch')
                loss=loss+c['auxiliary_weight']*sum(more[k]*c['auxiliary_task_weights'][k] for k in more)
        if not torch.isfinite(loss):raise ValueError('Nonfinite joint objective')
        return loss
