"""Synthetic-train-only task-affinity lookahead for the future specialist phases.

Adapts the prior normalized-gradient TAG measurement without updating the live
model. Group assignments still require a trained baseline and task manifest.
"""
import math
import torch
from .checkpoints import rng_state,restore_rng


def tag_batch(model,args,loss_function,tasks,shared_names,epsilon,split,kwargs=None):
    """Directed TAG[i,j] = 1 - loss_j(after normalized task-i step)/loss_j(before).

    loss_function(outputs) returns (scalar_losses, valid_example_counts). It must
    be pure for a fixed synthetic batch. Parameters/buffers are functional copies;
    temporary eval/TF32/RNG changes are restored even when measurement fails.
    """
    if split!='train':raise ValueError('Task affinity may use synthetic training batches only')
    if not math.isfinite(epsilon) or epsilon<=0:raise ValueError('Positive finite lookahead step required')
    tasks=tuple(tasks);shared_names=tuple(shared_names)
    if not tasks or len(set(tasks))!=len(tasks) or not shared_names or len(set(shared_names))!=len(shared_names):
        raise ValueError('Unique nonempty task and shared-parameter lists required')
    original=dict(model.named_parameters());buffers=dict(model.named_buffers())
    if any(name not in original or not original[name].requires_grad for name in shared_names):
        raise ValueError('Affinity parameters must be declared trainable shared parameters')
    if any(original[name].dtype!=torch.float32 for name in shared_names):raise ValueError('Measure shared lookahead parameters in FP32')
    device=original[shared_names[0]].device
    if any(original[name].device!=device for name in shared_names):raise ValueError('Shared parameters must use one device')
    base={name:p.detach().clone().requires_grad_(name in shared_names) for name,p in original.items()}
    original_buffers={name:b.detach().clone() for name,b in buffers.items()}
    modes=[(module,module.training) for module in model.modules()]
    saved_rng=rng_state();tf32=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    shared=[base[name] for name in shared_names]
    def evaluate(parameters):
        restore_rng(saved_rng)
        output=torch.func.functional_call(model,(parameters,{k:v.clone() for k,v in original_buffers.items()}),args,kwargs or {},strict=True)
        loss,support=loss_function(output)
        for name in tasks:
            if name not in loss or name not in support or loss[name].numel()!=1 or int(support[name])<0:
                raise ValueError('Incomplete task losses/support')
            if not torch.isfinite(loss[name]):raise ValueError('Nonfinite affinity loss: '+name)
        return loss,support
    size=len(tasks);tag=torch.full((size,size),torch.nan);cosine=tag.clone()
    gradients={};reasons={};norms={}
    try:
        model.eval();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with torch.enable_grad(),torch.autocast(device_type=device.type,enabled=False):
            losses,support=evaluate(base)
            values={name:float(losses[name].detach()) for name in tasks}
            for i,name in enumerate(tasks):
                if int(support[name])==0:reasons[name]='no_valid_labels';continue
                if values[name]<=1e-9:reasons[name]='near_zero_loss';continue
                if not losses[name].requires_grad:reasons[name]='no_shared_gradient';continue
                grads=torch.autograd.grad(losses[name],shared,retain_graph=True,allow_unused=True)
                flat=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for g,p in zip(grads,shared)])
                if not torch.isfinite(flat).all():raise ValueError('Nonfinite affinity gradient: '+name)
                norm=float(torch.linalg.vector_norm(flat.double()))
                norms[name]=norm
                if norm<=1e-12:reasons[name]='zero_shared_gradient';continue
                gradients[i]=(flat/norm).cpu()
            del losses
        valid=sorted(gradients)
        if valid:
            matrix=torch.stack([gradients[i] for i in valid]);similarity=matrix@matrix.T
            for a,i in enumerate(valid):
                for b,j in enumerate(valid):cosine[i,j]=similarity[a,b]
            with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
                for i in valid:
                    look=dict(base);offset=0;g=gradients[i].to(device)
                    for name in shared_names:
                        p=base[name];look[name]=p.detach()-epsilon*g[offset:offset+p.numel()].reshape_as(p);offset+=p.numel()
                    after,after_support=evaluate(look)
                    for j in valid:
                        name=tasks[j]
                        if int(after_support[name])!=int(support[name]):raise ValueError('Lookahead changed label support')
                        tag[i,j]=1-float(after[name])/values[name]
        result=dict(tasks=list(tasks),epsilon=epsilon,affinity=tag,cosine=cosine,
                    support={name:int(support[name]) for name in tasks},baseline_loss=values,
                    gradient_norm=norms,ineligible=reasons,shared_parameter_names=list(shared_names),
                    shared_parameter_count=sum(p.numel() for p in shared),split=split)
    finally:
        for module,training in modes:module.training=training
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=tf32
        restore_rng(saved_rng)
        if any(not torch.equal(p.detach(),base[name].detach()) for name,p in original.items()):
            raise RuntimeError('Live parameters changed during functional TAG')
        if any(not torch.equal(b,original_buffers[name]) for name,b in buffers.items()):
            raise RuntimeError('Live buffers changed during functional TAG')
    return result


def aggregate_tag(measurements):
    if not measurements:raise ValueError('No affinity measurements')
    first=measurements[0]
    for value in measurements:
        if value['tasks']!=first['tasks'] or value['epsilon']!=first['epsilon'] or value['shared_parameter_names']!=first['shared_parameter_names'] or value['split']!='train':
            raise ValueError('Incompatible affinity measurements')
    result={key:first[key] for key in ('tasks','epsilon','shared_parameter_names','shared_parameter_count','split')}
    for key in ('affinity','cosine'):
        stack=torch.stack([m[key] for m in measurements]);valid=torch.isfinite(stack)
        counts=valid.sum(0);mean=stack.nan_to_num().sum(0)/counts.clamp_min(1)
        mean[counts==0]=torch.nan
        result[key]=mean;result[key+'_counts']=counts
    diagonal=result['affinity'].diag();eligible=torch.isfinite(diagonal)
    result['eligible_tasks']=int(eligible.sum())
    result['positive_self_fraction']=float((diagonal[eligible]>0).float().mean()) if eligible.any() else None
    result['batch_count']=len(measurements)
    return result


def choose_epsilon(ordered_aggregates,minimum_positive_self=.9):
    """Choose the first measured valid step; never relax the self-affinity gate."""
    if not .9<=minimum_positive_self<=1:raise ValueError('Do not weaken the historical positive-self gate')
    if not ordered_aggregates:raise ValueError('No measured TAG sweep')
    first=ordered_aggregates[0]
    if any(a['tasks']!=first['tasks'] or a['shared_parameter_names']!=first['shared_parameter_names'] for a in ordered_aggregates):
        raise ValueError('TAG sweep changed tasks or shared parameters')
    for value in ordered_aggregates:
        if value['eligible_tasks']==len(value['tasks']) and value['positive_self_fraction']>=minimum_positive_self:
            return value
    raise ValueError('No fully represented valid TAG step; measure more train batches or backtrack the step before grouping')


def assigned_objective(losses,support,assigned,anchors,task_groups,group_weights):
    """Mask tasks BEFORE group averaging, with explicit preservation anchors."""
    groups={}
    for name in dict.fromkeys(tuple(assigned)+tuple(anchors)):
        if name not in losses or name not in support or name not in task_groups:raise ValueError('Unknown assigned/anchor task')
        if int(support[name])<=0:continue
        groups.setdefault(task_groups[name],[]).append(losses[name])
    if not groups:raise ValueError('No labelled active specialist tasks')
    values=[]
    for group,parts in groups.items():
        if group not in group_weights or not math.isfinite(group_weights[group]) or group_weights[group]<0:
            raise ValueError('Every active specialist group needs an explicit nonnegative weight')
        if group_weights[group]>0:values.append(torch.stack(parts).mean()*group_weights[group])
    if not values:raise ValueError('All active specialist groups have zero weight')
    result=torch.stack(values).sum()
    if not torch.isfinite(result):raise ValueError('Nonfinite specialist objective')
    return result
