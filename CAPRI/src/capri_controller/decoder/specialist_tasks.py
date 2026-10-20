"""Separate, evidence-counted tasks for joint training and measured specialists.

This adapter does not change the frozen running localization stage. Loss terms
are not a new neural head count. Weights/grouping are supplied by later protocols.
"""
import torch
from torch.nn import functional as F
from .appearance import ATTRIBUTE_DIMS
from .data import EFFECT_HEADS
from .objectives import instance_losses
from .affinity import assigned_objective

COUNT_TASKS=('count_ce','cardinality','count_consistency')
LOCAL_TASKS=('objectness','box_l1','box_giou','mask_bce','mask_dice','visibility')
COMPOSITION=('digit','uppercase','lowercase')


def task_manifest():
    groups={k:'count' for k in COUNT_TASKS}
    groups.update({k:'localization' for k in LOCAL_TASKS})
    groups.update({'font_contrastive':'font',**{'font_'+k:'font' for k in ATTRIBUTE_DIMS}})
    groups.update({'colour_rgb':'appearance','background_under':'appearance','rows':'composition'})
    groups.update({'composition/'+k:'composition' for k in COMPOSITION})
    groups.update({'presence/'+k:'effect_presence' for k in EFFECT_HEADS})
    groups.update({'intensity/'+k:'effect_intensity' for k in EFFECT_HEADS})
    return groups


def task_losses(pred,targets,assignments=None):
    """Return 96 scalar loss terms, valid evidence counts and glyph assignments.

Counts describe each term's evidence unit (images, matched glyphs, or eligible
contrastive anchors); they are not interchangeable metric denominators.
"""
    losses,assignment=instance_losses(pred,targets,assignments)
    required=set(COUNT_TASKS+LOCAL_TASKS+('font_contrastive','rows','colour_rgb','background_under'))
    required.update('font_'+k for k in ATTRIBUTE_DIMS)
    if not required.issubset(losses):raise ValueError('Complete model outputs required for the joint task adapter')
    device=pred['presence_logits'].device;batch=len(targets)
    losses={k:v for k,v in losses.items() if k not in ('effect_presence','effect_intensity','composition')}
    support={k:batch for k in losses}
    families=torch.cat([t['font_family'].to(device)[ti] for t,(_,ti) in zip(targets,assignment)])
    eye=torch.eye(len(families),dtype=torch.bool,device=device)
    positive=(families[:,None]==families[None,:])&~eye
    negative=(families[:,None]!=families[None,:])&~eye
    support['font_contrastive']=int((positive.any(-1)&negative.any(-1)).sum())
    for key in ATTRIBUTE_DIMS:
        support['font_'+key]=sum(int(t['font_attributes']['valid'][key].to(device)[ti].sum())
                                for t,(_,ti) in zip(targets,assignment))
    support['colour_rgb']=sum(int(torch.isfinite(t['colour_rgb'].to(device)[ti]).all(-1).sum())
                              for t,(_,ti) in zip(targets,assignment))
    presence=torch.stack([t['effect_presence'].to(device) for t in targets]).float()
    valid=torch.stack([t['intensity_valid'].to(device) for t in targets]).bool()
    intensity=torch.stack([t['effect_intensity'].to(device) for t in targets]).float()
    if presence.shape!=(batch,len(EFFECT_HEADS)) or not torch.equal(valid,presence.bool()):
        raise ValueError('Conditional intensity support must match known synthetic effect presence')
    with torch.autocast(device_type=device.type,enabled=False):
        per_presence=F.binary_cross_entropy_with_logits(pred['effect_presence_logits'].float(),presence,reduction='none')
        per_intensity=F.smooth_l1_loss(pred['intensity_predictions'].float(),intensity,reduction='none')
        for i,key in enumerate(EFFECT_HEADS):
            p='presence/'+key;j='intensity/'+key
            losses[p]=per_presence[:,i].mean();support[p]=batch
            support[j]=int(valid[:,i].sum())
            losses[j]=(per_intensity[:,i]*valid[:,i]).sum()/max(1,support[j])
        truth=torch.stack([t['composition'].to(device) for t in targets]).float()
        values=F.binary_cross_entropy_with_logits(pred['composition_logits'].float(),truth,reduction='none')
        for i,key in enumerate(COMPOSITION):
            losses['composition/'+key]=values[:,i].mean();support['composition/'+key]=batch
    if set(losses)!=set(task_manifest()) or set(support)!=set(losses):raise ValueError('Joint task contract mismatch')
    if not all(v.numel()==1 and torch.isfinite(v) for v in losses.values()):raise ValueError('Nonfinite joint task')
    return losses,support,assignment


def specialist_objective(losses,support,assigned,anchors,group_weights,task_scales):
    """Only assigned/anchor terms enter a group mean; unknown/empty tasks fail closed."""
    active=tuple(dict.fromkeys(tuple(assigned)+tuple(anchors)))
    if not active or any(k not in task_manifest() for k in active):raise ValueError('Unknown/empty active task set')
    if any(k not in task_scales or not 0<float(task_scales[k])<float('inf') for k in active):
        raise ValueError('Every active task needs a positive finite frozen scale')
    scaled={k:losses[k]*task_scales[k] for k in active}
    return assigned_objective(scaled,support,assigned,anchors,task_manifest(),group_weights)
