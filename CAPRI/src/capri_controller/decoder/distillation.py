"""Assigned-output distillation using synthetic glyph identity for query alignment.

Teacher tensors always detach. This module performs no teacher selection, model
loading, weight averaging, target inference, or optimization on held-out labels.
"""
import math
import torch
from torch.nn import functional as F
from .data import EFFECT_HEADS
from .appearance import ATTRIBUTE_DIMS
from .objectives import aligned_queries,match_instances
from .specialist_tasks import task_manifest,COMPOSITION


def categorical_kl(student,teacher,temperature):
    teacher=teacher.detach().float()/temperature;student=student.float()/temperature
    return F.kl_div(student.log_softmax(-1),teacher.softmax(-1),reduction='none').sum(-1).clamp_min(0)*temperature**2


def binary_kl(student,teacher,temperature):
    s=student.float()/temperature;t=teacher.detach().float()/temperature;p=t.sigmoid()
    return (p*(F.logsigmoid(t)-F.logsigmoid(s))+(1-p)*(F.logsigmoid(-t)-F.logsigmoid(-s))).clamp_min(0)*temperature**2


def aligned_distillation(student,teacher,targets,assigned,student_assignment=None,teacher_assignment=None,temperature=2.):
    """Return only the assigned losses and their support, with no implicit anchors.

    Instance objectness distills matched glyphs only; synthetic hard supervision
    remains responsible for unmatched negatives. Box L1/GIoU share a coordinate
    imitation loss; mask BCE/Dice share a valid-pixel soft-distribution loss.
    Auxiliary decoder layers retain hard-label losses, not arbitrary slot pairs.
    """
    assigned=tuple(assigned)
    if not assigned or len(set(assigned))!=len(assigned) or any(k not in task_manifest() for k in assigned):
        raise ValueError('Unique known assigned tasks required')
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Positive finite temperature required')
    if len(targets)!=len(student['presence_logits']) or student['presence_logits'].shape!=teacher['presence_logits'].shape:
        raise ValueError('Compatible teacher/student query batches required')
    sa=match_instances(student,targets) if student_assignment is None else student_assignment
    ta=match_instances(teacher,targets) if teacher_assignment is None else teacher_assignment
    if len(sa)!=len(targets) or len(ta)!=len(targets):raise ValueError('Incomplete assignments')
    device=student['presence_logits'].device
    aligned=[]
    for t,s,a in zip(targets,sa,ta):
        n=int(t['count'])
        for pi,ti in (s,a):
            if len(pi)!=n or len(set(pi.tolist()))!=n or sorted(ti.tolist())!=list(range(n)):
                raise ValueError('Alignment must cover every synthetic glyph exactly once')
        si,ti=aligned_queries(s,a);aligned.append((si,ti))
    def matched(key):
        return (torch.cat([student[key][b,si] for b,(si,_) in enumerate(aligned)]).float(),
                torch.cat([teacher[key][b,ti].detach() for b,(_,ti) in enumerate(aligned)]).float())
    losses={};support={};n=sum(int(t['count']) for t in targets)
    with torch.autocast(device_type=device.type,enabled=False):
        for key in assigned:
            if key=='count_ce':
                value=categorical_kl(student['count_logits'],teacher['count_logits'],temperature).mean();count=len(targets)
            elif key in ('cardinality','count_consistency'):
                s=student['presence_logits'].float().sigmoid().sum(-1)
                t=teacher['presence_logits'].detach().float().sigmoid().sum(-1)
                if key=='count_consistency':
                    axis=torch.arange(2,9,device=device)
                    s=s-(student['count_logits'].float().softmax(-1)*axis).sum(-1)
                    t=t-(teacher['count_logits'].detach().float().softmax(-1)*axis).sum(-1)
                value=F.smooth_l1_loss(s,t);count=len(targets)
            elif key in ('objectness','visibility'):
                s,t=matched('presence_logits' if key=='objectness' else 'visibility_logits')
                value=binary_kl(s,t,temperature).mean();count=n
            elif key in ('box_l1','box_giou','colour_rgb','font_contrastive'):
                field={'box_l1':'boxes','box_giou':'boxes','colour_rgb':'colour_rgb','font_contrastive':'font_embeddings'}[key]
                s,t=matched(field);count=n
                value=(1-F.cosine_similarity(s,t,dim=-1)).clamp_min(0).mean() if key=='font_contrastive' else F.smooth_l1_loss(s,t)
            elif key in ('mask_bce','mask_dice'):
                terms=[]
                for b,(si,ti) in enumerate(aligned):
                    mask=targets[b]['valid_mask'].to(device).float()
                    if mask.sum()<=0:raise ValueError('No valid mask pixels')
                    per=binary_kl(student['mask_logits'][b,si],teacher['mask_logits'][b,ti],temperature)
                    terms.append((per*mask).sum()/(mask.sum()*len(si)))
                value=torch.stack(terms).mean();count=len(targets)
            elif key=='rows':
                value=categorical_kl(student['row_logits'],teacher['row_logits'],temperature).mean();count=len(targets)
            elif key.startswith('composition/'):
                i=COMPOSITION.index(key.split('/')[1])
                value=binary_kl(student['composition_logits'][:,i],teacher['composition_logits'][:,i],temperature).mean();count=len(targets)
            elif key.startswith(('presence/','intensity/')):
                group,name=key.split('/');i=EFFECT_HEADS.index(name)
                if group=='presence':
                    value=binary_kl(student['effect_presence_logits'][:,i],teacher['effect_presence_logits'][:,i],temperature).mean();count=len(targets)
                else:
                    valid=torch.tensor([bool(t['intensity_valid'][i]) for t in targets],device=device)
                    count=int(valid.sum())
                    per=F.smooth_l1_loss(student['intensity_predictions'][:,i].float(),teacher['intensity_predictions'][:,i].detach().float(),reduction='none')
                    value=(per*valid).sum()/max(count,1)
            elif key.startswith('font_') and key[5:] in ATTRIBUTE_DIMS:
                attribute=key[5:];s=[];t=[];masks=[]
                for b,((si,ti),gt) in enumerate(zip(aligned,targets)):
                    # aligned_queries orders by glyph identity, so validity is in
                    # original glyph order, independent of query or assignment order.
                    masks.append(gt['font_attributes']['valid'][attribute].to(device))
                    s.append(student['font_attribute_logits'][attribute][b,si].float())
                    t.append(teacher['font_attribute_logits'][attribute][b,ti].detach().float())
                valid=torch.cat(masks);s=torch.cat(s);t=torch.cat(t);count=int(valid.sum())
                if attribute in ('category','stroke'):per=categorical_kl(s,t,temperature)
                elif attribute=='italic':per=binary_kl(s[:,0],t[:,0],temperature)
                else:
                    s=s[:,0].sigmoid() if attribute=='weight' else F.softplus(s[:,0])
                    t=t[:,0].sigmoid() if attribute=='weight' else F.softplus(t[:,0])
                    per=F.smooth_l1_loss(s,t,reduction='none')
                value=(per*valid).sum()/max(count,1)
            elif key=='background_under':
                terms=[]
                for b,target in enumerate(targets):
                    valid=target['valid_mask'].to(device).float()
                    per=F.smooth_l1_loss(student[key][b].float(),teacher[key][b].detach().float(),reduction='none')
                    terms.append((per*valid).sum()/(3*valid.sum().clamp_min(1)))
                value=torch.stack(terms).mean();count=len(targets)
            else:raise ValueError('No distillation adapter for '+key)
            if value.numel()!=1 or not torch.isfinite(value):raise ValueError('Nonfinite distillation: '+key)
            losses[key]=value;support[key]=count
    return losses,support
