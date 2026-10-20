"""Instance matching and separate, unweighted decoder task losses.

These primitives do not choose the training budget or task coefficients. All
targets are synthetic supervision and never enter the visual forward pass.
"""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .appearance import appearance_losses


def generalized_iou(a,b):
    """Pairwise GIoU for normalized xyxy boxes, including out-of-canvas boxes."""
    awh=(a[:,2:]-a[:,:2]).clamp_min(0);bwh=(b[:,2:]-b[:,:2]).clamp_min(0)
    lower=torch.maximum(a[:,None,:2],b[None,:,:2]);upper=torch.minimum(a[:,None,2:],b[None,:,2:])
    inter=(upper-lower).clamp_min(0).prod(-1)
    union=awh.prod(-1)[:,None]+bwh.prod(-1)[None,:]-inter
    iou=inter/union.clamp_min(1e-8)
    outer=(torch.maximum(a[:,None,2:],b[None,:,2:])-torch.minimum(a[:,None,:2],b[None,:,:2])).clamp_min(0).prod(-1)
    return iou-(outer-union)/outer.clamp_min(1e-8)


def pairwise_mask_cost(logits,target,valid):
    """BCE and soft Dice matrices without materializing Q x N x pixels."""
    q=logits.flatten(1).float();t=target.flatten(1).float();v=valid.flatten().float()
    area=v.sum().clamp_min(1)
    bce=(F.softplus(q)*v).sum(-1)[:,None]/area-(q@((t*v).T))/area
    p=q.sigmoid();intersection=(p*v)@t.T
    dice=1-(2*intersection+1)/((p*v).sum(-1)[:,None]+(t*v).sum(-1)[None,:]+1)
    return bce,dice


@torch.no_grad()
def match_instances(pred,targets,costs=None):
    """One-to-one assignment; no ordering or ground-truth count enters inference."""
    costs=dict(presence=1.,box=5.,giou=2.,mask_bce=1.,mask_dice=1.) if costs is None else costs
    if set(costs)!=set(('presence','box','giou','mask_bce','mask_dice')):
        raise ValueError('Specify all matching costs')
    if any(not torch.isfinite(torch.tensor(v)) or v<0 for v in costs.values()) or not any(costs.values()):
        raise ValueError('Matching weights must be finite, nonnegative and nonzero in aggregate')
    presence,boxes,masks=pred['presence_logits'],pred['boxes'],pred['mask_logits']
    batch,queries=presence.shape
    if batch!=len(targets) or boxes.shape!=(batch,queries,4) or masks.shape[:2]!=(batch,queries):
        raise ValueError('Inconsistent prediction shapes')
    if not all(torch.isfinite(value).all() for value in (presence,boxes,masks)):
        raise ValueError('Nonfinite matching predictions')
    assignments=[]
    with torch.autocast(device_type=presence.device.type,enabled=False):
        for i,target in enumerate(targets):
            gt=target['boxes_visible'].to(boxes.device).float()
            truth=target['masks']['visible'].to(masks.device).float()
            valid=target['valid_mask'].to(masks.device).float()
            n=len(gt)
            if not 1<=n<=queries or n!=int(target['count']) or truth.shape!=(n,*masks.shape[-2:]) or valid.shape!=masks.shape[-2:]:
                raise ValueError('Invalid target count/shape or insufficient query capacity')
            if not torch.isfinite(gt).all() or not torch.isfinite(truth).all() or not torch.isfinite(valid).all():
                raise ValueError('Nonfinite matching targets')
            if (gt[:,2:]<=gt[:,:2]).any() or (boxes[i,:,2:]<boxes[i,:,:2]).any():
                raise ValueError('Inverted or empty target box')
            if valid.sum()<=0 or truth.min()<0 or truth.max()>1 or valid.min()<0 or valid.max()>1:
                raise ValueError('Invalid mask coverage')
            bce,dice=pairwise_mask_cost(masks[i],truth,valid)
            cost=(-costs['presence']*presence[i].float().sigmoid()[:,None]
                  +costs['box']*torch.cdist(boxes[i].float(),gt,p=1)
                  -costs['giou']*generalized_iou(boxes[i].float(),gt)
                  +costs['mask_bce']*bce+costs['mask_dice']*dice)
            if not torch.isfinite(cost).all():raise ValueError('Nonfinite assignment cost')
            rows,cols=linear_sum_assignment(cost.cpu().numpy())
            assignments.append((torch.as_tensor(rows,device=boxes.device),torch.as_tensor(cols,device=boxes.device)))
    return assignments


def font_contrastive(embedding,families,temperature=.1):
    """Family positives; ignore anchors without both a positive and a negative."""
    if temperature<=0:raise ValueError('Positive contrastive temperature required')
    if len(embedding)<2:return embedding.sum()*0
    z=F.normalize(embedding.float(),dim=-1)
    logits=z@z.T/temperature
    self_mask=torch.eye(len(z),dtype=torch.bool,device=z.device)
    positives=(families[:,None]==families[None,:])&~self_mask
    negatives=(families[:,None]!=families[None,:])&~self_mask
    anchors=positives.any(-1)&negatives.any(-1)
    if not anchors.any():return embedding.sum()*0
    log_prob=logits-torch.logsumexp(logits.masked_fill(self_mask,-torch.inf),dim=-1,keepdim=True)
    per_anchor=-(log_prob.masked_fill(~positives,0).sum(-1)/positives.sum(-1).clamp_min(1))
    return per_anchor[anchors].mean()


def instance_losses(pred,targets,assignments=None,no_object_weight=.1):
    """Return per-task means; the future frozen protocol supplies aggregate weights.

    Optional intensity_predictions are bounded scalar severities in [0,1], not
    probabilities of effect presence. Choosing another intensity parameterization
    later requires an explicit adapter, rather than silently misreading logits.
    """
    if not 0<no_object_weight<=1:raise ValueError('Invalid absence weight')
    assignment=match_instances(pred,targets) if assignments is None else assignments
    device=pred['presence_logits'].device;batch=len(targets)
    if len(assignment)!=batch:raise ValueError('Assignment batch mismatch')
    parts={key:[] for key in ('objectness','box_l1','box_giou','mask_bce','mask_dice')}
    if 'visibility_logits' in pred:parts['visibility']=[]
    embeddings=[];families=[]
    with torch.autocast(device_type=device.type,enabled=False):
        for i,(target,(pi,ti)) in enumerate(zip(targets,assignment)):
            n=int(target['count'])
            if len(pi)!=n or len(set(pi.tolist()))!=n or sorted(ti.tolist())!=list(range(n)):
                raise ValueError('Incomplete or duplicated instance assignment')
            logits=pred['presence_logits'][i].float();truth=torch.zeros_like(logits);truth[pi]=1
            weights=torch.where(truth.bool(),1.,no_object_weight)
            parts['objectness'].append((F.binary_cross_entropy_with_logits(logits,truth,reduction='none')*weights).sum()/weights.sum())
            boxes=pred['boxes'][i,pi].float();gt=target['boxes_visible'].to(device)[ti].float()
            parts['box_l1'].append(F.l1_loss(boxes,gt))
            parts['box_giou'].append(1-generalized_iou(boxes,gt).diag().mean())
            masks=pred['mask_logits'][i,pi].float();t=target['masks']['visible'].to(device)[ti].float()
            v=target['valid_mask'].to(device).float()
            bce,dice=pairwise_mask_cost(masks,t,v)
            parts['mask_bce'].append(bce.diag().mean());parts['mask_dice'].append(dice.diag().mean())
            if 'visibility' in parts:
                parts['visibility'].append(F.smooth_l1_loss(pred['visibility_logits'][i,pi].float().sigmoid(),target['visibility'].to(device)[ti]))
            if 'font_embeddings' in pred:
                embeddings.append(pred['font_embeddings'][i,pi]);families.append(target['font_family'].to(device)[ti])
        losses={key:torch.stack(values).mean() for key,values in parts.items()}
        count=torch.stack([t['count'].to(device) for t in targets])
        if pred['count_logits'].shape!=(batch,7) or ((count<2)|(count>8)).any():raise ValueError('Count support is 2..8')
        losses['count_ce']=F.cross_entropy(pred['count_logits'].float(),count-2)
        instance_count=pred['presence_logits'].float().sigmoid().sum(-1)
        losses['cardinality']=F.smooth_l1_loss(instance_count,count.float())
        count_mean=(pred['count_logits'].float().softmax(-1)*torch.arange(2,9,device=device)).sum(-1)
        losses['count_consistency']=F.smooth_l1_loss(instance_count,count_mean)
        if embeddings:losses['font_contrastive']=font_contrastive(torch.cat(embeddings),torch.cat(families))
        if 'effect_presence_logits' in pred:
            truth=torch.stack([t['effect_presence'].to(device) for t in targets]).float()
            losses['effect_presence']=F.binary_cross_entropy_with_logits(pred['effect_presence_logits'].float(),truth)
        if 'intensity_predictions' in pred:
            intensity=pred['intensity_predictions'].float()
            if not torch.isfinite(intensity).all() or ((intensity<0)|(intensity>1)).any():raise ValueError('Expected bounded severity values')
            truth=torch.stack([t['effect_intensity'].to(device) for t in targets]).float()
            valid=torch.stack([t['intensity_valid'].to(device) for t in targets]).bool()
            if intensity.shape!=truth.shape or valid.shape!=truth.shape:raise ValueError('Intensity shape mismatch')
            losses['effect_intensity']=(F.smooth_l1_loss(intensity,truth,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
        if 'composition_logits' in pred:
            truth=torch.stack([t['composition'].to(device) for t in targets]).float()
            losses['composition']=F.binary_cross_entropy_with_logits(pred['composition_logits'].float(),truth)
        if 'row_logits' in pred:
            truth=torch.stack([t['rows'].to(device) for t in targets])-1
            losses['rows']=F.cross_entropy(pred['row_logits'].float(),truth)
        appearance,_=appearance_losses(pred,targets,assignment)
        losses.update(appearance)
    if not all(torch.isfinite(v) for v in losses.values()):raise ValueError('Nonfinite task loss')
    return losses,assignment


def aligned_queries(student_assignment,teacher_assignment):
    """Align both query sets through synthetic glyph identity, never query number."""
    sp,st=student_assignment;tp,tt=teacher_assignment
    if len(set(st.tolist()))!=len(st) or len(set(tt.tolist()))!=len(tt):raise ValueError('Duplicated ground-truth assignment')
    student={int(t):int(p) for p,t in zip(sp,st)};teacher={int(t):int(p) for p,t in zip(tp,tt)}
    if set(student)!=set(teacher):raise ValueError('Teacher/student matched different glyph sets')
    order=sorted(student)
    return (torch.tensor([student[t] for t in order],device=sp.device),
            torch.tensor([teacher[t] for t in order],device=tp.device))
