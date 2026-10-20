"""Pilot-only batched matching/localization; original trainer stays immutable."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .objectives import font_contrastive
from .appearance import appearance_losses


def packed(targets,device):
    counts=[len(t['boxes_visible']) for t in targets];n=max(counts)
    if any(c!=int(t['count']) or not 2<=c<=8 for c,t in zip(counts,targets)):raise ValueError('Invalid target counts')
    def pad(key):
        values=[key(t).to(device).float() for t in targets]
        return torch.stack([torch.cat((v,v.new_zeros((n-len(v),*v.shape[1:])))) for v in values])
    boxes=pad(lambda t:t['boxes_visible']);masks=pad(lambda t:t['masks']['visible']).flatten(2)
    valid=torch.stack([t['valid_mask'].to(device).float().flatten() for t in targets])
    active=torch.arange(n,device=device)[None]<torch.tensor(counts,device=device)[:,None]
    checks=torch.stack([torch.isfinite(boxes).all(),torch.isfinite(masks).all(),torch.isfinite(valid).all(),
        ((boxes[...,2:]>boxes[...,:2]).all(-1)|~active).all(),(valid.sum(-1)>0).all(),
        ((masks>=0)&(masks<=1)).all(),((valid>=0)&(valid<=1)).all()])
    if not checks.all():raise ValueError('Invalid packed supervision')
    return counts,boxes,masks,valid,active


def giou(a,b):
    lower=torch.maximum(a[...,:2],b[...,:2]);upper=torch.minimum(a[...,2:],b[...,2:])
    inter=(upper-lower).clamp_min(0).prod(-1)
    union=(a[...,2:]-a[...,:2]).clamp_min(0).prod(-1)+(b[...,2:]-b[...,:2]).clamp_min(0).prod(-1)-inter
    outer=(torch.maximum(a[...,2:],b[...,2:])-torch.minimum(a[...,:2],b[...,:2])).clamp_min(0).prod(-1)
    return inter/union.clamp_min(1e-8)-(outer-union)/outer.clamp_min(1e-8)


@torch.no_grad()
def match(pred,targets,pack=None):
    presence=pred['presence_logits'].float();boxes=pred['boxes'].float();masks=pred['mask_logits'].float().flatten(2)
    counts,truth,t,valid,active=packed(targets,boxes.device) if pack is None else pack
    if max(counts)>boxes.shape[1]:raise ValueError('Insufficient queries')
    if not torch.stack([torch.isfinite(x).all() for x in (presence,boxes,masks)]).all():raise ValueError('Nonfinite predictions')
    if (boxes[...,2:]<boxes[...,:2]).any():raise ValueError('Inverted prediction')
    v=valid[:,None];area=valid.sum(-1)[:,None,None].clamp_min(1)
    p=masks.sigmoid()
    bce=(F.softplus(masks)*v).sum(-1)[:,:,None]/area-torch.bmm(masks,(t*v).transpose(1,2))/area
    dice=1-(2*torch.bmm(p*v,t.transpose(1,2))+1)/((p*v).sum(-1)[:,:,None]+(t*v).sum(-1)[:,None]+1)
    cost=-presence.sigmoid()[:,:,None]+5*torch.cdist(boxes,truth,p=1)-2*giou(boxes[:,:,None],truth[:,None])+bce+dice
    if not torch.isfinite(cost).all():raise ValueError('Nonfinite assignment cost')
    cpu=cost.cpu().numpy()  # One batch transfer, exact SciPy assignment retained.
    pairs=[linear_sum_assignment(cpu[i,:,:n]) for i,n in enumerate(counts)]
    return [(torch.as_tensor(p,device=boxes.device),torch.as_tensor(t,device=boxes.device)) for p,t in pairs]


def losses(pred,targets,assignments=None,no_object_weight=.1):
    if not 0<no_object_weight<=1:raise ValueError('Invalid absence weight')
    device=pred['presence_logits'].device
    with torch.autocast(device.type,enabled=False):
        pack=packed(targets,device);counts,truth,t,valid,active=pack;b=len(targets);n=truth.shape[1]
        assignment=match(pred,targets,pack) if assignments is None else assignments
        # Keep original target order; GPU gathers replace per-image loss graphs.
        lookup=torch.zeros((b,n),dtype=torch.long,device=device)
        for i,(pi,ti) in enumerate(assignment):
            if len(pi)!=counts[i] or sorted(ti.tolist())!=list(range(counts[i])) or len(set(pi.tolist()))!=len(pi):
                raise ValueError('Incomplete/repeated assignment')
            lookup[i,ti]=pi
        batch=torch.arange(b,device=device)[:,None];div=torch.tensor(counts,device=device).float()
        def mean_glyph(x):return ((x*active).sum(-1)/div).mean()
        presence=pred['presence_logits'].float();binary=torch.zeros_like(presence)
        for i,(pi,_) in enumerate(assignment):binary[i,pi]=1
        weights=torch.where(binary.bool(),1.,no_object_weight)
        result={'objectness':((F.binary_cross_entropy_with_logits(presence,binary,reduction='none')*weights).sum(-1)/weights.sum(-1)).mean()}
        boxes=pred['boxes'][batch,lookup].float()
        result['box_l1']=mean_glyph((boxes-truth).abs().mean(-1));result['box_giou']=mean_glyph(1-giou(boxes,truth))
        masks=pred['mask_logits'][batch,lookup].float().flatten(2);v=valid[:,None];area=v.sum(-1).clamp_min(1)
        result['mask_bce']=mean_glyph(((F.softplus(masks)-masks*t)*v).sum(-1)/area)
        p=masks.sigmoid();result['mask_dice']=mean_glyph(1-(2*(p*t*v).sum(-1)+1)/((p*v).sum(-1)+(t*v).sum(-1)+1))
        if 'visibility_logits' in pred:
            target=torch.stack([F.pad(x['visibility'].to(device),(0,n-counts[i])) for i,x in enumerate(targets)])
            result['visibility']=mean_glyph(F.smooth_l1_loss(pred['visibility_logits'][batch,lookup].float().sigmoid(),target,reduction='none'))
        count=torch.tensor(counts,device=device);instance_count=presence.sigmoid().sum(-1)
        result['count_ce']=F.cross_entropy(pred['count_logits'].float(),count-2)
        result['cardinality']=F.smooth_l1_loss(instance_count,count.float())
        count_mean=(pred['count_logits'].float().softmax(-1)*torch.arange(2,9,device=device)).sum(-1)
        result['count_consistency']=F.smooth_l1_loss(instance_count,count_mean)
        if 'font_embeddings' in pred:
            result['font_contrastive']=font_contrastive(torch.cat([pred['font_embeddings'][i,pi] for i,(pi,_) in enumerate(assignment)]),
                torch.cat([target['font_family'].to(device)[ti] for target,(_,ti) in zip(targets,assignment)]))
        for output,target,name,kind in [('effect_presence_logits','effect_presence','effect_presence','binary'),('composition_logits','composition','composition','binary')]:
            if output in pred:result[name]=F.binary_cross_entropy_with_logits(pred[output].float(),torch.stack([t[target].to(device) for t in targets]).float())
        if 'intensity_predictions' in pred:
            prediction=pred['intensity_predictions'].float();y=torch.stack([t['effect_intensity'].to(device) for t in targets]);v=torch.stack([t['intensity_valid'].to(device) for t in targets])
            if ((prediction<0)|(prediction>1)).any():raise ValueError('Invalid severity')
            result['effect_intensity']=(F.smooth_l1_loss(prediction,y,reduction='none')*v).sum()/v.sum().clamp_min(1)
        if 'row_logits' in pred:result['rows']=F.cross_entropy(pred['row_logits'].float(),torch.stack([t['rows'].to(device) for t in targets])-1)
        extra,_=appearance_losses(pred,targets,assignment);result.update(extra)
        if not torch.stack([torch.isfinite(x) for x in result.values()]).all():raise ValueError('Nonfinite task loss')
        return result,assignment


def install():
    from . import specialist_tasks,joint_phase,joint_evaluation,training
    specialist_tasks.instance_losses=losses;joint_phase.instance_losses=losses;training.instance_losses=losses


def check():
    from .objective_checks import fixture
    from .objectives import instance_losses
    torch.manual_seed(918);devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
    reports=[]
    for device in devices:
        for noise in (0.,.05):
            pred,targets,_=fixture()
            # Keep bounded outputs in range and leave box geometry valid.
            for k in ('presence_logits','mask_logits','count_logits'):pred[k]=pred[k]+torch.randn_like(pred[k])*noise
            a={k:v.to(device).detach().requires_grad_() for k,v in pred.items()};z={k:v.detach().clone().requires_grad_() for k,v in a.items()}
            old,om=instance_losses(a,targets);new,nm=losses(z,targets)
            for (p,t),(q,u) in zip(om,nm):assert torch.equal(p,q) and torch.equal(t,u)
            for k in old:torch.testing.assert_close(old[k],new[k],atol=2e-6,rtol=2e-5)
            sum(old.values()).backward();sum(new.values()).backward()
            for k in a:torch.testing.assert_close(a[k].grad,z[k].grad,atol=2e-6,rtol=2e-4)
            reports.append(dict(device=device,noise=noise,losses_and_gradients_match=True))
    return reports
