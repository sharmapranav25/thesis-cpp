"""Joint font supervision from synthetic scenes and known TRAIN specimens.

Labels associate predictions after pixel-only forward. No library box/count
chooses a model crop. The caller freezes sampling, thresholds and temperature
in its phase protocol; this module does not launch or choose a training phase.
"""
import math
import torch
from torch.nn import functional as F
from .font_extraction import visual_regions,associate_library_regions,extraction_policy
from .specialist_tasks import task_losses


def reference_vectors(pred,batch,family_indices,policy):
    """Keep gradients in accepted predicted embeddings; association is detached."""
    policy=extraction_policy(policy)
    labels=batch['library_labels'];audits=batch['audit']
    if len(labels)!=len(pred['font_embeddings']) or len(audits)!=len(labels):
        raise ValueError('Reference batch mismatch')
    if (not family_indices or any(type(v) is not int or v<0 for v in family_indices.values())
        or len(set(family_indices.values()))!=len(family_indices)):
        raise ValueError('Unique nonnegative family indices required')
    for label in labels:
        if label['split']!='train' or label['family_group'] not in family_indices:
            raise ValueError('Reference requires a known TRAIN family')
    ids=[label['specimen_id'] for label in labels]
    if len(set(ids))!=len(ids):raise ValueError('Repeated reference specimen in one batch')
    regions=visual_regions(pred,policy['instance_threshold'],policy['visibility_threshold'])
    # The extraction adapter audits one-to-one IoU without a differentiable box
    # association. Re-index the original graph instead of its detached index bank.
    _,records,reports=associate_library_regions(regions,labels,audits,policy)
    lookup={key:i for i,key in enumerate(ids)}
    values=[pred['font_embeddings'][lookup[r['specimen_id']],r['query_id']] for r in records]
    vectors=torch.stack(values) if values else pred['font_embeddings'].reshape(-1,128)[:0]
    families=torch.tensor([family_indices[r['family_group']] for r in records],device=vectors.device,dtype=torch.long)
    # Exact whole-specimen pixel equality plus glyph identity, not whole-family
    # merging. Duplicate renderings are neither easy positives nor false negatives.
    duplicate_keys=[(r['source_pixel_sha256'],r['library_glyph_index']) for r in records]
    return vectors,families,duplicate_keys,reports


def contrastive_with_references(vectors,families,duplicate_keys,domains,temperature):
    """Supervised contrastive mean with explicit eligibility and duplicate masks."""
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Positive finite temperature required')
    n=len(vectors)
    if (vectors.ndim!=2 or vectors.shape[1]!=128 or families.shape!=(n,) or families.dtype!=torch.long
        or families.device!=vectors.device or len(duplicate_keys)!=n or len(domains)!=n):
        raise ValueError('Font contrastive shape/device mismatch')
    if not torch.isfinite(vectors).all() or (families<0).any():raise ValueError('Invalid font contrastive values')
    if n and (vectors.float().norm(dim=-1)<1e-8).any():raise ValueError('Zero font embedding')
    if any(d not in ('scene','library') for d in domains):raise ValueError('Unknown font domain')
    if any(k is not None and (not isinstance(k,tuple) or len(k)!=2 or not isinstance(k[0],str)
                              or not k[0] or type(k[1]) is not int or k[1]<0) for k in duplicate_keys):
        raise ValueError('Invalid reference pixel/glyph identity')
    duplicate=torch.tensor([[i!=j and a is not None and a==b for j,b in enumerate(duplicate_keys)]
                            for i,a in enumerate(duplicate_keys)],device=vectors.device,dtype=torch.bool).reshape(n,n)
    same=families[:,None]==families[None,:]
    same_specimen=torch.tensor([[a is not None and b is not None and a[0]==b[0] for b in duplicate_keys]
                               for a in duplicate_keys],device=vectors.device,dtype=torch.bool).reshape(n,n)
    # All cross-family glyph pairs from an identical specimen are ambiguous:
    # otherwise glyph A in one alias can attract B in its own family while being
    # forced away from pixel-identical B in the other alias. Do not merge families.
    ambiguous=same_specimen&~same
    allowed=~(torch.eye(n,device=vectors.device,dtype=torch.bool)|duplicate|ambiguous)
    positive=same&allowed;negative=~same&allowed
    eligible=positive.any(-1)&negative.any(-1)
    support=int(eligible.sum())
    cross=torch.tensor([[a!=b for b in domains] for a in domains],device=vectors.device,dtype=torch.bool).reshape(n,n)
    audit=dict(eligible_anchors=support,scene_vectors=domains.count('scene'),library_vectors=domains.count('library'),
               duplicate_pairs_excluded=int(duplicate.sum())//2,
               ambiguous_specimen_pairs_excluded=int(ambiguous.sum())//2,
               total_excluded_pairs=int((duplicate|ambiguous).sum())//2,
               cross_domain_positive_anchors=int((eligible&(positive&cross).any(-1)).sum()))
    if not support:return vectors.sum()*0,audit
    with torch.autocast(device_type=vectors.device.type,enabled=False):
        z=F.normalize(vectors.float(),dim=-1)
        # Evaluate only eligible rows: an excluded singleton must never produce
        # an all-masked logsumexp whose NaN gradient leaks through an unused row.
        logits=(z[eligible]@z.T)/temperature
        log_prob=logits-torch.logsumexp(logits.masked_fill(~allowed[eligible],-torch.inf),dim=-1,keepdim=True)
        loss=-(log_prob.masked_fill(~positive[eligible],0).sum(-1)/positive[eligible].sum(-1)).mean()
    if not torch.isfinite(loss):raise ValueError('Nonfinite font objective')
    return loss,audit


def joint_task_forward(model,scene_batch,reference_batch,family_indices,policy,temperature):
    """Return the existing 96 tasks, adding library evidence to font contrastive.

The caller supplies already-device-placed batches and performs weighting,
auxiliary localization supervision, optimizer/EMA/resume and phase selection.
No mutable sampling cursor is consumed here, so same-batch AMP retry is safe.
GPU memory/throughput readiness is required before production integration.
"""
    references=[reference_batch] if isinstance(reference_batch,dict) else list(reference_batch)
    if not references:raise ValueError('Explicit nonempty reference batches required')
    for batch in [scene_batch]+references:
        if set(batch['inputs'])!={'pixels','valid_pixels'}:raise ValueError('Only pixels and validity enter the model')
    # Refuse held-out/unknown references before either model forward.
    labels=[t for batch in references for t in batch['library_labels']]
    if any(t['split']!='train' or t['family_group'] not in family_indices for t in labels):
        raise ValueError('Only known TRAIN references may train font embeddings')
    if len({t['specimen_id'] for t in labels})!=len(labels):raise ValueError('Repeated reference specimen across batches')
    pred=model(**scene_batch['inputs'])
    losses,support,assignment=task_losses(pred,scene_batch['targets'])
    vectors=[];family_parts=[];keys=[];reports=[]
    for batch in references:
        library_pred=model(**batch['inputs'])
        values,ids,identities,rows=reference_vectors(library_pred,batch,family_indices,policy)
        vectors.append(values);family_parts.append(ids);keys.extend(identities);reports.extend(rows)
    library=torch.cat(vectors);families=torch.cat(family_parts)
    scene=torch.cat([pred['font_embeddings'][i,pi] for i,(pi,_) in enumerate(assignment)])
    scene_families=torch.cat([t['font_family'].to(scene.device)[ti] for t,(_,ti) in zip(scene_batch['targets'],assignment)])
    if not set(scene_families.tolist()).issubset(set(family_indices.values())):
        raise ValueError('Scene family mapping differs from TRAIN reference mapping')
    value,audit=contrastive_with_references(torch.cat((scene,library)),torch.cat((scene_families,families)),
        [None]*len(scene)+keys,['scene']*len(scene)+['library']*len(library),temperature)
    losses['font_contrastive']=value;support['font_contrastive']=audit['eligible_anchors']
    audit['library_specimens']=reports
    audit['reference_forwards']=len(references)
    return dict(losses=losses,support=support,assignments=assignment,scene_prediction=pred,font_audit=audit)
