"""Extract visual font prototypes from predicted regions of known TRAIN specimens.

Known library boxes are used only to audit/associate predictions after forward.
They never determine model crops, query count, or embeddings. Target retrieval
uses the same visual selector and never receives these library-only labels.
"""
import math
from collections import Counter
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from .reference_data import reference_batches,reference_collate
from .retrieval import FontIndex,identity
from .checkpoints import rng_state,restore_rng,execution_state,restore_execution


def extraction_policy(value):
    keys={'instance_threshold','visibility_threshold','library_iou_threshold'}
    if set(value)!=keys:raise ValueError('Explicit complete font extraction thresholds required')
    result={k:float(v) for k,v in value.items()}
    if any(not math.isfinite(v) or not 0<=v<=1 for v in result.values()):raise ValueError('Invalid extraction threshold')
    if result['library_iou_threshold']==0:raise ValueError('Library IoU threshold must be positive')
    return result


def visual_regions(pred,instance_threshold,visibility_threshold):
    """Pixel-prediction-only region selector, shared by library and target paths."""
    for value in (instance_threshold,visibility_threshold):
        if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid visual selector threshold')
    presence=pred['presence_logits'].float();visibility=pred['visibility_logits'].float()
    boxes=pred['boxes'].float();embeddings=pred['font_embeddings'].float()
    if (presence.ndim!=2 or visibility.shape!=presence.shape or boxes.shape!=(*presence.shape,4)
        or embeddings.shape!=(*presence.shape,128)):raise ValueError('Font region output shape mismatch')
    if any(not torch.isfinite(x).all() for x in (presence,visibility,boxes,embeddings)):
        raise ValueError('Nonfinite font region predictions')
    if ((boxes<0)|(boxes>1)).any() or (boxes[...,2:]<boxes[...,:2]).any():raise ValueError('Invalid normalized region boxes')
    confidence=presence.sigmoid();visible=visibility.sigmoid()
    selected=(confidence>=instance_threshold)&(visible>=visibility_threshold)&(boxes[...,2:]>boxes[...,:2]).all(-1)
    result=[]
    for i in range(len(presence)):
        ids=torch.nonzero(selected[i]).flatten();vectors=embeddings[i,ids]
        if len(ids) and (vectors.norm(dim=-1)<1e-8).any():raise ValueError('Zero selected font embedding')
        result.append(dict(query_ids=ids,embeddings=torch.nn.functional.normalize(vectors,dim=-1),
                           boxes=boxes[i,ids],quality=(confidence[i,ids]*visible[i,ids])))
    return result


def associate_library_regions(regions,labels,audits,policy):
    """Audit only the library side, with one-to-one matching to known glyphs."""
    policy=extraction_policy(policy)
    if len(regions)!=len(labels) or len(labels)!=len(audits):raise ValueError('Library batch mismatch')
    vectors=[];records=[];reports=[]
    for region,label,audit in zip(regions,labels,audits):
        if label['split']!='train':raise ValueError('Only TRAIN references may build this bank')
        truth=label['boxes_visible'].detach().float().cpu();boxes=region['boxes'].detach().cpu()
        if truth.ndim!=2 or truth.shape[1]!=4 or not len(truth) or not torch.isfinite(truth).all():
            raise ValueError('Invalid library boxes')
        if ((truth<0)|(truth>1)).any() or (truth[:,2:]<=truth[:,:2]).any():raise ValueError('Invalid library box geometry')
        selected=len(boxes);accepted=[]
        if selected:
            inter=(torch.minimum(boxes[:,None,2:],truth[None,:,2:])-torch.maximum(boxes[:,None,:2],truth[None,:,:2])).clamp_min(0).prod(-1)
            union=(boxes[:,2:]-boxes[:,:2]).prod(-1)[:,None]+(truth[:,2:]-truth[:,:2]).prod(-1)[None,:]-inter
            iou=inter/union.clamp_min(1e-8)
            cost=-(iou>=policy['library_iou_threshold']).float()*2-iou/(max(iou.shape)+1)
            pi,ti=linear_sum_assignment(cost.numpy())
            accepted=[(int(p),int(t),float(iou[p,t])) for p,t in zip(pi,ti) if iou[p,t]>=policy['library_iou_threshold']]
        for p,t,iou in accepted:
            q=int(region['query_ids'][p]);record={k:label[k] for k in ('font_key','font_sha256','family_group','style','split','specimen_id')}
            record.update(prototype_id=label['specimen_id']+f'/query_{q:02d}',query_id=q,library_glyph_index=t,
                          source_pixel_sha256=audit['pixel_sha256'],source_stem=audit['stem'],
                          axes=label['axes'],predicted_box=boxes[p].tolist(),quality=float(region['quality'][p]),library_iou=iou)
            vectors.append(region['embeddings'][p].detach().cpu());records.append(record)
        reports.append(dict(specimen_id=label['specimen_id'],font_key=label['font_key'],family_group=label['family_group'],
                            pixel_sha256=audit['pixel_sha256'],selected_regions=selected,accepted_regions=len(accepted),
                            known_library_glyphs=len(truth),missed_library_glyphs=len(truth)-len(accepted),
                            unmatched_selected_regions=selected-len(accepted)))
    return torch.stack(vectors) if vectors else torch.empty(0,128),records,reports


def extract_train_bank(model,dataset,embedding_identity,policy,batch_size=8,workers=0):
    """Full TRAIN-reference extraction; no writes, fitting or checkpoint selection.

    Caller must bind the loaded raw/EMA model to embedding_identity and freeze
    thresholds/source identities. Returns a bank plus complete coverage audit;
    caller decides readiness before publishing it. This never forces six queries.
    """
    contract=identity(embedding_identity);policy=extraction_policy(policy)
    if workers<0:raise ValueError('Invalid worker count')
    device=next(model.parameters()).device
    modes=execution_state(model);rng=rng_state();features=[];records=[];reports=[]
    generator=torch.Generator().manual_seed(731)
    try:
        model.eval()
        loader=DataLoader(dataset,batch_sampler=reference_batches(dataset,batch_size),collate_fn=reference_collate,
                          num_workers=workers,generator=generator,pin_memory=device.type=='cuda')
        with torch.inference_mode():
            for batch in loader:
                if set(batch['inputs'])!={'pixels','valid_pixels'}:raise ValueError('Only pixels/validity may enter model')
                if any(label['split']!='train' for label in batch['library_labels']):raise ValueError('Only TRAIN references may build this bank')
                inputs={k:v.to(device,non_blocking=True) for k,v in batch['inputs'].items()}
                with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):pred=model(**inputs)
                regions=visual_regions(pred,policy['instance_threshold'],policy['visibility_threshold'])
                embeddings,items,rows=associate_library_regions(regions,batch['library_labels'],batch['audit'],policy)
                if len(embeddings):features.append(embeddings)
                records.extend(items);reports.extend(rows)
    finally:restore_execution(modes,model);restore_rng(rng)
    if len(reports)!=len(dataset) or len({r['specimen_id'] for r in reports})!=len(reports):
        raise ValueError('Reference coverage/repeated specimen mismatch')
    font_counts=Counter(r['font_key'] for r in records);family_counts=Counter(r['family_group'] for r in records)
    pixel_families={}
    for row in reports:pixel_families.setdefault(row['pixel_sha256'],set()).add(row['family_group'])
    audit=dict(snapshot_hash=dataset.snapshot_hash,embedding_identity=contract,policy=policy,references=len(reports),
               prototypes=len(records),per_font=dict(sorted(font_counts.items())),per_family=dict(sorted(family_counts.items())),
               missing_fonts=sorted({r['font_key'] for r in reports}-set(font_counts)),
               missing_families=sorted({r['family_group'] for r in reports}-set(family_counts)),
               zero_region_specimens=sum(r['accepted_regions']==0 for r in reports),
               selected_regions=sum(r['selected_regions'] for r in reports),
               missed_library_glyphs=sum(r['missed_library_glyphs'] for r in reports),
               unmatched_selected_regions=sum(r['unmatched_selected_regions'] for r in reports),
               identical_pixel_family_ambiguities={k:sorted(v) for k,v in pixel_families.items() if len(v)>1},
               specimens=reports,calibrated=False,
               limitation='Threshold-filtered library extraction; partial/empty coverage is reported, never filled with ground-truth crops. Source labels are library-only. Cosine similarity is not calibrated confidence.')
    bank=FontIndex(torch.cat(features),records,contract) if features else None
    return bank,audit
