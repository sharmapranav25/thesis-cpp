"""Bounded DEV count/localization and unseen-family retrieval diagnostics."""
import collections
import time
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from .data import ShapeBatchPlan,exact_shape_collate
from .reference_data import reference_collate
from .font_extraction import visual_regions
from .metrics import count_statistics,localization_statistics
from .common import write,now


def association(boxes,truth):
    boxes=boxes.detach().float().cpu();truth=truth.detach().float().cpu()
    if not len(boxes):return []
    inter=(torch.minimum(boxes[:,None,2:],truth[None,:,2:])-torch.maximum(boxes[:,None,:2],truth[None,:,:2])).clamp_min(0).prod(-1)
    union=(boxes[:,2:]-boxes[:,:2]).prod(-1)[:,None]+(truth[:,2:]-truth[:,:2]).prod(-1)[None]-inter
    iou=inter/union.clamp_min(1e-8);p,t=linear_sum_assignment((-(iou>=.5).float()*2-iou/(max(iou.shape)+1)).numpy())
    return [(int(a),int(b)) for a,b in zip(p,t) if iou[a,b]>=.5]


def gallery_rows(train,dev):
    selected=[]
    for dataset,split in ((train,'train'),(dev,'dev')):
        groups=collections.defaultdict(list)
        for i,row in enumerate(dataset.rows):
            if row['record']['split']!=split:raise ValueError('Gallery split changed')
            groups[row['record']['font_key']].append(i)
        for key,indices in sorted(groups.items()):
            # First three deterministic pages per file; bounded candidate bank.
            # No query labels, text or accuracy determine these choices.
            for i in sorted(indices,key=lambda i:dataset.rows[i]['record']['specimen_id'])[:3]:selected.append((dataset,i))
    return selected


@torch.no_grad()
def evaluate(model,dev,train_refs,dev_refs,destination,deadline,workers=4):
    def budget():
        if time.time()>deadline:raise TimeoutError('Pilot evaluation deadline reached')
    model.eval();device=next(model.parameters()).device;vectors=[];labels=[];seen=[];missed=0
    rows=gallery_rows(train_refs,dev_refs);expected_families={ds.rows[i]['record']['family_group'] for ds,i in rows}
    buckets=collections.defaultdict(list)
    for ds,i in rows:buckets[tuple(ds.rows[i]['shape'])].append((ds,i))
    for _,items in sorted(buckets.items()):
        for start in range(0,len(items),8):
            budget();batch=reference_collate([ds[i] for ds,i in items[start:start+8]])
            inputs={k:v.to(device) for k,v in batch['inputs'].items()}
            with torch.autocast(device.type,dtype=torch.float16):pred=model(**inputs)
            regions=visual_regions(pred,.5,.25)
            for region,label in zip(regions,batch['library_labels']):
                pairs=association(region['boxes'],label['boxes_visible']);missed+=len(label['boxes_visible'])-len(pairs)
                for p,t in pairs:vectors.append(region['embeddings'][p].cpu());labels.append(label['family_group'])
                seen.append(label['specimen_id'])
    families=sorted(expected_families);family_id={f:i for i,f in enumerate(families)}
    bank=torch.stack(vectors).to(device) if vectors else torch.empty((0,128),device=device)
    ids=torch.tensor([family_id[f] for f in labels],device=device);counts=[];logits=[];presence=[];boxes=[];truth=[]
    correct=top5=matched=total=0;images=0;perfamily=collections.defaultdict(lambda:[0,0,0])
    plan=ShapeBatchPlan(dev,8,epoch=0,shuffle=False)
    generator=torch.Generator().manual_seed(731)
    loader=DataLoader(dev,batch_sampler=plan,collate_fn=exact_shape_collate,num_workers=workers,pin_memory=True,generator=generator)
    for batch in loader:
        budget();inputs={k:v.to(device) for k,v in batch['inputs'].items()}
        with torch.autocast(device.type,dtype=torch.float16):pred=model(**inputs)
        regions=visual_regions(pred,.5,.25)
        # Search only predicted visual vectors. Ground truth is accessed below
        # solely for scoring after all candidates have already been ranked.
        ranks=[]
        for region in regions:
            if len(bank) and len(region['embeddings']):
                similarity=region['embeddings'].float()@bank.T
                scores=torch.full((len(similarity),len(families)),-torch.inf,device=device)
                scores.scatter_reduce_(1,ids[None].expand(len(similarity),-1),similarity,reduce='amax',include_self=True)
                # Stable alphabetical tie breaking; no calibration/threshold fit.
                ranked=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:5].cpu().tolist()
                ranks.append([[families[k] for k in row if torch.isfinite(scores[i,k])] for i,row in enumerate(ranked)])
            else:ranks.append([[] for _ in region['query_ids']])
        inverse={v:k for k,v in dev.family_indices.items()}
        for region,candidates,target in zip(regions,ranks,batch['targets']):
            target_families=[inverse[int(x)] for x in target['font_family']];total+=len(target_families)
            for f in target_families:perfamily[f][2]+=1
            for p,t in association(region['boxes'],target['boxes_visible']):
                family=target_families[t];matched+=1;hit=bool(candidates[p]) and candidates[p][0]==family
                correct+=int(hit);top5+=int(family in candidates[p]);perfamily[family][0]+=int(hit);perfamily[family][1]+=1
        counts.extend(int(t['count']) for t in batch['targets']);truth.extend(t['boxes_visible'] for t in batch['targets'])
        logits.append(pred['count_logits'].float().cpu());presence.append(pred['presence_logits'].float().cpu());boxes.append(pred['boxes'].float().cpu())
        images+=len(batch['targets'])
        if images%200<8:write(destination.with_name(destination.stem+'_progress.json'),dict(state='evaluating',images=images,total=len(dev)))
    assert images==len(dev) and len(seen)==len(rows) and len(set(seen))==len(seen)
    result=dict(created_at=now(),images=images,split='dev',snapshot=dev.snapshot_hash,
        count=count_statistics(torch.cat(logits),counts,torch.cat(presence),.5),
        localization=localization_statistics(torch.cat(boxes),torch.cat(presence),truth,.5),
        font=dict(total_glyphs=total,matched_glyphs=matched,correct_top1=correct,correct_top5=top5,
            end_to_end_top1=correct/max(1,total),end_to_end_top5=top5/max(1,total),
            conditional_top1=correct/matched if matched else None,conditional_top5=top5/matched if matched else None,
            per_family=dict(perfamily),candidate_families=len(families),represented_families=len(set(labels)),
            missing_families=sorted(expected_families-set(labels)),library_specimens=len(rows),prototypes=len(labels),missed_library_glyphs=missed,
            policy='All TRAIN and DEV font files, first three sorted specimens each; predicted regions only, presence>=0.5 visibility>=0.25; library/query scoring IoU>=0.5; max cosine per family, stable ties.'),
        limitation='Bounded synthetic DEV pilot, not final test or calibrated real-world accuracy. Missing localizations count as failures in end-to-end retrieval. Sparse library glyph/style coverage may limit retrieval.')
    write(destination,result);model.train();return result
