"""Count and localization diagnostics; no threshold fitting or checkpoint selection."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def count_statistics(logits,counts,presence_logits=None,instance_threshold=None,bins=15):
    logits=torch.as_tensor(logits).detach().float().cpu()
    counts=torch.as_tensor(counts).detach().long().cpu()
    if logits.shape!=(len(counts),7) or not len(counts) or ((counts<2)|(counts>8)).any() or not torch.isfinite(logits).all():
        raise ValueError('Expected finite count logits on support 2..8')
    if bins<1:raise ValueError('Invalid calibration bin count')
    probs=logits.softmax(-1);estimate=probs.argmax(-1)+2
    confidence=probs.max(-1).values;correct=estimate==counts
    matrix=torch.zeros(7,7,dtype=torch.long)
    for truth,guess in zip(counts,estimate):matrix[truth-2,guess-2]+=1
    onehot=torch.nn.functional.one_hot(counts-2,7).float()
    rows=[];ece=0.
    bucket=torch.clamp((confidence*bins).long(),max=bins-1)
    for index in range(bins):
        chosen=bucket==index;n=int(chosen.sum())
        accuracy=float(correct[chosen].float().mean()) if n else None
        conf=float(confidence[chosen].mean()) if n else None
        rows.append(dict(bin=index,n=n,accuracy=accuracy,confidence=conf))
        if n:ece+=n/len(counts)*abs(accuracy-conf)
    report=dict(n=len(counts),correct=int(correct.sum()),accuracy=float(correct.float().mean()),
                mae=float((estimate-counts).abs().float().mean()),
                nll=float(-logits.log_softmax(-1)[torch.arange(len(counts)),counts-2].mean()),
                brier=float((probs-onehot).square().sum(-1).mean()),ece=ece,calibration_bins=rows,
                support=list(range(2,9)),confusion_true_rows_predicted_columns=matrix.tolist(),
                per_length={str(length):dict(n=int((counts==length).sum()),
                    correct=int(correct[counts==length].sum()),
                    accuracy=float(correct[counts==length].float().mean()) if (counts==length).any() else None)
                    for length in range(2,9)})
    if presence_logits is not None:
        if instance_threshold is None or not 0<=instance_threshold<=1:
            raise ValueError('Supply an explicit instance threshold; this diagnostic never fits one')
        presence=torch.as_tensor(presence_logits).detach().float().cpu()
        if presence.ndim!=2 or len(presence)!=len(counts) or not torch.isfinite(presence).all():raise ValueError('Invalid instance logits')
        expected=presence.sigmoid().sum(-1);hard=(presence.sigmoid()>=instance_threshold).sum(-1)
        report['instances']=dict(threshold=instance_threshold,expected_mae=float((expected-counts).abs().mean()),
            hard_correct=int((hard==counts).sum()),hard_accuracy=float((hard==counts).float().mean()),
            hard_mae=float((hard-counts).abs().float().mean()),count_head_disagreements=int((hard!=estimate).sum()),
            hard_below_support=int((hard<2).sum()),hard_above_support=int((hard>8).sum()))
    return report


def localization_statistics(boxes,presence_logits,target_boxes,instance_threshold,iou_threshold=.5):
    if not 0<=instance_threshold<=1 or not 0<iou_threshold<=1:raise ValueError('Invalid explicit thresholds')
    boxes=torch.as_tensor(boxes).detach().float().cpu();presence=torch.as_tensor(presence_logits).detach().float().cpu()
    if boxes.shape!=(*presence.shape,4) or len(boxes)!=len(target_boxes) or not torch.isfinite(boxes).all() or not torch.isfinite(presence).all():
        raise ValueError('Invalid localization predictions')
    tp=fp=fn=0;matched_ious=[]
    for predicted,scores,truth in zip(boxes,presence,target_boxes):
        predicted=predicted[scores.sigmoid()>=instance_threshold]
        truth=torch.as_tensor(truth).float().cpu()
        if not len(predicted):fn+=len(truth);continue
        if not len(truth):fp+=len(predicted);continue
        inter=(torch.minimum(predicted[:,None,2:],truth[None,:,2:])-torch.maximum(predicted[:,None,:2],truth[None,:,:2])).clamp_min(0).prod(-1)
        union=(predicted[:,2:]-predicted[:,:2]).clamp_min(0).prod(-1)[:,None]+(truth[:,2:]-truth[:,:2]).clamp_min(0).prod(-1)[None,:]-inter
        iou=inter/union.clamp_min(1e-8)
        # Maximize accepted match COUNT first, then IoU. Pure summed IoU matching
        # can choose fewer above-threshold matches and misreport misses/duplicates.
        cost=-(iou>=iou_threshold).float()*2-iou/(max(iou.shape)+1)
        pi,ti=linear_sum_assignment(cost.numpy());values=iou[pi,ti]
        good=values>=iou_threshold;found=int(good.sum())
        tp+=found;fp+=len(predicted)-found;fn+=len(truth)-found;matched_ious.extend(values[good].tolist())
    return dict(true_positive=tp,false_positive=fp,false_negative=fn,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                mean_matched_iou=float(np.mean(matched_ious)) if matched_ious else None,
                instance_threshold=instance_threshold,iou_threshold=iou_threshold)
