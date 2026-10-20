"""Analytic fixtures for matching, losses, query alignment and counting metrics."""
import copy
from pathlib import Path
import torch
from .common import write,now,sha
from .objectives import match_instances,instance_losses,aligned_queries,font_contrastive,generalized_iou
from .metrics import count_statistics,localization_statistics


def fixture():
    batch,queries,height,width=2,7,16,32
    pred=dict(presence_logits=torch.full((batch,queries),-9.),
              boxes=torch.tensor([0.,0.,.04,.04]).repeat(batch,queries,1),
              mask_logits=torch.full((batch,queries,height,width),-9.),
              count_logits=torch.full((batch,7),-9.),visibility_logits=torch.full((batch,queries),9.),
              font_embeddings=torch.zeros(batch,queries,8),effect_presence_logits=torch.full((batch,37),-9.),
              intensity_predictions=torch.full((batch,37),.4))
    targets=[];expected=[]
    for b,n in enumerate([3,2]):
        masks=torch.zeros(n,height,width);boxes=[];families=torch.tensor([0,0,1] if b==0 else [1,2])
        slots=[4,1,6][:n];expected.append({i:q for i,q in enumerate(slots)})
        for i,q in enumerate(slots):
            x=2+i*8;y=2+b*3;masks[i,y:y+5,x:x+5]=1
            box=torch.tensor([x/width,y/height,(x+5)/width,(y+5)/height])
            boxes.append(box);pred['boxes'][b,q]=box
            pred['presence_logits'][b,q]=9.;pred['mask_logits'][b,q]=masks[i]*18-9
            pred['font_embeddings'][b,q,int(families[i])]=1
        valid=torch.zeros(height,width);valid[:13,:28]=1
        effects=torch.zeros(37);intensity=torch.zeros(37)
        if b==0:
            effects[0]=1;intensity[0]=.4;pred['effect_presence_logits'][b,0]=9.
        targets.append(dict(count=torch.tensor(n),boxes_visible=torch.stack(boxes),masks={'visible':masks},
            valid_mask=valid,visibility=torch.ones(n),font_family=families,effect_presence=effects,
            effect_intensity=intensity,intensity_valid=effects.bool()))
        pred['count_logits'][b,n-2]=9.
    return pred,targets,expected


def check_objectives(run):
    torch.set_num_threads(2);checks={}
    pred,target,expected=fixture();matches=match_instances(pred,target)
    for (pi,ti),mapping in zip(matches,expected):assert {int(t):int(p) for p,t in zip(pi,ti)}==mapping
    losses,_=instance_losses(pred,target,matches)
    assert losses['box_l1']==0 and abs(float(losses['box_giou']))<1e-6
    assert losses['count_ce']<1e-5 and losses['mask_bce']<.001 and losses['mask_dice']<.002
    checks['known_geometry_assignment_and_near_perfect_loss']=True
    permutation=torch.tensor([2,6,0,4,3,1,5]);query_keys=('presence_logits','boxes','mask_logits','visibility_logits','font_embeddings')
    shuffled={k:v[:,permutation] if k in query_keys else v.clone() for k,v in pred.items()}
    altered,tm=instance_losses(shuffled,target)
    assert all(torch.allclose(losses[k],altered[k],atol=1e-6) for k in losses)
    for b,(student,teacher) in enumerate(zip(matches,tm)):
        si,ti=aligned_queries(student,teacher)
        assert torch.equal(pred['boxes'][b,si],shuffled['boxes'][b,ti])
        assert torch.equal(pred['font_embeddings'][b,si],shuffled['font_embeddings'][b,ti])
    checks['query_permutation_invariance_and_teacher_alignment']=True
    reversed_target=copy.deepcopy(target)
    for t in reversed_target:
        for key in ('boxes_visible','visibility','font_family'):t[key]=t[key].flip(0)
        t['masks']['visible']=t['masks']['visible'].flip(0)
    reversed_losses,_=instance_losses(pred,reversed_target)
    assert all(torch.allclose(losses[k],reversed_losses[k],atol=1e-6) for k in losses)
    checks['ground_truth_order_invariance']=True
    padding=copy.deepcopy(pred)
    for b,t in enumerate(target):padding['mask_logits'][b,:,t['valid_mask']==0]=80
    padding_loss,_=instance_losses(padding,target)
    assert torch.allclose(losses['mask_bce'],padding_loss['mask_bce'],atol=1e-6)
    assert torch.allclose(losses['mask_dice'],padding_loss['mask_dice'],atol=1e-6)
    checks['padding_excluded_from_assignment_and_masks']=True
    tracked={k:v.clone().requires_grad_() for k,v in pred.items()}
    differentiable,_=instance_losses(tracked,target);sum(differentiable.values()).backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in tracked.values())
    checks['all_present_task_gradients_finite']=True
    severity=copy.deepcopy(pred);severity['intensity_predictions']=torch.full((2,37),.8,requires_grad=True)
    severity_loss,_=instance_losses(severity,target);severity_loss['effect_intensity'].backward()
    grad=severity['intensity_predictions'].grad
    assert grad[0,0]>0 and torch.count_nonzero(grad)==1
    checks['absent_intensity_has_zero_gradient']=True
    families=torch.tensor([0,0,1,1]);good=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    assert font_contrastive(good,families)<font_contrastive(torch.ones_like(good),families)
    assert font_contrastive(good,torch.zeros(4,dtype=torch.long))==0
    assert font_contrastive(good,torch.arange(4))==0
    checks['font_positive_negative_semantics_and_empty_anchor_handling']=True
    for label,broken in [('nonfinite',copy.deepcopy(pred)),('capacity',{k:v[:,:2] if k in query_keys else v for k,v in pred.items()})]:
        if label=='nonfinite':broken['boxes'][0,0,0]=float('nan')
        try:match_instances(broken,target)
        except ValueError:pass
        else:raise AssertionError('Invalid '+label+' predictions accepted')
    checks['nonfinite_and_insufficient_capacity_refused']=True
    counts=torch.tensor([3,2])
    report=count_statistics(pred['count_logits'],counts,pred['presence_logits'],.5)
    assert report['correct']==2 and report['instances']['count_head_disagreements']==0 and report['nll']<1e-5
    boxes=[t['boxes_visible'] for t in target]
    detection=localization_statistics(pred['boxes'],pred['presence_logits'],boxes,.5)
    assert (detection['true_positive'],detection['false_positive'],detection['false_negative'])==(5,0,0)
    duplicate=copy.deepcopy(pred);duplicate['boxes'][0,0]=target[0]['boxes_visible'][0];duplicate['presence_logits'][0,0]=9
    detection=localization_statistics(duplicate['boxes'],duplicate['presence_logits'],boxes,.5)
    assert (detection['true_positive'],detection['false_positive'],detection['false_negative'])==(5,1,0)
    disagreement=count_statistics(duplicate['count_logits'],counts,duplicate['presence_logits'],.5)
    assert disagreement['correct']==2 and disagreement['instances']['count_head_disagreements']==1
    missed=copy.deepcopy(pred);missed['presence_logits'][0,4]=-9
    detection=localization_statistics(missed['boxes'],missed['presence_logits'],boxes,.5)
    assert (detection['true_positive'],detection['false_positive'],detection['false_negative'])==(4,0,1)
    checks['duplicate_miss_and_count_head_disagreement_reported']=True
    extreme=torch.full((1,7),-1000.);extreme[0,0]=1000.
    assert count_statistics(extreme,torch.tensor([3]))['nll']==2000
    checks['confident_wrong_count_not_hidden_by_nll_clipping']=True
    cuda=False
    if torch.cuda.is_available():
        gpu={k:v.cuda().requires_grad_() for k,v in pred.items()}
        with torch.autocast('cuda',dtype=torch.float16):gpu_losses,_=instance_losses(gpu,target)
        assert all(torch.allclose(losses[k],gpu_losses[k].cpu(),atol=2e-5) for k in losses)
        sum(gpu_losses.values()).backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in gpu.values())
        cuda=True
    checks['cuda_amp_losses_match_cpu_and_gradients_finite']=cuda
    result=dict(created_at=now(),passed=True,checks=checks,
                sources={name:sha(Path(__file__).with_name(name)) for name in ('objectives.py','metrics.py','objective_checks.py')},
                limitation='Analytic predictions only: no decoder trained, no data/architecture changes, no final loss coefficients, threshold selection or specialist grouping fixed. Model-output integration and measured task affinity remain required.')
    write(run/'objective_checks.json',result)
    return result
