"""CPU-only analytic specialist contracts; safe alongside the live GPU trainer."""
import copy
from pathlib import Path
import torch
from .common import write,sha,now
from .objective_checks import fixture
from .appearance import ATTRIBUTE_DIMS
from .data import EFFECT_HEADS
from .specialist_tasks import task_manifest,task_losses,specialist_objective
from .distillation import aligned_distillation
from .objectives import instance_losses


def extended_fixture():
    pred,targets,_=fixture();batch,queries=pred['presence_logits'].shape
    pred.update(composition_logits=torch.zeros(batch,3),row_logits=torch.zeros(batch,2),
                colour_rgb=torch.full((batch,queries,3),.4),background_under=torch.full((batch,3,16,32),.8),
                font_attribute_logits={k:torch.zeros(batch,queries,size) for k,size in ATTRIBUTE_DIMS.items()})
    for t in targets:
        n=int(t['count'])
        t.update(composition=torch.tensor([1.,1.,0.]),rows=torch.tensor(1),colour_rgb=torch.full((n,3),.4),
                 background_under=torch.full((3,16,32),.8)*t['valid_mask'][None],
                 font_attributes=dict(values={k:torch.zeros(n,dtype=torch.long) if k in ('category','stroke') else torch.full((n,),.5)
                                             for k in ATTRIBUTE_DIMS},valid={k:torch.ones(n,dtype=torch.bool) for k in ATTRIBUTE_DIMS}))
        t['font_attributes']['valid']['x_height'][0]=False
    return pred,targets


def tracked(value):
    if torch.is_tensor(value):return value.detach().clone().requires_grad_(value.is_floating_point())
    if isinstance(value,dict):return {k:tracked(v) for k,v in value.items()}
    return copy.deepcopy(value)


def tensors(value):
    if torch.is_tensor(value):yield value
    elif isinstance(value,dict):
        for child in value.values():yield from tensors(child)


def check_specialists(run):
    torch.set_num_threads(2);checks={}
    pred,targets=extended_fixture();losses,support,matches=task_losses(pred,targets)
    assert len(losses)==96 and set(losses)==set(task_manifest())
    assert support['font_contrastive']==4 and support['font_x_height']==3
    assert support['intensity/'+EFFECT_HEADS[0]]==1 and support['intensity/'+EFFECT_HEADS[1]]==0
    checks['all_96_tasks_and_exact_evidence_support']=True
    old,_=instance_losses(pred,targets,matches)
    torch.testing.assert_close(torch.stack([losses['presence/'+k] for k in EFFECT_HEADS]).mean(),old['effect_presence'])
    torch.testing.assert_close(torch.stack([losses['composition/'+k] for k in ('digit','uppercase','lowercase')]).mean(),old['composition'])
    intensity=sum(losses['intensity/'+k]*support['intensity/'+k] for k in EFFECT_HEADS)/sum(support['intensity/'+k] for k in EFFECT_HEADS)
    torch.testing.assert_close(intensity,old['effect_intensity'])
    checks['per_head_losses_reconstruct_original_supported_aggregates']=True
    # Only a selected presence output receives a gradient; mean is over the
    # selected task, not all 37 outputs or a silently attached auxiliary loss.
    p=tracked(pred);values,valid,_=task_losses(p,targets);name='presence/'+EFFECT_HEADS[1]
    loss=specialist_objective(values,valid,[name],[],{'effect_presence':1.},{name:1.})
    torch.testing.assert_close(loss,values[name]);loss.backward()
    assert p['effect_presence_logits'].grad[:,1].abs().sum()>0
    assert torch.count_nonzero(p['effect_presence_logits'].grad)==2
    assert all(v.grad is None for k,v in p.items() if torch.is_tensor(v) and k!='effect_presence_logits')
    checks['unassigned_outputs_excluded_before_group_mean']=True
    p=tracked(pred);values,valid,_=task_losses(p,targets)
    name='intensity/'+EFFECT_HEADS[0]
    with torch.no_grad():p['intensity_predictions'][0,0]=.8
    values,valid,_=task_losses(p,targets)
    specialist_objective(values,valid,[name],[],{'effect_intensity':1.},{name:1.}).backward()
    assert torch.count_nonzero(p['intensity_predictions'].grad)==1
    try:specialist_objective(values,valid,['intensity/'+EFFECT_HEADS[1]],[],{'effect_intensity':1.},{'intensity/'+EFFECT_HEADS[1]:1.})
    except ValueError:pass
    else:raise AssertionError('Empty specialist evidence accepted')
    checks['conditional_intensity_and_empty_evidence_gate']=True
    # Different query slot orders must represent the same teacher.
    permutation=torch.tensor([2,6,0,4,3,1,5]);query_fields={'presence_logits','boxes','mask_logits','visibility_logits','font_embeddings','colour_rgb'}
    teacher={k:({a:v[:,permutation].clone() for a,v in value.items()} if k=='font_attribute_logits' else
                value[:,permutation].clone() if k in query_fields else value.clone()) for k,value in pred.items()}
    terms,counts=aligned_distillation(pred,teacher,targets,list(task_manifest()))
    assert len(terms)==96 and all(float(v)<1e-6 for v in terms.values())
    checks['all_output_types_zero_for_identical_permuted_teacher']=True
    # Teacher detachment and sparse student gradients under a nonzero discrepancy.
    student=tracked(pred);teacher=tracked(teacher)
    with torch.no_grad():teacher['effect_presence_logits'][:,1]=3.
    name='presence/'+EFFECT_HEADS[1]
    soft,valid=aligned_distillation(student,teacher,targets,[name])
    assert list(soft)==[name] and float(soft[name])>0;soft[name].backward()
    assert all(t.grad is None for t in tensors(teacher))
    assert torch.count_nonzero(student['effect_presence_logits'].grad)==2
    assert all(t.grad is None for k,t in student.items() if torch.is_tensor(t) and k!='effect_presence_logits')
    checks['teacher_detached_and_only_assigned_student_output_distilled']=True
    # Mask padding and unknown intrinsic metadata cannot affect the loss.
    teacher=copy.deepcopy(pred)
    for b,t in enumerate(targets):teacher['mask_logits'][b,:,t['valid_mask']==0]=71.
    terms,_=aligned_distillation(pred,teacher,targets,['mask_bce','mask_dice'])
    assert all(float(v)==0 for v in terms.values())
    bad=copy.deepcopy(pred)
    for b,(_,ti) in enumerate(matches):
        slot=int(matches[b][0][torch.nonzero(ti==0).item()]);bad['font_attribute_logits']['x_height'][b,slot]=20.
    terms,s=aligned_distillation(pred,bad,targets,['font_x_height'])
    assert float(terms['font_x_height'])==0 and s['font_x_height']==3
    checks['padding_and_unknown_font_metadata_excluded']=True
    # Nonzero aligned instance losses are unchanged by a teacher permutation.
    teacher=copy.deepcopy(pred);teacher['font_embeddings']=torch.roll(teacher['font_embeddings'],1,-1)
    original,_=aligned_distillation(pred,teacher,targets,['font_contrastive'])
    teacher['font_embeddings']=teacher['font_embeddings'][:,permutation]
    for k in query_fields-{'font_embeddings'}:teacher[k]=teacher[k][:,permutation]
    for k in teacher['font_attribute_logits']:teacher['font_attribute_logits'][k]=teacher['font_attribute_logits'][k][:,permutation]
    shuffled,_=aligned_distillation(pred,teacher,targets,['font_contrastive'])
    torch.testing.assert_close(original['font_contrastive'],shuffled['font_contrastive'])
    assert original['font_contrastive']>0
    checks['nonzero_font_distillation_uses_glyph_identity']=True
    try:aligned_distillation(pred,pred,targets,['unassigned_unknown'])
    except ValueError:pass
    else:raise AssertionError('Unknown teacher task accepted')
    incomplete=[(a[:-1],b[:-1]) for a,b in matches]
    try:aligned_distillation(pred,pred,targets,['count_ce'],student_assignment=incomplete)
    except ValueError:pass
    else:raise AssertionError('Incomplete alignment accepted')
    checks['unknown_tasks_and_incomplete_alignment_refused']=True
    result=dict(created_at=now(),passed=True,device='cpu',checks=checks,task_terms=len(task_manifest()),
                task_groups=task_manifest(),sources={name:sha(Path(__file__).with_name(name)) for name in
                     ('specialist_tasks.py','distillation.py','specialist_checks.py')},
                limitation='Analytic CPU fixtures only. No teacher selection, actual TAG groups, font index, or specialist training. Live baseline and its frozen source files were not changed.')
    write(run/'specialist_contract_checks.json',result)
    return result
