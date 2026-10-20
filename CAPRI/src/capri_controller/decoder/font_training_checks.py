"""CPU gradient and label-boundary checks for later joint font training."""
import copy
from pathlib import Path
import torch
from .common import now,sha,write
from .specialist_checks import extended_fixture
from .specialist_tasks import task_losses
from .font_extraction_checks import ToyLibrary,BOXES
from .reference_data import reference_collate
from .font_training import joint_task_forward,reference_vectors,contrastive_with_references


class ToyJoint(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scene,self.targets=extended_fixture();self.calls=[]
        self.scene_font=torch.nn.Parameter(torch.randn(2,7,128))
        self.library_font=torch.nn.Parameter(torch.randn(2,4,128))

    def forward(self,pixels,valid_pixels):
        self.calls.append(tuple(pixels.shape))
        if float(pixels[0,0,0,0])==2:
            return dict(self.scene,font_embeddings=self.scene_font)
        pred=dict(presence_logits=torch.full((2,4),8.,requires_grad=True),
            visibility_logits=torch.tensor([[8.,8.,8.,-8.]]).repeat(2,1).requires_grad_(),
            boxes=BOXES[None].repeat(2,1,1).requires_grad_(),font_embeddings=self.library_font,
            count_logits=torch.tensor([[0.,0.,0.,0.,0.,0.,99.]]).repeat(2,1))
        self.library_prediction=pred
        return pred


def check_font_training(run):
    if torch.cuda.is_available():raise RuntimeError('Hide CUDA for joint font checks')
    torch.set_num_threads(2);torch.manual_seed(819);checks={}
    model=ToyJoint();library=reference_collate([ToyLibrary()[i] for i in range(2)])
    scene=dict(inputs=dict(pixels=torch.full((2,3,16,32),2.),valid_pixels=torch.ones(2,16,32,dtype=torch.bool)),
               targets=model.targets)
    families=dict(alpha=0,beta=1,gamma=2)
    policy=dict(instance_threshold=.5,visibility_threshold=.5,library_iou_threshold=.5)
    result=joint_task_forward(model,scene,library,families,policy,.1)
    assert len(result['losses'])==96 and len(model.calls)==2
    original,support,_=task_losses(result['scene_prediction'],model.targets,result['assignments'])
    for key in original:
        if key!='font_contrastive':
            torch.testing.assert_close(original[key],result['losses'][key]);assert support[key]==result['support'][key]
    assert result['font_audit']['library_vectors']==4
    assert result['font_audit']['cross_domain_positive_anchors']>0
    assert result['font_audit']['duplicate_pairs_excluded']==2
    assert result['font_audit']['ambiguous_specimen_pairs_excluded']==4
    checks['pixel_only_forward_existing_96_tasks_other_losses_unchanged']=True
    result['losses']['font_contrastive'].backward()
    assert model.scene_font.grad.norm()>0 and model.library_font.grad.norm()>0
    assert torch.isfinite(model.scene_font.grad).all() and torch.isfinite(model.library_font.grad).all()
    assert model.library_prediction['boxes'].grad is None and model.library_prediction['presence_logits'].grad is None
    assert torch.equal(model.library_font.grad[:,3],torch.zeros_like(model.library_font.grad[:,3]))
    for b,(pi,_) in enumerate(result['assignments']):
        unmatched=sorted(set(range(7))-set(pi.tolist()))
        assert not model.scene_font.grad[b,unmatched].count_nonzero()
    checks['gradients_reach_both_domains_only_accepted_embeddings']=True
    # Query permutations must not change supervision or force the model's count argmax.
    # Remove the duplicate tie for a meaningful permutation-invariance comparison.
    # Equal-box duplicate candidates with different embeddings have an arbitrary
    # Hungarian tie; checking loss invariance for that tie would be incorrect.
    clean=model(**library['inputs']);clean['visibility_logits']=torch.tensor([[8.,8.,-8.,-8.]]).repeat(2,1)
    a,af,ak,_=reference_vectors(clean,library,families,policy)
    order=torch.tensor([3,2,1,0]);permuted={k:(v[:,order] if k!='count_logits' else v) for k,v in clean.items()}
    b,bf,bk,_=reference_vectors(permuted,library,families,policy)
    # Canonicalize through the known library glyph identity rather than query number.
    a_rows=sorted(zip(ak,af.tolist(),a),key=lambda r:(r[1],r[0]))
    b_rows=sorted(zip(bk,bf.tolist(),b),key=lambda r:(r[1],r[0]))
    for x,y in zip(a_rows,b_rows):assert x[:2]==y[:2];torch.testing.assert_close(x[2],y[2])
    assert len(a)==4  # Argmax count is eight; only two visual regions per specimen survive.
    checks['query_permutation_and_no_forced_library_count']=True
    # Analytic denominator excludes an exact rendering pair across distinct labels.
    v=torch.randn(4,128,requires_grad=True);f=torch.tensor([0,1,0,2]);keys=[('p',0),('p',0),None,None]
    loss,audit=contrastive_with_references(v,f,keys,['library','library','scene','scene'],.2)
    z=torch.nn.functional.normalize(v,dim=-1);logits=z@z.T/.2
    expected=(-logits[0,2]+torch.logsumexp(logits[0,torch.tensor([2,3])],0)
              -logits[2,0]+torch.logsumexp(logits[2,torch.tensor([0,1,3])],0))/2
    torch.testing.assert_close(loss,expected)
    assert audit['eligible_anchors']==2 and audit['duplicate_pairs_excluded']==1
    loss.backward();assert torch.isfinite(v.grad).all()
    checks['identical_renderings_excluded_without_family_merging']=True
    ambiguous=torch.randn(4,128,requires_grad=True)
    zero,audit=contrastive_with_references(ambiguous,torch.tensor([0,0,1,1]),
        [('p',0),('p',1),('p',0),('p',1)],['library']*4,.1)
    assert zero==0 and audit['eligible_anchors']==0 and audit['ambiguous_specimen_pairs_excluded']==4
    zero.backward();assert not ambiguous.grad.count_nonzero()
    checks['cross_glyph_alias_contradictions_excluded_within_identical_specimen']=True
    # Same-family exact duplicates cannot supply an easy positive.
    v=torch.randn(3,128,requires_grad=True)
    zero,audit=contrastive_with_references(v,torch.tensor([0,0,1]),[('p',0),('p',0),None],['library']*3,.1)
    assert audit['eligible_anchors']==0 and zero==0
    zero.backward();assert torch.equal(v.grad,torch.zeros_like(v))
    for n in (0,1):
        empty=torch.randn(n,128,requires_grad=True)
        zero,audit=contrastive_with_references(empty,torch.zeros(n,dtype=torch.long),[None]*n,['scene']*n,.1)
        zero.backward();assert audit['eligible_anchors']==0 and torch.isfinite(empty.grad).all()
    checks['duplicates_and_unsupported_anchors_have_finite_zero_gradient']=True
    absent=dict(clean,presence_logits=torch.full((2,4),-8.))
    vectors,ids,keys,report=reference_vectors(absent,library,families,policy)
    assert vectors.shape==(0,128) and all(r['accepted_regions']==0 for r in report)
    checks['missing_library_regions_reported_no_ground_truth_fallback']=True
    extra=copy.deepcopy(library)
    for label in extra['library_labels']:label['specimen_id']='extra/'+label['specimen_id']
    multiple=joint_task_forward(model,scene,[library,extra],families,policy,.1)
    reversed_batches=joint_task_forward(model,scene,[extra,library],families,policy,.1)
    torch.testing.assert_close(multiple['losses']['font_contrastive'],reversed_batches['losses']['font_contrastive'])
    assert multiple['font_audit']['reference_forwards']==2 and multiple['font_audit']['library_vectors']==8
    checks['multiple_reference_subbatches_preserve_evidence_and_order_invariance']=True
    def refuses(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Invalid font training input accepted')
    for split in ('dev','calibration','test'):
        bad=copy.deepcopy(library);bad['library_labels'][0]['split']=split
        calls=len(model.calls);refuses(lambda:joint_task_forward(model,scene,bad,families,policy,.1));assert len(model.calls)==calls
    bad=copy.deepcopy(library);bad['inputs']['font_family']=torch.tensor([0,1])
    refuses(lambda:joint_task_forward(model,scene,bad,families,policy,.1))
    bad=copy.deepcopy(library);bad['library_labels'][1]['specimen_id']=bad['library_labels'][0]['specimen_id']
    refuses(lambda:reference_vectors(clean,bad,families,policy))
    refuses(lambda:reference_vectors(clean,library,dict(alpha=0,beta=0),policy))
    refuses(lambda:reference_vectors(clean,library,dict(alpha=0),policy))
    refuses(lambda:joint_task_forward(model,scene,[library,library],families,policy,.1))
    refuses(lambda:contrastive_with_references(torch.zeros(2,128),torch.tensor([0,1]),[None]*2,['scene']*2,.1))
    refuses(lambda:contrastive_with_references(torch.randn(2,128),torch.tensor([0,1]),[None]*2,['scene']*2,float('nan')))
    checks['heldout_label_input_unknown_mapping_duplicate_specimen_invalid_vectors_refused']=True
    report=dict(created_at=now(),passed=True,device='cpu',checks=checks,source_sha256=sha(Path(__file__).with_name('font_training.py')),
        limitation='CPU analytic gradients and task-adapter checks only. No trained font quality, production reference sampler, phase budget, GPU memory readiness or joint phase launch claimed.')
    write(Path(run)/'font_training_checks.json',report);return report
