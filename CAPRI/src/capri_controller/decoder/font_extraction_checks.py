"""Analytic CPU extraction fixtures, not a trained font index."""
import copy
import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import now,write,sha
from .font_extraction import visual_regions,associate_library_regions,extract_train_bank
from .retrieval import FontIndex,IDENTITY_FIELDS
from .checkpoints import rng_state,execution_state
from .phase_engine_checks import equal


BOXES=torch.tensor([[.1,.2,.3,.8],[.6,.2,.8,.8],[.1,.2,.3,.8],[.4,.1,.5,.2]])


class ToyLibrary:
    rows=[dict(shape=(16,32)) for _ in range(3)]
    snapshot_hash='cpu-diagnostic-three-library-specimens'
    def __len__(self):return 3
    def __getitem__(self,i):
        digest=str(i+1)*64
        return dict(inputs=dict(pixels=torch.ones(3,16,32) if i<2 else torch.zeros(3,16,32),
                                valid_pixels=torch.ones(16,32,dtype=torch.bool)),
                    library_labels=dict(specimen_id='toy/'+str(i),font_key=digest[:20],font_sha256=digest,
                                        family_group=('alpha','beta','gamma')[i],style='regular',split='train',axes={},
                                        boxes_visible=BOXES[:2].clone()),
                    audit=dict(stem='toy_'+str(i),pixel_sha256=('a' if i<2 else 'b')*64))


class ToyExtractor(torch.nn.Module):
    def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(0.));self.calls=0
    def forward(self,pixels,valid_pixels):
        assert not torch.is_grad_enabled() and not self.training
        self.calls+=1;random.random();np.random.rand();torch.rand(1)
        n=len(pixels);features=torch.zeros(n,4,128);features[:,:,0]=1.
        presence=torch.full((n,4),8.);presence[pixels.flatten(1).sum(-1)==0]=-8.
        return dict(presence_logits=presence,visibility_logits=torch.tensor([[8.,8.,8.,-8.]]).repeat(n,1),
                    boxes=BOXES[None].repeat(n,1,1),font_embeddings=features,
                    count_logits=torch.tensor([[0.,0.,0.,0.,0.,0.,99.]]).repeat(n,1))


def check_font_extraction(run):
    if torch.cuda.is_available():raise RuntimeError('Hide CUDA for extraction diagnostics')
    torch.set_num_threads(2)
    contract={k:'a'*64 for k in IDENTITY_FIELDS[:4]};contract.update(weights='ema',diagnostic=True)
    policy=dict(instance_threshold=.5,visibility_threshold=.5,library_iou_threshold=.5)
    dataset=ToyLibrary();model=ToyExtractor();before_rng=rng_state();before_modes=execution_state(model)
    bank,audit=extract_train_bank(model,dataset,contract,policy,batch_size=2)
    assert model.calls==2 and equal(before_rng,rng_state()) and equal(before_modes,execution_state(model))
    assert audit['prototypes']==4 and audit['selected_regions']==6 and audit['unmatched_selected_regions']==2
    assert audit['zero_region_specimens']==1 and audit['missed_library_glyphs']==2
    assert audit['missing_families']==['gamma'] and audit['missing_fonts']==['3'*20]
    assert audit['identical_pixel_family_ambiguities']=={'a'*64:['alpha','beta']}
    assert all(r['library_iou']==1. and r['query_id'] in (0,1,2) for r in bank.records)
    assert not bank.embeddings.requires_grad
    checks=dict(pixel_only_forward_and_predicted_regions=True,no_forced_count_or_ground_truth_crops=True,
                one_to_one_duplicate_rejection=True,missing_regions_fonts_and_families_reported=True,
                identical_pixels_different_families_preserved=True,rng_and_modes_restored=True)
    query=bank.embeddings[:1];result=bank.search(query,contract,topk=2)
    assert len(result['aggregate'])==2 and all(r['best_match_weight_fraction']==.5 for r in result['aggregate'])
    destination=Path(run)/('font_extraction_smoke_'+str(time.time_ns()))
    bank.save(destination/'index');write(destination/'extraction_audit.json',audit)
    loaded=FontIndex.load(destination/'index',contract,allow_diagnostic=True)
    assert loaded.search(query,contract,topk=2)==result
    checks['diagnostic_index_roundtrip_and_ambiguity_ties']=True
    model.eval()
    with torch.inference_mode():pred=model(dataset[0]['inputs']['pixels'][None],dataset[0]['inputs']['valid_pixels'][None])
    regions=visual_regions(pred,.5,.5);row=dataset[0]
    values,records,reports=associate_library_regions(regions,[row['library_labels']],[row['audit']],policy)
    altered=copy.deepcopy(row['library_labels']);altered['boxes_visible']=torch.tensor([[0.,0.,.01,.01]])
    empty,_,rejected=associate_library_regions(regions,[altered],[row['audit']],policy)
    assert not len(empty) and rejected[0]['accepted_regions']==0
    assert torch.equal(regions[0]['embeddings'],visual_regions(pred,.5,.5)[0]['embeddings'])
    assert len(regions[0]['query_ids'])==3  # Count argmax says eight; selector still uses visual evidence.
    checks['library_labels_only_audit_predictions_after_forward']=True
    def refuses(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Invalid extraction accepted')
    refuses(lambda:associate_library_regions(regions,[dict(row['library_labels'],split='dev')],[row['audit']],policy))
    refuses(lambda:extract_train_bank(model,dataset,contract,dict(policy,instance_threshold=float('nan'))))
    for key in ('presence_logits','boxes','font_embeddings'):
        bad=copy.deepcopy(pred);bad[key].flatten()[0]=float('nan')
        refuses(lambda:visual_regions(bad,.5,.5))
    bad=copy.deepcopy(pred);bad['font_embeddings'].zero_();refuses(lambda:visual_regions(bad,.5,.5))
    checks['heldout_invalid_threshold_nonfinite_and_zero_vectors_refused']=True
    class NoEvidence(ToyExtractor):
        def forward(self,pixels,valid_pixels):
            p=super().forward(pixels,valid_pixels);p['presence_logits'].fill_(-99.);return p
    empty,audit_empty=extract_train_bank(NoEvidence(),dataset,contract,policy)
    assert empty is None and audit_empty['prototypes']==0 and len(audit_empty['missing_families'])==3
    checks['empty_bank_returns_audit_without_fabrication']=True
    class Failure(ToyExtractor):
        def forward(self,pixels,valid_pixels):
            super().forward(pixels,valid_pixels);raise RuntimeError('Injected extraction failure')
    failing=Failure();before=rng_state();modes=execution_state(failing)
    try:extract_train_bank(failing,dataset,contract,policy)
    except RuntimeError as exc:assert 'Injected' in str(exc)
    else:raise AssertionError('Failure injection did not stop')
    assert equal(before,rng_state()) and equal(modes,execution_state(failing))
    checks['failure_restores_modes_and_rng']=True
    report=dict(created_at=now(),passed=True,device='cpu',checks=checks,diagnostic_root=str(destination),
                source_sha256=sha(Path(__file__).with_name('font_extraction.py')),
                limitation='Fabricated predictions only. Real pretrained/font-trained extraction and coverage acceptance remain unrun; no production font index or font accuracy claim.')
    write(Path(run)/'font_extraction_checks.json',report);return report
