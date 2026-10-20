"""Analytic missingness/assignment/colour checks plus TRAIN-only label audit."""
import copy
from pathlib import Path
import torch
from .appearance import ATTRIBUTE_DIMS, font_targets, appearance_losses
from .data import CorpusDataset
from .common import write, now, sha


def check_appearance(run):
    checks = {}
    font = dict(axes=[dict(tag='wght', default=400.)], weight_class=400)
    glyph = dict(axes={'wght':400.}, italic=False, category='Serif', stroke_class='Serif',
                 x_height=.7, intrinsic_width=.8, char='x')
    meta = font_targets([glyph], [font])
    assert all(v.item() for v in meta['valid'].values())
    assert abs(meta['values']['weight'].item()-.4)<1e-6
    varied = dict(glyph, axes={'wght':700.}, char='A')
    vm = font_targets([varied], [font])
    assert not vm['valid']['x_height'].item() and not vm['valid']['intrinsic_width'].item()
    assert abs(vm['values']['weight'].item()-.7)<1e-6
    absent = dict(glyph, category='unknown', stroke_class='unknown', x_height=None, intrinsic_width=float('nan'), italic=None)
    missing = font_targets([absent], [font])
    assert not any(missing['valid'][k].item() for k in ('category','stroke','x_height','intrinsic_width','italic'))
    italic = font_targets([dict(glyph, axes={'ital':.5})], [dict(font, axes=[dict(tag='ital',default=0)])])
    assert not italic['valid']['italic'].item()
    checks['unknown_metadata_nondefault_axes_lowercase_evidence_masks']=True
    results = {}
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        pred = dict(font_attribute_logits={k:torch.zeros(1,4,n,device=device,requires_grad=True) for k,n in ATTRIBUTE_DIMS.items()},
                    colour_rgb=torch.full((1,4,3),.4,device=device,requires_grad=True),
                    background_under=torch.full((1,3,2,2),.8,device=device,requires_grad=True))
        v = torch.tensor([[1.,.5],[0.,0.]])
        target = dict(font_attributes=meta, colour_rgb=torch.tensor([[.8,.8,.8]]),
                      background_under=torch.full((3,2,2),.8)*v, valid_mask=v)
        assign = [(torch.tensor([2],device=device),torch.tensor([0],device=device))]
        with torch.autocast(device_type=device, enabled=device=='cuda', dtype=torch.float16):
            loss, support = appearance_losses(pred,[target],assign)
        assert loss['background_under']==0 and all(n==1 for n in support.values())
        sum(loss.values()).backward()
        for k,p in pred['font_attribute_logits'].items():
            assert torch.isfinite(p.grad).all() and p.grad[:,[0,1,3]].abs().sum()==0
            assert p.grad[:,2].abs().sum()>0
        assert pred['colour_rgb'].grad[:,[0,1,3]].abs().sum()==0 and pred['colour_rgb'].grad[:,2].abs().sum()>0
        # Padding carries neither value nor gradient, including deliberately
        # corrupted padding predictions; partial valid cells retain RGB .8.
        bg=pred['background_under'].detach().clone();bg[:,:,1,:]=19.;bg.requires_grad_()
        bg_loss,_=appearance_losses({'background_under':bg},[target],assign)
        assert bg_loss['background_under']==0
        bg_loss['background_under'].backward();assert bg.grad[:,:,1,:].abs().sum()==0
        unknown=copy.deepcopy(target)
        unknown['font_attributes']['valid']={k:torch.zeros_like(x) for k,x in meta['valid'].items()}
        unknown['font_attributes']['values']={k:torch.full_like(x,0 if x.dtype==torch.long else float('nan')) for k,x in meta['values'].items()}
        unknown['colour_rgb']=torch.full((1,3),float('nan'))
        p2={k:x.detach().clone().requires_grad_() for k,x in pred['font_attribute_logits'].items()}
        c2=pred['colour_rgb'].detach().clone().requires_grad_()
        zero, counts=appearance_losses(dict(font_attribute_logits=p2,colour_rgb=c2),[unknown],assign)
        assert all(x==0 for x in zero.values()) and all(x==0 for x in counts.values())
        sum(zero.values()).backward();assert all(x.grad.abs().sum()==0 for x in p2.values()) and c2.grad.abs().sum()==0
        permutation=torch.tensor([2,0,3,1],device=device)
        shuffled=dict(font_attribute_logits={k:x[:,permutation] for k,x in pred['font_attribute_logits'].items()},
                      colour_rgb=pred['colour_rgb'][:,permutation],background_under=pred['background_under'])
        swapped,_=appearance_losses(shuffled,[target],[(torch.tensor([0],device=device),assign[0][1])])
        assert all(torch.equal(loss[k],swapped[k]) for k in loss)
        results[device]=dict(losses={k:float(x) for k,x in loss.items()},support=support)
    checks['matched_only_gradients_query_permutation_cpu_amp']=True
    checks['unknown_nan_targets_zero_gradient_and_zero_support']=True
    checks['background_partial_coverage_and_padding_invariance']=True
    ds=CorpusDataset(run,allow_partial=True,max_shards=1)
    support={k:0 for k in ATTRIBUTE_DIMS}; glyphs=0; images=0
    try:
        for i in range(0,len(ds),19):
            sample=ds[i];t=sample['targets'];images+=1;glyphs+=int(t['count'])
            assert set(sample['inputs'])=={'pixels','valid_pixels'}
            for k in support:support[k]+=int(t['font_attributes']['valid'][k].sum())
            bg=t['background_under'];v=t['valid_mask'];normalized=bg/v.clamp_min(1e-8)[None]
            assert torch.isfinite(normalized).all() and normalized[:,v>0].min()>=0 and normalized[:,v>0].max()<=1.000001
    finally: ds.close()
    result=dict(created_at=now(),passed=True,checks=checks,analytic=results,
                train_sample=dict(images=images,glyphs=glyphs,valid_attribute_counts=support),
                source_hashes={n:sha(Path(__file__).with_name(n)) for n in ('appearance.py','appearance_checks.py','data.py','model.py','objectives.py')},
                limitation='No training or held-out scoring. Metadata may be heuristic; missing/nondefault measurements are masked. Weight is nominal library weight, not observed stroke thickness; background is the saved pre-photometric under-layer, not a final background segmentation. No calibrated retrieval/palette/geometry contract yet.')
    write(run/'appearance_checks.json',result)
    return result
