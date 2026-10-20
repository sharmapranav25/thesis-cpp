"""Analytic attention tests and actual architecture diagnostics; no optimization."""
import inspect
import time
from pathlib import Path
import torch
from .common import read, write, now, sha, ROOT
from .attention import sample_scales, DeformableAttention
from .model import CapriDecoder, aspect_region_pool
from .data import CorpusDataset, exact_shape_collate
from .objectives import instance_losses


def attention_checks(device):
    value = torch.tensor([[[[[1., 2.], [3., 4.]]]]], device=device, requires_grad=True)
    locations = torch.tensor([[[[[[.5, .5]]]]]], device=device, requires_grad=True)
    weights = torch.ones(1, 1, 1, 1, 1, device=device, requires_grad=True)
    result = sample_scales([value], locations, weights)
    torch.testing.assert_close(result, torch.tensor([[[2.5]]], device=device))
    result.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, .25))
    torch.testing.assert_close(locations.grad.reshape(2), torch.tensor([2., 4.], device=device))
    assert sample_scales([value.detach()], torch.full_like(locations, 2.), weights.detach()).abs().max() == 0
    mixed = sample_scales([value.detach(), value.detach()*2], locations.detach().expand(1, 1, 1, 2, 1, 2),
                          weights.detach().expand(1, 1, 1, 2, 1)*.5)
    torch.testing.assert_close(mixed, result.detach()*1.5)
    attn = DeformableAttention(width=8, heads=2, levels=1, points=2).to(device)
    feature = torch.randn(1, 8, 4, 4, device=device, requires_grad=True)
    mask = torch.ones(1, 4, 4, device=device); mask[:, :, 2:] = 0
    query = torch.randn(1, 3, 8, device=device, requires_grad=True)
    reference = torch.full((1, 3, 2), .5, device=device)
    first = attn(query, [feature], [mask], reference)
    altered = feature.detach().clone(); altered[:, :, :, 2:] = 1000
    torch.testing.assert_close(first, attn(query, [altered], [mask], reference))
    first.square().sum().backward()
    assert feature.grad[:, :, :, 2:].abs().max() == 0 and torch.isfinite(query.grad).all()
    # Known aspect ratio: 6.4-wide by 12.8-high glyph occupies half a square crop.
    features = torch.ones(1, 2, 16, 32, device=device, requires_grad=True)
    boxes = torch.tensor([[[.4, .1, .6, .9]]], device=device, requires_grad=True)
    masks = torch.full((1, 1, 16, 32), 20., device=device, requires_grad=True)
    crop = aspect_region_pool(features, boxes, masks, torch.ones(1, 16, 32, device=device))
    assert (crop[0, 0].sum(0) > 0).sum() == 8
    crop.sum().backward()
    assert boxes.grad is None and masks.grad is None and features.grad.abs().sum() > 0


def _check_model(run):
    torch.manual_seed(731); torch.set_num_threads(4)
    config = read(ROOT/'configs/local.json'); source = config['decoder_dinov3_source']
    checks = {}
    attention_checks('cpu'); checks['analytic_cpu_attention_and_aspect_pooling'] = True
    if not torch.cuda.is_available(): raise RuntimeError('CUDA architecture check required')
    attention_checks('cuda'); checks['analytic_cuda_attention_and_aspect_pooling'] = True
    try: CapriDecoder(source)
    except ValueError: pass
    else: raise AssertionError('Missing pretrained weights were silently accepted')
    checks['missing_pretrained_weights_refused'] = True
    model = CapriDecoder(source, diagnostic_random=True).cuda()
    counts = model.parameter_counts()
    assert counts['dino'] == 21601152 and counts['cnn'] > 10000000
    assert list(inspect.signature(model.forward).parameters) == ['pixels', 'valid_pixels']
    data = CorpusDataset(run, allow_partial=True, max_shards=1)
    # Find a declared same-content/different-font TRAIN pair with valid labels
    # for every new auxiliary, so actual head gradients are exercised.
    batch = None
    for first in range(4,min(len(data)-1,1000),20):
        candidate = [data[first],data[first+1]]
        if (len({tuple(x['inputs']['pixels'].shape) for x in candidate})==1
            and all(any(x['targets']['font_attributes']['valid'][k].any() for x in candidate)
                    for k in candidate[0]['targets']['font_attributes']['valid'])
            and len(torch.cat([x['targets']['font_family'] for x in candidate]).unique())>1):
            batch = exact_shape_collate(candidate);break
    if batch is None:raise AssertionError('No TRAIN diagnostic pair covers all font attributes')
    inputs = {k:v.cuda() for k,v in batch['inputs'].items()}
    data.close()
    model.train()
    assert not model.dino.training and not any(p.requires_grad for p in model.dino.parameters())
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.float16):
        pred = model(**inputs)
        losses, assignment = instance_losses(pred, batch['targets'])
        for i, aux in enumerate(pred['auxiliary']):
            aux_loss, _ = instance_losses(aux, batch['targets'])
            losses.update({f'layer{i}/{k}':v for k,v in aux_loss.items()})
        # Integration diagnostic includes gradients for exported auxiliary heads;
        # these coefficients are not a frozen training objective or schedule.
        loss = sum(losses.values())+pred['font_embeddings'][..., 0].mean()+pred['colour_rgb'].mean()+pred['background_under'].mean()
    loss.backward(); torch.cuda.synchronize()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.cnn.stages[0][0].main[0].weight.grad.abs().sum() > 0
    assert model.dino_projection.weight.grad.abs().sum() > 0
    assert model.font_project[0].weight.grad.abs().sum() > 0
    assert model.font_style[0].weight.grad.abs().sum() > 0
    assert all(head.weight.grad is not None and head.weight.grad.abs().sum()>0 for head in model.font_attributes.values())
    assert not any(p.grad is not None for p in model.dino.parameters())
    assert pred['boxes'].shape == (2, 16, 4) and pred['count_logits'].shape == (2, 7)
    assert pred['effect_presence_logits'].shape == pred['intensity_predictions'].shape == (2, 37)
    torch.testing.assert_close(pred['font_embeddings'].norm(dim=-1), torch.ones(2, 16, device='cuda'), atol=1e-5, rtol=1e-5)
    measured = dict(input_shape=list(inputs['pixels'].shape), backward_seconds=time.perf_counter()-start,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    losses={k:float(v.detach()) for k,v in losses.items()})
    del pred, losses, loss, assignment
    model.zero_grad(set_to_none=True); model.eval()
    with torch.no_grad():
        a = model(**{k:v[:1] for k,v in inputs.items()})
        b = model(**inputs)
        for key in ('presence_logits', 'boxes', 'mask_logits', 'count_logits', 'font_embeddings'):
            torch.testing.assert_close(a[key], b[key][:1], atol=2e-4, rtol=2e-4)
        altered = {k:v[:1].clone() for k,v in inputs.items()}
        # The selected corpus pair may have no padding. Explicitly test a valid
        # rectangle ending three pixels before the deterministic padded edge.
        altered['valid_pixels'][:, :, -3:] = False
        padded = model(**altered)
        altered['pixels'].masked_fill_(~altered['valid_pixels'][:, None], .123)
        changed = model(**altered)
        for key in ('presence_logits', 'boxes', 'mask_logits', 'count_logits', 'font_embeddings'):
            torch.testing.assert_close(padded[key], changed[key])
        extra = {k:v.clone() for k,v in altered.items()}
        extra['valid_pixels'][:, :, -32:] = False
        try: model(**extra)
        except ValueError: pass
        else: raise AssertionError('Batch-dependent padding accepted')
    del a, b, changed, padded
    checks['batch_composition_and_padding_content_invariant'] = True
    checks['actual_synthetic_matching_auxiliary_losses_amp_gradients'] = True
    checks['pixel_only_forward_and_frozen_dino'] = True
    checks['all_six_font_attribute_heads_actual_label_gradients'] = True
    model.set_dino_trainable(4); model.train()
    assert all(not p.requires_grad for block in model.dino.blocks[:-4] for p in block.parameters())
    assert all(p.requires_grad for block in model.dino.blocks[-4:] for p in block.parameters())
    # Worst allowed canvas, with controlled partial DINO fine-tuning and real backward.
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.float16):
        worst = model(torch.rand(2, 3, 128, 768, device='cuda'), torch.ones(2, 128, 768, dtype=torch.bool, device='cuda'))
        objective = worst['count_logits'].float().square().mean()+worst['mask_logits'].float().square().mean()+worst['font_embeddings'][..., 0].mean()
    objective.backward(); torch.cuda.synchronize()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.dino.blocks[-1].parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    worst_case = dict(input_shape=[2,3,128,768], last_dino_blocks_trainable=4,
                      backward_seconds=time.perf_counter()-start, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    checks['maximum_canvas_partial_dino_amp_backward'] = True
    result = dict(created_at=now(), passed=True, diagnostic_random_dino=True, optimization_steps=0,
                  parameters=counts, partial_dino_parameters=model.parameter_counts(), checks=checks,
                  measured=measured, worst_case=worst_case,
                  source_hashes={name:sha(Path(__file__).with_name(name)) for name in ('model.py','attention.py','model_checks.py','appearance.py','objectives.py','data.py')},
                  remaining='Approved pretrained weights; remaining palette/geometry and font retrieval contracts; final-model optimizer/EMA resume; measured training objective/budget; baseline, specialists, distillation, calibration and final evaluation.')
    write(run/'model_checks.json', result)
    return result


def check_model(run):
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        # Isolate geometry from batch-dependent TF32 rounding. AMP gradients
        # and the maximum-canvas memory test are exercised independently.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return _check_model(run)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
