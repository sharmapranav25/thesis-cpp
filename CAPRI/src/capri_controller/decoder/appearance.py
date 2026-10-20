"""Masked font metadata and observed colour supervision, never model inputs.

Metadata describes library style, not a uniquely identifiable renderer knob.
Default-instance measurements are withheld for nondefault variable instances.
"""
import math
import torch
from torch.nn import functional as F

CATEGORIES = ('Sans Serif', 'Serif', 'Display', 'Handwriting', 'Monospace')
STROKES = ('Sans Serif', 'Serif', 'Slab Serif')
ATTRIBUTE_DIMS = dict(category=len(CATEGORIES), stroke=len(STROKES), italic=1,
                      weight=1, x_height=1, intrinsic_width=1)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def font_targets(glyphs, fonts):
    """Known synthetic metadata, with a separate validity bit for every field."""
    if len(glyphs) != len(fonts): raise ValueError('Glyph/font metadata mismatch')
    values = {k: [] for k in ATTRIBUTE_DIMS}; valid = {k: [] for k in ATTRIBUTE_DIMS}
    for glyph, font in zip(glyphs, fonts):
        axes = glyph.get('axes', {})
        defaults = {a['tag']: a['default'] for a in font['axes']}
        default_instance = set(axes)==set(defaults) and all(number(v) and abs(v-defaults[k]) <= 1e-6 for k,v in axes.items())
        row = {}; known = {}
        for key, vocabulary, source in (('category', CATEGORIES, 'category'), ('stroke', STROKES, 'stroke_class')):
            value = glyph.get(source); known[key] = value in vocabulary
            row[key] = vocabulary.index(value) if known[key] else 0
        italic = glyph.get('italic')
        if 'ital' in axes:
            italic = bool(axes['ital']) if axes['ital'] in (0., 1.) else None
        elif 'slnt' in axes and axes['slnt'] != defaults.get('slnt'):
            italic = None
        known['italic'] = isinstance(italic, bool); row['italic'] = float(italic) if known['italic'] else 0.
        weight = axes.get('wght', font.get('weight_class'))
        known['weight'] = number(weight) and 1 <= weight <= 1000
        row['weight'] = weight / 1000 if known['weight'] else 0.
        for key, source in (('x_height', 'x_height'), ('intrinsic_width', 'intrinsic_width')):
            value = glyph.get(source)
            # x-height has weak direct evidence in uppercase/digit-only glyphs.
            known[key] = default_instance and number(value) and value > 0 and (key != 'x_height' or glyph.get('char', '').islower())
            row[key] = float(value) if known[key] else 0.
        for key in values:
            values[key].append(row[key]); valid[key].append(known[key])
    return dict(values={k: torch.tensor(v, dtype=torch.long if k in ('category','stroke') else torch.float32) for k,v in values.items()},
                valid={k: torch.tensor(v, dtype=torch.bool) for k,v in valid.items()})


def appearance_losses(pred, targets, assignments):
    """Separate task means and evidence counts for later task-affinity adapters.

Missing labels contribute differentiable zero, never an invented negative.
Background targets are area-downsampled with zero padding; divide out valid
coverage at boundary cells before comparison, then weight by that coverage.
"""
    losses = {}; support = {}
    if len(targets) != len(assignments): raise ValueError('Assignment batch mismatch')
    attributes = pred.get('font_attribute_logits')
    if attributes is not None:
        if set(attributes) != set(ATTRIBUTE_DIMS): raise ValueError('Font attribute output contract mismatch')
        for key, prediction in attributes.items():
            chosen = []; truth = []
            for i, (t, (pi, ti)) in enumerate(zip(targets, assignments)):
                meta = t['font_attributes']; mask = meta['valid'][key].to(prediction.device)[ti]
                chosen.append(prediction[i,pi][mask].float()); truth.append(meta['values'][key].to(prediction.device)[ti][mask])
            p = torch.cat(chosen); y = torch.cat(truth); support['font_'+key] = len(y)
            if not len(y): value = prediction.float().sum()*0
            elif key in ('category', 'stroke'): value = F.cross_entropy(p, y)
            elif key == 'italic': value = F.binary_cross_entropy_with_logits(p[:,0], y.float())
            else:
                estimate = p[:,0].sigmoid() if key == 'weight' else F.softplus(p[:,0])
                value = F.smooth_l1_loss(estimate, y.float())
            losses['font_'+key] = value
    if 'colour_rgb' in pred:
        p = pred['colour_rgb']; terms = []; count = 0
        for i, (t, (pi,ti)) in enumerate(zip(targets, assignments)):
            y = t['colour_rgb'].to(p.device)[ti].float(); valid = torch.isfinite(y).all(-1)
            if ((y[valid]<0)|(y[valid]>1)).any(): raise ValueError('Invalid observed RGB target')
            if valid.any(): terms.append(F.smooth_l1_loss(p[i,pi][valid].float(), y[valid], reduction='sum')/3)
            count += int(valid.sum())
        losses['colour_rgb'] = sum(terms, p.float().sum()*0)/max(1,count); support['colour_rgb'] = count
    if 'background_under' in pred:
        p = pred['background_under'].float(); terms = []; count = 0
        for i,t in enumerate(targets):
            y = t['background_under'].to(p.device).float(); v = t['valid_mask'].to(p.device).float()
            if y.shape != p[i].shape or v.shape != y.shape[-2:]: raise ValueError('Background shape mismatch')
            if not torch.isfinite(y).all() or not torch.isfinite(v).all() or (v<0).any() or (v>1).any() or v.sum()<=0:
                raise ValueError('Invalid background target/coverage')
            y = y/v.clamp_min(1e-8)[None]
            if ((y[:,v>0]<-1e-6)|(y[:,v>0]>1+1e-6)).any(): raise ValueError('Invalid unpremultiplied background')
            terms.append((F.smooth_l1_loss(p[i],y,reduction='none')*v[None]).sum()/(3*v.sum()))
            count += 1
        losses['background_under'] = torch.stack(terms).mean(); support['background_under'] = count
    if not all(torch.isfinite(v) for v in losses.values()): raise ValueError('Nonfinite appearance loss')
    return losses, support
