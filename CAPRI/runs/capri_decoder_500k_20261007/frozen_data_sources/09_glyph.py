"""Glyph effects: applied to the text's own pixels after the text is drawn (by any renderer or CAPTCHA library).

The text pixels come from the label maps: records with head "text" (standalone draw_text) or any ids the caller passes as
`text_ids` (e.g. intercepted glyph ids of a library stage). Effects:
  shadow / extrude / outline   add pixels next to the glyphs (new labelled ops under/around them)
  hollow / thin / cut-free     remove glyph interior or edges (painted with the local background; ids cleared)
  bold                         grow the glyphs (new pixels keep the nearest glyph's id, so glyph boxes stay right)
  fill_texture / translucent   change the glyph colours in place
Each records the measured visible amount (fraction of the canvas or of the glyph pixels it changed).
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from captcha_effects import styles as S
from captcha_effects.effects import effect, _pal


def text_ids(cv, ids=None):
    if ids is not None:
        return np.asarray(sorted(ids), np.uint32)
    return np.asarray([r["id"] for r in cv.records if r["head"] == "text"], np.uint32)


def glyph_mask(cv, ids=None):
    return np.isin(cv.opaque, text_ids(cv, ids))


def _morph(mask, px, grow=True):
    k = 2 * int(px) + 1
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MaxFilter(k) if grow else ImageFilter.MinFilter(k))
    return np.asarray(im) > 127


def _background_under(cv, mask):
    """Estimate of the background behind `mask` pixels: normalised blur of the non-mask image."""
    keep = (~mask).astype(np.float32)
    num = np.stack([_gauss(cv.img[..., c] * keep) for c in range(3)], -1)
    den = _gauss(keep)[..., None]
    return num / np.maximum(den, 1e-3)


def _gauss(a, r=4):
    """Separable Gaussian blur (sigma r) with edge padding, float32."""
    x = np.arange(-3 * r, 3 * r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / r) ** 2)
    k /= k.sum()
    pad = len(k) // 2
    a = np.pad(a.astype(np.float32), pad, mode="edge")
    a = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, a)


def _glyph_colour(cv, m):
    return tuple(int(v) for v in np.median(cv.img[m], 0)) if m.any() else (0, 0, 0)


def _frac_grow(mask, wf, rng):
    """Grow by a fractional width: full rings up to floor(wf), the next ring kept with probability frac(wf)."""
    w0 = int(math.floor(wf))
    inner = _morph(mask, w0, True) if w0 > 0 else mask.copy()
    if wf - w0 > 0:
        nxt = _morph(mask, w0 + 1, True) & ~inner
        inner |= nxt & (rng.random(mask.shape) < (wf - w0))
    return inner


def _shifted(mask, dx, dy):
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, xs = slice(max(0, dy), h + min(0, dy)), slice(max(0, dx), w + min(0, dx))
    yd, xd = slice(max(0, -dy), h + min(0, -dy)), slice(max(0, -dx), w + min(0, -dx))
    out[ys, xs] = mask[yd, xd]
    return out


def _paint(cv, m, colour, effect_name, head, opacity=1.0, params=None):
    return cv.blend(m.astype(np.float32), colour, effect_name, head, "glyph", opacity, params)


@effect("shadow", "glyph_shadow", where="glyph")
def shadow(cv, level, rng, ids=None, **kw):
    if level <= 0:
        return {}
    g = glyph_mask(cv, ids)
    d = 1 + int(round(4 * level / 10))
    ang = rng.uniform(0, 2 * math.pi)
    dx, dy = int(round(d * math.cos(ang))), int(round(d * math.sin(ang))) or 1
    sh = _shifted(g, dx, dy) & ~g
    pal = _pal(rng, cv, kw, ("dark", "grey", "shades", "random"))
    col = S.colour(rng, pal, cv.background, cv.text_colour)
    _paint(cv, sh, col, "shadow", "glyph_shadow", float(rng.choice([1.0, 0.7])), dict(dx=dx, dy=dy))
    return dict(dx=dx, dy=dy, palette=pal, measured=float(sh.mean()))


@effect("extrude", "extrude_3d", where="glyph")
def extrude(cv, level, rng, ids=None, **kw):
    if level <= 0:
        return {}
    g = glyph_mask(cv, ids)
    steps = 1 + int(round(5 * level / 10))
    ang = rng.uniform(0, 2 * math.pi)
    acc = np.zeros_like(g)
    for k in range(1, steps + 1):
        acc |= _shifted(g, int(round(k * math.cos(ang))), int(round(k * math.sin(ang))))
    acc &= ~g
    pal = _pal(rng, cv, kw, ("dark", "shades", "grey", "random"))
    col = S.colour(rng, pal, cv.background, cv.text_colour)
    _paint(cv, acc, col, "extrude", "extrude_3d", 1.0, dict(steps=steps))
    return dict(steps=steps, palette=pal, measured=float(acc.mean()))


@effect("outline", "glyph_outline", where="glyph")
def outline(cv, level, rng, ids=None, **kw):
    if level <= 0:
        return {}
    g = glyph_mask(cv, ids)
    w = 0.3 + 2.7 * level / 10
    ring = _frac_grow(g, w, rng) & ~g
    pal = _pal(rng, cv, kw, ("contrast", "dark", "random", "saturated"))
    col = S.colour(rng, pal, cv.background, cv.text_colour)
    _paint(cv, ring, col, "outline", "glyph_outline", 1.0, dict(width=w))
    return dict(width=w, palette=pal, measured=float(ring.mean()))


@effect("hollow", "hollow_outline_glyphs", where="glyph")
def hollow(cv, level, rng, ids=None, **kw):
    """Keep only a rim of each glyph; the interior shows the background."""
    if level <= 0:
        return {}
    g = glyph_mask(cv, ids)
    inner = _morph(g, 1, False) & g                  # 1-px rim kept
    clear = 0.1 + 0.9 * level / 10                   # how strongly the interior is replaced by the background
    bg = _background_under(cv, g)
    cv.img[inner] = cv.img[inner] * (1 - clear) + bg[inner] * clear
    if clear >= 0.5:
        cv.opaque[inner] = 0
    frac = clear  # how strongly the interior is cleared (the interior's share depends on stroke width, not on the level)
    cv.record(effect="hollow", head="hollow_outline_glyphs", layer="glyph", params=dict(clear=clear), measured=frac)
    return dict(clear=clear, measured=frac)


@effect("bold", "stroke_bold", where="glyph")
def bold(cv, level, rng, ids=None, **kw):
    if level <= 0:
        return {}
    g = glyph_mask(cv, ids)
    w = 0.3 + 2.7 * level / 10
    grown = _frac_grow(g, w, rng) & ~g
    col = _glyph_colour(cv, g)
    k = 2 * int(math.ceil(w)) + 1
    ids_grown = np.asarray(Image.fromarray(np.where(g, cv.opaque, 0).astype(np.int32)).filter(ImageFilter.MaxFilter(k)))
    cv.img[grown] = np.array(col, np.float32)
    cv.opaque[grown] = ids_grown[grown].astype(np.uint32)
    cv.record(effect="bold", head="stroke_bold", layer="glyph", params=dict(width=w), measured=float(grown.sum() / max(1, g.sum())))
    return dict(width=w, measured=float(grown.sum() / max(1, g.sum())))


@effect("thin", "stroke_thin", where="glyph")
def thin(cv, level, rng, ids=None, **kw):
    if level <= 0:
        return {}
    g = glyph_mask(cv, ids)
    w = 1 if level < 7 else 2
    kept = _morph(g, w, False) & g
    removed = g & ~kept
    if level < 7:  # partial erosion at low levels: remove only a random share of the edge
        removed &= rng.random(g.shape) < (0.3 + 0.7 * level / 7)
    bg = _background_under(cv, g)
    cv.img[removed] = bg[removed]
    cv.opaque[removed] = 0
    frac = float(removed.sum() / max(1, g.sum()))
    cv.record(effect="thin", head="stroke_thin", layer="glyph", params=dict(width=w), measured=frac)
    return dict(width=w, measured=frac)


@effect("fill_texture", "glyph_fill_texture", where="glyph")
def fill_texture(cv, level, rng, ids=None, **kw):
    """Stripes, gradient or noise inside the glyphs."""
    if level <= 0:
        return {}
    s = level / 10
    g = glyph_mask(cv, ids)
    kind = kw.get("kind") or str(rng.choice(["stripes", "gradient", "noise", "checker"]))
    base = np.array(_glyph_colour(cv, g), np.float32)
    other = np.array(S.with_lum_distance(S.colour(rng, _pal(rng, cv, kw, ("random", "saturated", "pastel", "contrast")), cv.background),
                                         tuple(base), 90 + 130 * s), np.float32)
    yy, xx = np.mgrid[0:cv.h, 0:cv.w].astype(np.float32)
    p = max(3.0, (0.3 - 0.18 * s) * cv.h)
    if kind == "stripes":
        t = ((xx + yy * rng.uniform(-1, 1)) / p % 1.0 < 0.5).astype(np.float32)
    elif kind == "checker":
        t = ((np.floor(xx / p) + np.floor(yy / p)) % 2).astype(np.float32)
    elif kind == "gradient":
        t = yy / cv.h if rng.random() < 0.5 else xx / cv.w
    else:
        t = rng.random((cv.h, cv.w)).astype(np.float32)
    mix = (0.45 + 0.55 * s) * t[..., None]
    new = base * (1 - mix) + other * mix
    cv.img[g] = new[g]
    cv.record(effect="fill_texture", head="glyph_fill_texture", layer="glyph", params=dict(kind=kind),
              measured=float(0.45 + 0.55 * s) * float(np.linalg.norm(other - base)) / 441.7)
    return dict(kind=kind, measured=float(0.45 + 0.55 * s) * float(np.linalg.norm(other - base)) / 441.7)


@effect("translucent", "translucent_text", where="glyph")
def translucent(cv, level, rng, ids=None, **kw):
    if level <= 0:
        return {}
    alpha = 0.95 - 0.7 * level / 10          # remaining glyph opacity
    g = glyph_mask(cv, ids)
    bg = _background_under(cv, g)
    cv.img[g] = cv.img[g] * alpha + bg[g] * (1 - alpha)
    cv.record(effect="translucent", head="translucent_text", layer="glyph", params=dict(alpha=alpha), measured=1 - alpha)
    return dict(alpha=alpha, measured=1 - alpha)


GLYPH_EFFECTS = ["shadow", "extrude", "outline", "hollow", "bold", "thin", "fill_texture", "translucent"]
