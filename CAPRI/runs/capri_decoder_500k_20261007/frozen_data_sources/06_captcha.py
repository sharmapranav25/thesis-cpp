"""Complete labelled CAPTCHA in one call: effects (under) -> our text -> glyph effects -> over-text -> post.

    bank = FontBank()
    out = render_captcha(rng, bank, size=(200, 60))               # random effects + random text
    out = render_captcha(rng, bank, size, effects_spec, text_spec)  # or given specs
    out["image"], out["opaque"], out["translucent"]                   # pixels + per-pixel op ids
    out["effects"]  {head: {present, level, value, style, effect}}    # level from the visible amount (calibration)
    out["text"]     text_labels(...): text heads (0-10 where meaningful), content heads, style, per-glyph slots
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from captcha_effects import render as R
from captcha_effects.calibrate import level_of
from captcha_effects.effects import REGISTRY
from captcha_effects.measure import visible_amount
from captcha_effects.text import draw_text_spec, sample_text_spec, text_labels


@lru_cache(maxsize=1)
def calibration(path=None):
    p = Path(path or Path(__file__).resolve().parents[1] / "calibration.json")
    return json.loads(p.read_text(encoding="utf-8"))


MIN_VISIBLE = 0.6     # each glyph keeps >= 60% of the pixels it had when drawn
MIN_CONTRAST = 22.0   # median |luminance(glyph) - luminance(ring around it)| over glyphs, 0..255


def legibility(cv, txt):
    """Can a person still read it? Per glyph: visible share of its drawn pixels, and luminance contrast against a ring
    of pixels around it (in the final image)."""
    from PIL import Image, ImageFilter
    img = cv.image().astype(np.float32)
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    vis, con = [], []
    for g in txt["glyphs"]:
        m = cv.opaque == g["id"]
        drawn = max(1, g.get("visible_px_after_glyph_fx", g.get("visible_px", 1)))  # hidden by over-text / post only
        vis.append(float(m.sum()) / drawn)
        if m.sum() < 5:
            con.append(0.0)
            continue
        ring = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))) > 127
        ring &= ~m
        con.append(abs(float(np.median(lum[m])) - float(np.median(lum[ring]))) if ring.any() else 0.0)
    ok = bool(min(vis) >= MIN_VISIBLE and float(np.median(con)) >= MIN_CONTRAST)
    return dict(min_visible=round(min(vis), 3), median_contrast=round(float(np.median(con)), 1), ok=ok)


def render_captcha(rng, bank, size=(200, 60), effects_spec=None, text_spec=None, font_split="train", p_on=None):
    """One labelled CAPTCHA. Check out["text"]["usable"] (no clipping, legible) and redraw when False."""
    effects_spec = effects_spec or R.sample_spec(rng, p_on=p_on or {n: 0.1 for n in R.SAMPLED})
    text_spec = text_spec or sample_text_spec(rng, bank, split=font_split)
    cv = R.new_canvas(size[0], size[1], effects_spec, rng)
    log = R.draw_under(cv, effects_spec, rng)
    txt = draw_text_spec(cv, text_spec, rng, bank)
    if txt is None:
        return None
    log += R.apply_glyph(cv, effects_spec, rng, ids=txt["ids"])
    for g in txt["glyphs"]:  # glyph pixels after glyph effects (hollow / thin remove pixels by design)
        g["visible_px_after_glyph_fx"] = int((cv.opaque == g["id"]).sum())
    log += R.draw_over(cv, effects_spec, rng, txt["text_box"])
    log += R.apply_post(cv, effects_spec, rng)
    table = calibration()
    effects = {}
    for e in log:  # several effects can share a head (e.g. scribbles + scribbles2): keep the strongest
        v = visible_amount(e["name"], cv.records, cv.opaque, cv.translucent, cv.background, e["params"].get("measured"))
        lv = level_of(table, e["name"], v)
        cur = effects.get(e["head"])
        if cur is None or lv > cur["level"]:
            effects[e["head"]] = dict(present=int(lv > 0), level=lv, value=round(v, 5), style=e["style"], effect=e["name"],
                                      level_requested=e["level_requested"])
    # every head not drawn is explicitly absent
    for name in R.SAMPLED:
        effects.setdefault(REGISTRY[name]["head"], dict(present=0, level=0, value=0.0, style={}, effect=None))
    # glyph boxes after all post effects (warps move labels with pixels)
    for g in txt["glyphs"]:
        ys, xs = np.nonzero(cv.opaque == g["id"])
        g["box_final"] = None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    tl = text_labels(txt)
    for s, g in zip(tl["slots"], txt["glyphs"]):
        s["box"] = g["box_final"]
    leg = legibility(cv, txt)
    tl["legibility"] = leg
    tl["usable"] = bool(tl["usable"] and leg["ok"])
    return dict(image=cv.image(), opaque=cv.opaque, translucent=cv.translucent, background=list(cv.background),
                effects=effects, text=tl, effects_spec=effects_spec, text_spec=text_spec, records=cv.records)
