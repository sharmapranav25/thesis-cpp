"""The single definition of an effect's *visible amount*, used by calibration and by whoever labels rendered images.

  pixel effects     (layers under/over the text, glyph additions): visible pixels of the effect's own ops in the FINAL
                    label maps / canvas pixels -- so later effects that cover them reduce the amount
  contrast effects  (blocks, pattern: can cover the whole canvas): sum over the effect's ops of visible pixels x
                    |luminance(op colour) - luminance(background)| / 255, / canvas pixels
  measured effects  (fields, glyph in-place changes, warps, degradations): the amount the op recorded when it ran
                    (contrast spread, changed share of glyph pixels, displacement, sigma, 100-quality, ...)
"""
import numpy as np

from captcha_effects.styles import lum

CONTRAST = {"blocks", "pattern"}
MEASURED = {"gradient", "texture", "hollow", "bold", "thin", "fill_texture", "translucent",
            "wave", "morph", "perspective", "blur", "swirl", "invert", "grain", "jpeg", "pixelate", "resample", "quantize"}


def visible_by_id(opaque, translucent):
    ids = np.where(opaque > 0, opaque, translucent)
    u, c = np.unique(ids, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


def visible_amount(name, records, opaque, translucent, background, measured=None):
    """records: the canvas records (dicts with id, effect, colour); measured: the value the effect returned, if any."""
    if name in MEASURED:
        return float(measured or 0.0)
    area = float(opaque.shape[0] * opaque.shape[1])
    vis = visible_by_id(opaque, translucent)
    own = [r for r in records if r.get("effect") == name]
    if name in CONTRAST:
        lb = lum(background)
        return float(sum(vis.get(r["id"], 0) * abs(lum(r["colour"]) - lb) / 255.0 for r in own if r.get("colour"))) / area
    return float(sum(vis.get(r["id"], 0) for r in own)) / area
