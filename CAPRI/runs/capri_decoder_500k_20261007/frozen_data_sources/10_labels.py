"""Style labels: the categorical choices each effect made, so a model can predict a complete renderer recipe
(effect + level + style) and a controller can render it back.

STYLE_SCHEMA[effect] = {field: [allowed values]}. `style_labels(effect, params)` turns the params an effect returned
into one value per field (fields the effect did not report are "na"). Opacity is binned (solid / semi / faint).
"""
from captcha_effects.styles import PALETTES

PAL = list(PALETTES)
STROKE = ["clean", "rough", "dashed"]
OPACITY = ["solid", "semi", "faint"]

STYLE_SCHEMA = {
    "gradient": dict(kind=["linear", "radial", "two_axis"], palette=PAL),
    "texture": dict(scale=["fine", "coarse"], palette=PAL),
    "blocks": dict(layout=["random", "tiled", "overlap_stack", "strips", "dense"], palette=PAL, opacity=OPACITY,
                   speckle=[False, True]),
    "grid": dict(palette=PAL, multi=[False, True], slanted=[False, True], stroke=STROKE, opacity=OPACITY),
    "circles": dict(filled=[False, True], palette=PAL, opacity=OPACITY),
    "decoys": dict(palette=PAL, opacity=OPACITY),
    "border": dict(palette=PAL),
    "lines": dict(palette=PAL, stroke=STROKE, opacity=OPACITY),
    "bands": dict(palette=PAL, opacity=OPACITY),
    "curves": dict(palette=PAL, stroke=STROKE, opacity=OPACITY),
    "scribbles": dict(palette=PAL, stroke=STROKE, opacity=OPACITY),
    "scribbles2": dict(kind=["loops", "zigzag", "spiral"], palette=PAL),
    "dots": dict(palette=PAL, multi=[False, True]),
    "blobs": dict(palette=PAL, opacity=OPACITY),
    "pattern": dict(kind=["stripes", "checker", "hatch", "dot_lattice", "wavy"], palette=PAL),
    "caption": dict(position=["top", "bottom"]),
    "shadow": dict(palette=PAL),
    "extrude": dict(palette=PAL),
    "outline": dict(palette=PAL),
    "fill_texture": dict(kind=["stripes", "gradient", "noise", "checker"]),
    "strike": dict(palette=PAL, stroke=STROKE),
    "noise_over": dict(kind=["dots", "lines"], palette=PAL),
    "occluders": dict(shape=["rect", "ellipse", "polygon"], palette=PAL, opacity=OPACITY),
    "wave": dict(axis=["y", "x", "both"]),
    "grain": dict(mono=[False, True]),
    "resample": dict(down=["BILINEAR", "NEAREST", "BOX"]),
    "quantize": dict(dither=[False, True]),
}


def _opacity_bin(o):
    return "solid" if o >= 0.95 else "semi" if o >= 0.55 else "faint"


def style_labels(effect, params):
    """One categorical value per schema field (or "na")."""
    schema = STYLE_SCHEMA.get(effect, {})
    p = dict(params or {})
    if "opacity" in p:
        p["opacity"] = _opacity_bin(float(p["opacity"]))
    if "style" in p:
        p["stroke"] = p["style"]
    if "angle" in p:
        p["slanted"] = abs(float(p["angle"])) > 1e-6
    out = {}
    for field, allowed in schema.items():
        v = p.get(field, "na")
        out[field] = v if v in allowed else "na"
    return out
