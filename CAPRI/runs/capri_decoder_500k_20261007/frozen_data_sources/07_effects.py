"""Effects: one function per effect, `fx(canvas, level, rng, **style) -> params`.

`level` is 0..10 (0 draws nothing). Each effect maps s = level / 10 to its settings, then samples a style (palette,
stroke, opacity, layout) so the same level looks different every time. Every op is labelled with the effect's `head`
(the hybrid-v3 feature name it trains). The *labelled* intensity is not the requested level: it is measured from the
visible pixels and mapped through the calibration table (calibrate.py).

Ranges start from what the original CAPTCHA libraries draw (ranges/library_ranges.json, tools/measure_libraries.py)
and are widened. Nothing here is copied from those libraries.
"""
import math

import numpy as np
from PIL import ImageFont

from captcha_effects import styles as S

REGISTRY = {}


def effect(name, head, where="under"):
    def deco(f):
        REGISTRY[name] = dict(fn=f, head=head, where=where)
        f.head, f.where = head, where
        return f
    return deco


def _pal(rng, cv, kw, default=("random", "pastel", "muted", "dark", "grey", "saturated")):
    return kw.get("palette") or S.palette_choice(rng, default)


def _line_mask(cv, pts, width, style):
    def draw(d, s):
        if style["kind"] == "dashed":
            for seg in S.dashes(pts, None):
                d.line([(x * s, y * s) for x, y in seg], fill=255, width=max(1, int(round(width * s))))
        else:
            d.line([(x * s, y * s) for x, y in pts], fill=255, width=max(1, int(round(width * s))), joint="curve")
    return cv.mask(draw, style["antialias"])


def _polyline(cv, rng, pts, width, colour, effect_name, head, layer, opacity, style, params):
    if style["kind"] == "rough":
        pts = S.jitter_polyline(pts, rng, 0.6 + 0.4 * width)
    return cv.blend(_line_mask(cv, pts, width, style), colour, effect_name, head, layer, opacity, params)


# ------------------------------------------------------------------ backgrounds (under the text)

@effect("gradient", "bg_gradient")
def gradient(cv, level, rng, **kw):
    if level <= 0:
        return {}
    s = level / 10
    kind = kw.get("kind") or str(rng.choice(["linear", "linear", "radial", "two_axis"]))
    pal = _pal(rng, cv, kw)
    c0, c1 = _level_pair(rng, pal, cv, 12 + 150 * s)  # luminance distance between the ends set by the level
    yy, xx = np.mgrid[0:cv.h, 0:cv.w].astype(np.float32)
    if kind == "radial":
        t = np.hypot(xx - rng.uniform(0, cv.w), yy - rng.uniform(0, cv.h))
    elif kind == "two_axis":
        t = xx / cv.w + yy / cv.h * rng.uniform(0.3, 1.0)
    else:
        ang = rng.uniform(0, 2 * math.pi)
        t = xx * math.cos(ang) + yy * math.sin(ang)
    t = (t - t.min()) / max(1e-6, t.max() - t.min())
    field = c0[None, None] * (1 - t[..., None]) + c1[None, None] * t[..., None]
    cv.apply_field(field, "gradient", "bg_gradient", "under", 1.0, dict(kind=kind, palette=pal))
    return dict(kind=kind, palette=pal, measured=_contrast(field))


def _level_pair(rng, pal, cv, dl):
    """Two colours of the palette whose luminance differs by ~dl (hue/saturation random)."""
    c0 = np.array(S.colour(rng, pal, cv.background, cv.text_colour), np.float32)
    l0 = S.lum(c0)
    sign = 1.0 if l0 + dl <= 255 or l0 - dl < 0 else -1.0
    c1 = np.array(S.colour(rng, pal, cv.background, cv.text_colour), np.float32)
    c1 = c1 + (l0 + sign * dl - S.lum(c1))
    return np.clip(c0, 0, 255), np.clip(c1, 0, 255)


def _contrast(field):
    """Visible amount of a whole-canvas field: spread (p95 - p5) of its luminance, 0..255."""
    l = 0.299 * field[..., 0] + 0.587 * field[..., 1] + 0.114 * field[..., 2]
    return float(np.percentile(l, 95) - np.percentile(l, 5))


def _value_noise(rng, h, w, cell):
    gh, gw = h // cell + 2, w // cell + 2
    g = rng.random((gh, gw)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / cell
    y0, x0 = yy.astype(int), xx.astype(int)
    fy, fx = yy - y0, xx - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a, b, c, d = g[y0, x0], g[y0, x0 + 1], g[y0 + 1, x0], g[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


@effect("texture", "bg_image_texture")
def texture(cv, level, rng, **kw):
    if level <= 0:
        return {}
    s = level / 10
    octaves = int(rng.integers(2, 5))
    base = float(rng.choice([3, 6, 12, 24]))
    n = sum(_value_noise(rng, cv.h, cv.w, max(1, int(base * 2 ** k))) / 2 ** k for k in range(octaves))
    n = (n - n.min()) / max(1e-6, n.max() - n.min())
    pal = _pal(rng, cv, kw)
    c0, c1 = _level_pair(rng, pal, cv, 12 + 150 * s)
    amp = 1.0
    field = c0[None, None] * (1 - n[..., None]) + c1[None, None] * n[..., None]
    cv.apply_field(field, "texture", "bg_image_texture", "under", 1.0, dict(octaves=octaves, base=base, amplitude=amp))
    return dict(octaves=octaves, base=base, amplitude=amp, palette=pal, scale='fine' if base <= 6 else 'coarse',
                measured=_contrast(field))


@effect("blocks", "bg_blocks")
def blocks(cv, level, rng, **kw):
    """Rectangles behind the text: tiled squares, overlapping rectangles, translucent pastel, speckled fills."""
    if level <= 0:
        return {}
    s = level / 10
    layout = kw.get("layout") or str(rng.choice(["random", "random", "tiled", "overlap_stack", "strips", "dense"]))
    pal = _pal(rng, cv, kw, ("pastel", "pastel", "muted", "random", "grey", "dark", "shades", "shades"))
    opacity = kw.get("opacity") or float(rng.choice([1.0, 0.85, 0.6, 0.4]))
    speckle = kw["speckle"] if "speckle" in kw else bool(rng.random() < 0.3)
    n = int(round(3 + 37 * s))
    min_side, max_side = 0.06, 0.12 + 0.3 * float(rng.random())
    rects = []
    if layout == "tiled":
        side = int(max(4, cv.h * rng.uniform(0.12, 0.3)))
        for y in range(0, cv.h, side):
            for x in range(0, cv.w, side):
                if rng.random() < 0.25 + 0.6 * s:
                    rects.append((x, y, x + side - int(rng.integers(0, 3)), y + side - int(rng.integers(0, 3))))
    elif layout == "dense":  # large overlapping rectangles covering the canvas edge to edge (mostly tall)
        for _ in range(int(round(10 + 20 * s))):
            ww = rng.uniform(0.06, 0.3) * cv.w
            hh = rng.uniform(0.5, 1.2) * cv.h
            x, y = rng.uniform(-0.1 * cv.w, cv.w), rng.uniform(-0.4 * cv.h, 0.6 * cv.h)
            rects.append((x, y, x + ww, y + hh))
    elif layout == "strips":
        for _ in range(int(round(2 + 10 * s))):
            x = rng.uniform(0, cv.w)
            wd = rng.uniform(0.03, 0.15) * cv.w
            rects.append((x, 0, x + wd, cv.h))
    else:
        for _ in range(n):
            ww = rng.uniform(min_side, max_side) * cv.w * (2.0 if layout == "overlap_stack" else 1.0)
            hh = rng.uniform(min_side * 2, max_side * 2) * cv.h
            x, y = rng.uniform(-0.1 * cv.w, cv.w), rng.uniform(-0.1 * cv.h, cv.h)
            rects.append((x, y, x + ww, y + hh))
    dl = 15 + 120 * s  # luminance distance of each block from the background, set by the level
    for (x0, y0, x1, y1) in rects:
        col = S.with_lum_distance(S.colour(rng, pal, cv.background, cv.text_colour), cv.background, dl * rng.uniform(0.8, 1.2))
        cov = cv.mask(lambda d, sc: d.rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], fill=255), antialias=False)
        if speckle:
            cov = cov * (rng.random(cov.shape) < rng.uniform(0.4, 0.8))
        cv.blend(cov, col, "blocks", "bg_blocks", "under", opacity, dict(layout=layout))
    return dict(layout=layout, palette=pal, opacity=opacity, n=len(rects), speckle=speckle)


@effect("grid", "bg_grid")
def grid(cv, level, rng, **kw):
    if level <= 0:
        return {}
    s = level / 10
    spacing = max(4.0, (0.5 - 0.38 * s) * cv.h * rng.uniform(0.88, 1.12))
    width = 1.0 + float(s > 0.6 and rng.random() < 0.5)
    pal = _pal(rng, cv, kw, ("pastel", "random", "grey", "near_background", "muted"))
    multi = bool(rng.random() < 0.5)
    opacity = float(rng.choice([1.0, 0.8, 0.6]))
    angle = 0.0 if rng.random() < 0.75 else rng.uniform(-30, 30)
    style = S.stroke_style(rng)
    col0 = S.colour(rng, pal, cv.background, cv.text_colour)
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    L = 2 * (cv.w + cv.h)
    cx, cy = cv.w / 2, cv.h / 2
    k = -int(L / spacing)
    while k * spacing < L:
        off = k * spacing + rng.uniform(0, 0.15 * spacing)
        for horiz in (True, False):
            if horiz:
                p0 = (cx - L * ca - off * sa, cy - L * sa + off * ca)
                p1 = (cx + L * ca - off * sa, cy + L * sa + off * ca)
            else:
                p0 = (cx + off * ca + L * sa, cy + off * sa - L * ca)
                p1 = (cx + off * ca - L * sa, cy + off * sa + L * ca)
            col = S.colour(rng, pal, cv.background, cv.text_colour) if multi else col0
            cv.blend(_line_mask(cv, [p0, p1], width, style), col, "grid", "bg_grid", "under", opacity,
                     dict(spacing=round(spacing, 2)))
        k += 1
    return dict(spacing=spacing, width=width, palette=pal, multi=multi, angle=angle, opacity=opacity, style=style["kind"])


@effect("circles", "bg_circles")
def circles(cv, level, rng, **kw):
    if level <= 0:
        return {}
    s = level / 10
    n = int(round(2 + 28 * s))
    pal = _pal(rng, cv, kw)
    filled = bool(rng.random() < 0.5)
    opacity = float(rng.choice([1.0, 0.7, 0.45]))
    r_hi = (0.08 + 0.17 * s) if filled else (0.1 + 0.25 * s)
    for _ in range(n):
        r = rng.uniform(0.04, r_hi) * cv.h
        x, y = rng.uniform(0, cv.w), rng.uniform(0, cv.h)
        wd = float(rng.choice([1, 1, 2]))
        col = S.colour(rng, pal, cv.background, cv.text_colour)
        cov = cv.mask(lambda d, sc: d.ellipse([(x - r) * sc, (y - r) * sc, (x + r) * sc, (y + r) * sc],
                                              fill=255 if filled else None, outline=255, width=max(1, int(wd * sc))))
        cv.blend(cov, col, "circles", "bg_circles", "under", opacity)
    return dict(n=n, filled=filled, palette=pal, opacity=opacity)


@effect("decoys", "bg_decoy_chars")
def decoys(cv, level, rng, font_paths=(), **kw):
    if level <= 0:
        return {}
    s = level / 10
    n = int(round(3 + 22 * s))
    pal = _pal(rng, cv, kw, ("pastel", "grey", "near_background", "random"))
    opacity = float(rng.choice([1.0, 0.6, 0.4]))
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"
    for _ in range(n):
        size = int(rng.uniform(0.15, 0.45) * cv.h)
        fp = str(rng.choice(list(font_paths))) if font_paths else None
        font = S.font(fp, max(6, size * cv.supersample))
        ch = str(rng.choice(list(chars)))
        x, y = rng.uniform(0, cv.w), rng.uniform(-0.1 * cv.h, cv.h)
        col = S.colour(rng, pal, cv.background, cv.text_colour)
        cov = cv.mask(lambda d, sc: d.text((x * sc, y * sc), ch, fill=255, font=font))
        cv.blend(cov, col, "decoys", "bg_decoy_chars", "under", opacity)
    return dict(n=n, palette=pal, opacity=opacity)


@effect("border", "border_frame")
def border(cv, level, rng, **kw):
    if level <= 0:
        return {}
    wd = int(round(1 + 5 * level / 10 + rng.uniform(-0.4, 0.4)))
    pal = _pal(rng, cv, kw, ("contrast", "dark", "random"))
    col = S.colour(rng, pal, cv.background, cv.text_colour)
    cov = cv.mask(lambda d, sc: d.rectangle([0, 0, cv.w * sc - 1, cv.h * sc - 1], outline=255, width=wd * sc))
    cv.blend(cov, col, "border", "border_frame", "under", 1.0, dict(width=wd))
    return dict(width=wd, palette=pal)


# ------------------------------------------------------------------ interference (under or over the text)

def _random_line(cv, rng, min_len, max_len):
    x0, y0 = rng.uniform(-0.1 * cv.w, 1.1 * cv.w), rng.uniform(-0.1 * cv.h, 1.1 * cv.h)
    ang = rng.uniform(0, math.pi)
    L = rng.uniform(min_len, max_len) * cv.w
    return [(x0, y0), (x0 + L * math.cos(ang), y0 + L * math.sin(ang))]


def _strokes(name, head, cv, level, rng, kw, n, width_rng, gen_pts, palettes):
    if level <= 0:
        return {}
    pal = _pal(rng, cv, kw, palettes)
    style = S.stroke_style(rng)
    opacity = float(rng.choice([1.0, 1.0, 0.75, 0.5]))
    where = kw.get("where", "under")
    for _ in range(n):
        wd = float(rng.uniform(*width_rng))
        _polyline(cv, rng, gen_pts(), wd, S.colour(rng, pal, cv.background, cv.text_colour), name, head, where, opacity, style, {})
    return dict(n=n, palette=pal, opacity=opacity, style=style["kind"], width=width_rng)


@effect("lines", "straight_lines")
def lines(cv, level, rng, **kw):
    s = level / 10
    return _strokes("lines", "straight_lines", cv, level, rng, kw, int(round(1 + 24 * s)), (1.0, 1.0 + 2.0 * float(rng.random())),
                    lambda: _random_line(cv, rng, 0.2, 1.2), ("random", "dark", "grey", "contrast", "muted", "pastel", "match_text"))


@effect("bands", "bands")
def bands(cv, level, rng, **kw):
    """Wide, near-horizontal bars crossing the whole canvas (solid strokes; e.g. light grey bars behind/through text)."""
    if level <= 0:
        return {}
    s = level / 10
    n = int(round(1 + 4 * s))
    pal = _pal(rng, cv, kw, ("grey", "near_background", "muted", "pastel"))
    opacity = float(rng.choice([1.0, 1.0, 0.75]))
    w_lo = max(3.0, (0.04 + 0.06 * s) * cv.h)
    params = dict(n=n, palette=pal, opacity=opacity)
    for _ in range(n):
        wd = float(rng.uniform(w_lo, w_lo + (0.03 + 0.08 * s) * cv.h))
        y0, y1 = rng.uniform(0, cv.h), rng.uniform(0, cv.h)
        y1 = y0 + float(np.clip(y1 - y0, -0.6 * cv.h, 0.6 * cv.h))
        pts = [(-0.05 * cv.w, y0), (1.05 * cv.w, y1)]
        style = dict(kind="clean", antialias=bool(rng.random() < 0.5))
        _polyline(cv, rng, pts, wd, S.colour(rng, pal, cv.background, cv.text_colour), "bands", "bands", kw.get("where", "under"),
                  opacity, style, {})
    return params


def _curve_pts(cv, rng):
    x = np.linspace(-0.05 * cv.w, 1.05 * cv.w, 60)
    amp = rng.uniform(0.05, 0.4) * cv.h
    per = rng.uniform(0.3, 1.5) * cv.w
    ph = rng.uniform(0, 2 * math.pi)
    y = rng.uniform(0.15, 0.85) * cv.h + amp * np.sin(2 * math.pi * x / per + ph)
    x0, x1 = sorted(rng.uniform(0, 1, 2))
    keep = (x >= x0 * cv.w - 0.3 * cv.w) & (x <= x1 * cv.w + 0.3 * cv.w)
    if keep.sum() < 4:
        keep[:] = True
    return list(zip(x[keep], y[keep]))


@effect("curves", "curves")
def curves(cv, level, rng, **kw):
    s = level / 10
    return _strokes("curves", "curves", cv, level, rng, kw, int(round(1 + 14 * s)), (1.0, 1.0 + 2.0 * float(rng.random())),
                    lambda: _curve_pts(cv, rng), ("random", "pastel", "grey", "dark", "muted", "match_text"))


def _walk(cv, rng):
    n = int(rng.integers(6, 30))
    p = np.array([rng.uniform(0, cv.w), rng.uniform(0, cv.h)])
    pts = [tuple(p)]
    for _ in range(n):
        p = p + rng.normal(0, 0.12 * cv.h, 2)
        pts.append(tuple(p))
    return pts


@effect("scribbles", "scribbles")
def scribbles(cv, level, rng, **kw):
    s = level / 10
    return _strokes("scribbles", "scribbles", cv, level, rng, kw, int(round(1 + 9 * s)), (1.0, 2.5),
                    lambda: _walk(cv, rng), ("random", "dark", "pastel", "grey"))


@effect("dots", "dot_noise")
def dots(cv, level, rng, **kw):
    if level <= 0:
        return {}
    s = level / 10
    area = cv.w * cv.h
    pal = _pal(rng, cv, kw, ("random", "dark", "grey", "pastel", "contrast", "match_text"))
    r_max = float(rng.choice([0.6, 0.8, 1.0, 1.5]))
    r_mean = (0.3 + r_max) / 2
    target = 0.006 + 0.16 * s                         # visible fraction the level asks for
    n = int(round(target * area / (math.pi * max(r_mean, 0.5) ** 2 + 1.0)))
    multi = bool(rng.random() < 0.5)
    where = kw.get("where", "under")
    col0 = S.colour(rng, pal, cv.background, cv.text_colour)
    groups = 6 if multi else 1
    for gi in range(groups):
        k = n // groups
        xs, ys = rng.uniform(0, cv.w, k), rng.uniform(0, cv.h, k)
        rs = rng.uniform(0.3, r_max, k)
        col = S.colour(rng, pal, cv.background, cv.text_colour) if multi else col0

        def draw(d, sc, xs=xs, ys=ys, rs=rs):
            for x, y, r in zip(xs, ys, rs):
                d.ellipse([(x - r) * sc, (y - r) * sc, (x + r) * sc, (y + r) * sc], fill=255)
        cv.blend(cv.mask(draw), col, "dots", "dot_noise", where, 1.0, dict(n=k))
    return dict(n=n, palette=pal, r_max=r_max, multi=multi)


@effect("blobs", "blob_noise")
def blobs(cv, level, rng, **kw):
    if level <= 0:
        return {}
    s = level / 10
    n = int(round(3 + 40 * s))
    pal = _pal(rng, cv, kw)
    opacity = float(rng.choice([1.0, 0.7, 0.5]))
    where = kw.get("where", "under")
    for _ in range(n):
        r = rng.uniform(0.03, 0.12) * cv.h + 1.5
        cx, cy = rng.uniform(0, cv.w), rng.uniform(0, cv.h)
        k = int(rng.integers(5, 10))
        ang = np.sort(rng.uniform(0, 2 * math.pi, k))
        rad = r * rng.uniform(0.6, 1.3, k)
        pts = list(zip(cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
        cov = cv.mask(lambda d, sc: d.polygon([(x * sc, y * sc) for x, y in pts], fill=255))
        cv.blend(cov, S.colour(rng, pal, cv.background, cv.text_colour), "blobs", "blob_noise", where, opacity)
    return dict(n=n, palette=pal, opacity=opacity)


# ------------------------------------------------------------------ over the text (post step)

@effect("strike", "strike_through", where="over")
def strike(cv, level, rng, text_box=None, **kw):
    """Lines through the text's own box (needs text_box = (x0, y0, x1, y1) of the drawn text)."""
    if level <= 0:
        return {}
    x0, y0, x1, y1 = text_box or (0.1 * cv.w, 0.3 * cv.h, 0.9 * cv.w, 0.7 * cv.h)
    n = 1 + int(round(2 * level / 10))
    wd = 1.0 + 3.0 * level / 10
    pal = _pal(rng, cv, kw, ("contrast", "dark", "random"))
    style = S.stroke_style(rng)
    for _ in range(n):
        ya, yb = rng.uniform(y0, y1), rng.uniform(y0, y1)
        pts = [(x0 - rng.uniform(0, 0.1) * cv.w, ya), (x1 + rng.uniform(0, 0.1) * cv.w, yb)]
        if rng.random() < 0.4:
            pts = _curve_pts(cv, rng)
        _polyline(cv, rng, pts, wd, S.colour(rng, pal, cv.background, cv.text_colour), "strike", "strike_through", "over", 1.0, style, {})
    return dict(n=n, width=wd, palette=pal, style=style['kind'])


@effect("noise_over", "noise_over_text", where="over")
def noise_over(cv, level, rng, **kw):
    """Dots/short strokes drawn on top of everything, including the glyphs."""
    if level <= 0:
        return {}
    kind = str(rng.choice(["dots", "dots", "lines"]))
    if kind == "dots":
        p = dots(cv, level, rng, where="over", **{k: v for k, v in kw.items() if k != "where"})
    else:
        p = lines(cv, level, rng, where="over", **{k: v for k, v in kw.items() if k != "where"})
    for r in cv.records:
        if r["layer"] == "over" and r["head"] in ("dot_noise", "straight_lines"):
            r["head"], r["effect"] = "noise_over_text", "noise_over"
    return dict(kind=kind, **p)


BACKGROUND_EFFECTS = ["gradient", "texture", "blocks", "grid", "circles", "decoys", "border"]
INTERFERENCE_EFFECTS = ["lines", "bands", "curves", "scribbles", "dots", "blobs"]
OVER_EFFECTS = ["strike", "noise_over"]
