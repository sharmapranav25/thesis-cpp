"""More effects: pattern backgrounds, caption, occluders, cuts, inverted regions, local warps and image-level degradations.

Same contract as effects.py: `fx(canvas, level, rng, **style) -> params`, level 0 draws nothing, labels exact.
Image-level degradations (grain, jpeg, pixelate, resample, quantize, invert) change pixel values only; they record the
measured amount they applied (no pixel ownership changes, except invert which records its region).
"""
import io
import math

import numpy as np
from PIL import Image

from captcha_effects import styles as S
from captcha_effects.effects import effect, _pal, _polyline, _line_mask, _curve_pts, _value_noise
from captcha_effects.warp import _remap


# ------------------------------------------------------------------ backgrounds

@effect("pattern", "bg_pattern")
def pattern(cv, level, rng, **kw):
    """Regular printed patterns: stripes, checker, cross-hatch, dot lattice, wavy stripes."""
    if level <= 0:
        return {}
    s = level / 10
    kind = kw.get("kind") or str(rng.choice(["stripes", "checker", "hatch", "dot_lattice", "wavy"]))
    period = max(3.0, (0.45 - 0.33 * s) * cv.h * rng.uniform(0.85, 1.15))
    ang = math.radians(rng.choice([0, 45, 90, -45, rng.uniform(-60, 60)]))
    pal = _pal(rng, cv, kw, ("pastel", "grey", "muted", "shades", "random"))
    col = S.with_lum_distance(S.colour(rng, pal, cv.background, cv.text_colour), cv.background, 15 + 120 * s)
    opacity = 1.0
    yy, xx = np.mgrid[0:cv.h, 0:cv.w].astype(np.float32)
    u = xx * math.cos(ang) + yy * math.sin(ang)
    v = -xx * math.sin(ang) + yy * math.cos(ang)
    duty = 0.3 + 0.15 * s
    if kind == "stripes":
        m = ((u / period) % 1.0) < duty
    elif kind == "checker":
        m = ((np.floor(u / period) + np.floor(v / period)) % 2) == 0
    elif kind == "hatch":
        m = (((u / period) % 1.0) < duty * 0.5) | (((v / period) % 1.0) < duty * 0.5)
    elif kind == "dot_lattice":
        r = period * (0.15 + 0.25 * s)
        m = np.hypot((u % period) - period / 2, (v % period) - period / 2) < r
    else:
        m = (((u + 0.25 * period * np.sin(2 * math.pi * v / (2 * period))) / period) % 1.0) < duty
    cv.blend(m.astype(np.float32), col, "pattern", "bg_pattern", "under", opacity, dict(kind=kind, period=round(period, 2)))
    return dict(kind=kind, period=period, palette=pal, opacity=opacity)


@effect("caption", "fixed_caption")
def caption(cv, level, rng, font_paths=(), **kw):
    """A fixed line of small text (credit/watermark) along the top or bottom edge."""
    if level <= 0:
        return {}
    s = level / 10
    words = ["captcha", "secure", "verify", "powered", "v1.4", "example.org", "antispam", "imagecode"]
    txt = " ".join(rng.choice(words, size=1 + int(round(2 * s))))
    size = max(6, int((0.1 + 0.1 * s) * cv.h))
    font = S.font(str(rng.choice(list(font_paths))) if font_paths else None, size * cv.supersample)
    position = 'bottom' if rng.random() < 0.7 else 'top'
    y = cv.h - size - 1 if position == 'bottom' else 1
    x = rng.uniform(0, 0.3) * cv.w
    col = S.colour(rng, _pal(rng, cv, kw, ("contrast", "grey", "dark")), cv.background)
    cov = cv.mask(lambda d, sc: d.text((x * sc, y * sc), txt, fill=255, font=font))
    cv.blend(cov, col, "caption", "fixed_caption", "under", 0.5 + 0.5 * s, dict(text=txt))
    return dict(text=txt, size=size, position=position)


# ------------------------------------------------------------------ over the text

@effect("occluders", "occlusion", where="over")
def occluders(cv, level, rng, text_box=None, **kw):
    """Shapes drawn on top of the glyphs, hiding parts of them."""
    if level <= 0:
        return {}
    s = level / 10
    x0, y0, x1, y1 = text_box or (0.1 * cv.w, 0.25 * cv.h, 0.9 * cv.w, 0.75 * cv.h)
    n = int(round(1 + 9 * s))
    shape = str(rng.choice(["rect", "ellipse", "polygon"]))
    pal = _pal(rng, cv, kw, ("near_background", "random", "pastel", "shades", "grey"))
    opacity = float(rng.choice([1.0, 0.8, 0.6]))
    for _ in range(n):
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        r = rng.uniform(0.05, 0.08 + 0.12 * s) * cv.h
        if shape == "rect":
            draw = (lambda cx=cx, cy=cy, r=r: lambda d, sc: d.rectangle([(cx - r) * sc, (cy - r * 0.6) * sc, (cx + r) * sc, (cy + r * 0.6) * sc], fill=255))()
        elif shape == "ellipse":
            draw = (lambda cx=cx, cy=cy, r=r: lambda d, sc: d.ellipse([(cx - r) * sc, (cy - r) * sc, (cx + r) * sc, (cy + r) * sc], fill=255))()
        else:
            k = int(rng.integers(3, 7))
            ang = np.sort(rng.uniform(0, 2 * math.pi, k))
            pts = list(zip(cx + r * np.cos(ang), cy + r * np.sin(ang)))
            draw = (lambda pts=pts: lambda d, sc: d.polygon([(x * sc, y * sc) for x, y in pts], fill=255))()
        cv.blend(cv.mask(draw), S.colour(rng, pal, cv.background, cv.text_colour), "occluders", "occlusion", "over", opacity, dict(shape=shape))
    return dict(n=n, shape=shape, palette=pal, opacity=opacity)


@effect("cuts", "glyph_cuts", where="over")
def cuts(cv, level, rng, text_box=None, **kw):
    """Gaps through the glyphs: thin strokes in the background colour across the text."""
    if level <= 0:
        return {}
    s = level / 10
    x0, y0, x1, y1 = text_box or (0.1 * cv.w, 0.25 * cv.h, 0.9 * cv.w, 0.75 * cv.h)
    n = int(round(1 + 7 * s))
    wd = 1.0 + 2.0 * s
    style = dict(kind="clean", antialias=bool(rng.random() < 0.5))
    for _ in range(n):
        if rng.random() < 0.5:  # horizontal-ish slice
            ya, yb = rng.uniform(y0, y1), rng.uniform(y0, y1)
            pts = [(x0 - 2, ya), (x1 + 2, yb)]
        else:                   # short slash across one glyph
            cx = rng.uniform(x0, x1)
            pts = [(cx - rng.uniform(2, 8), y0 - 2), (cx + rng.uniform(2, 8), y1 + 2)]
        _polyline(cv, rng, pts, wd, cv.background, "cuts", "glyph_cuts", "over", 1.0, style, {})
    return dict(n=n, width=wd)


@effect("invert", "inverted_regions", where="post")
def invert(cv, level, rng, **kw):
    """Colour-inverted bands or rectangles (negative regions)."""
    if level <= 0:
        return {}
    s = level / 10
    m = np.zeros((cv.h, cv.w), bool)
    for _ in range(int(round(1 + 3 * s))):
        if rng.random() < 0.5:
            x = rng.uniform(0, cv.w)
            wd = rng.uniform(0.05, 0.08 + 0.2 * s) * cv.w
            m[:, int(x):int(x + wd)] = True
        else:
            x, y = rng.uniform(0, cv.w), rng.uniform(0, cv.h)
            ww, hh = rng.uniform(0.1, 0.1 + 0.3 * s) * cv.w, rng.uniform(0.2, 0.6) * cv.h
            m[int(y):int(y + hh), int(x):int(x + ww)] = True
    cv.img[m] = 255 - cv.img[m]
    frac = float(m.mean())
    cv.record(effect="invert", head="inverted_regions", layer="post", params={}, measured=frac)
    return dict(measured=frac)


# ------------------------------------------------------------------ local warps and degradations (post)

@effect("swirl", "local_warp", where="post")
def swirl(cv, level, rng, **kw):
    """Local swirls/ripples at a few centres (as opposed to a whole-image wave)."""
    if level <= 0:
        return {}
    s = level / 10
    yy, xx = np.mgrid[0:cv.h, 0:cv.w].astype(np.float32)
    dx, dy = np.zeros_like(xx), np.zeros_like(yy)
    for _ in range(1 + int(round(2 * s))):
        cx, cy = rng.uniform(0.15, 0.85) * cv.w, rng.uniform(0.2, 0.8) * cv.h
        rad = rng.uniform(0.45, 0.65) * cv.h
        strength = (0.25 + 0.95 * s) * float(rng.choice([-1, 1]))  # capped: letters bent, not torn apart
        rx, ry = xx - cx, yy - cy
        r = np.hypot(rx, ry)
        th = strength * np.exp(-(r / rad) ** 2)
        c, sn = np.cos(th), np.sin(th)
        dx += (c * rx - sn * ry) - rx
        dy += (sn * rx + c * ry) - ry
    disp = _remap(cv, dx, dy)
    cv.record(effect="swirl", head="local_warp", layer="post", params={}, measured=disp)
    return dict(measured=disp)


@effect("grain", "pixel_noise", where="post")
def grain(cv, level, rng, **kw):
    sigma = 3.0 + 42.0 * level / 10 if level else 0.0
    if not sigma:
        return {}
    mono = bool(rng.random() < 0.6)
    n = rng.normal(0, sigma, (cv.h, cv.w, 1) if mono else (cv.h, cv.w, 3)).astype(np.float32)
    cv.img = np.clip(cv.img + n, 0, 255)
    cv.record(effect="grain", head="pixel_noise", layer="post", params=dict(mono=mono), measured=sigma)
    return dict(sigma=sigma, mono=mono, measured=sigma)


@effect("jpeg", "compression", where="post")
def jpeg(cv, level, rng, **kw):
    if level <= 0:
        return {}
    q = int(round(85 - 80 * level / 10))
    buf = io.BytesIO()
    Image.fromarray(cv.image()).save(buf, "JPEG", quality=q)
    out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), np.float32)
    err = float(np.abs(out - cv.img).mean())
    cv.img = out
    cv.record(effect="jpeg", head="compression", layer="post", params=dict(quality=q), measured=100.0 - q)
    return dict(quality=q, mean_abs_error=err, measured=100.0 - q)


@effect("pixelate", "pixelation", where="post")
def pixelate(cv, level, rng, **kw):
    if level <= 0:
        return {}
    block = 1.4 + 4.6 * level / 10
    w2, h2 = max(1, int(cv.w / block)), max(1, int(cv.h / block))
    im = Image.fromarray(cv.image()).resize((w2, h2), Image.BILINEAR).resize((cv.w, cv.h), Image.NEAREST)
    cv.img = np.asarray(im, np.float32)
    cv.record(effect="pixelate", head="pixelation", layer="post", params=dict(block=block), measured=block)
    return dict(block=block, measured=block)


@effect("resample", "low_resolution", where="post")
def resample(cv, level, rng, **kw):
    """Rendered small and upscaled smoothly (soft, low-resolution look)."""
    if level <= 0:
        return {}
    f = 1.3 + 3.7 * level / 10
    w2, h2 = max(2, int(cv.w / f)), max(2, int(cv.h / f))
    down = str(rng.choice(["BILINEAR", "NEAREST", "BOX"]))
    im = Image.fromarray(cv.image()).resize((w2, h2), getattr(Image, down)).resize((cv.w, cv.h), Image.BICUBIC)
    cv.img = np.asarray(im, np.float32)
    cv.record(effect="resample", head="low_resolution", layer="post", params=dict(factor=f, down=down), measured=f)
    return dict(factor=f, measured=f)


@effect("quantize", "colour_quantization", where="post")
def quantize(cv, level, rng, **kw):
    """Few-colour palette (GIF-like), optionally dithered."""
    if level <= 0:
        return {}
    ncol = int(round(64 * (1 - level / 10) ** 2 + 2))
    dither = bool(rng.random() < 0.5)
    im = Image.fromarray(cv.image()).quantize(colors=ncol, dither=Image.FLOYDSTEINBERG if dither else Image.NONE).convert("RGB")
    cv.img = np.asarray(im, np.float32)
    cv.record(effect="quantize", head="colour_quantization", layer="post", params=dict(colours=ncol, dither=dither),
              measured=float(66 - ncol))
    return dict(colours=ncol, dither=dither, measured=float(66 - ncol))


# ------------------------------------------------------------------ scribble styles (widened)

@effect("scribbles2", "scribbles")
def scribbles2(cv, level, rng, **kw):
    """Hand-drawn squiggles: loops, zig-zags, spirals."""
    if level <= 0:
        return {}
    s = level / 10
    kind = kw.get("kind") or str(rng.choice(["loops", "zigzag", "spiral"]))
    pal = _pal(rng, cv, kw, ("random", "dark", "pastel", "grey", "contrast"))
    n = int(round(1 + 6 * s))
    for _ in range(n):
        cx, cy = rng.uniform(0, cv.w), rng.uniform(0, cv.h)
        L = rng.uniform(0.2, 0.25 + 0.5 * s) * cv.w
        t = np.linspace(0, 1, 120)
        if kind == "loops":
            r = rng.uniform(0.08, 0.25) * cv.h
            f = rng.uniform(3, 9)
            pts = list(zip(cx - L / 2 + L * t + r * np.cos(2 * math.pi * f * t), cy + r * np.sin(2 * math.pi * f * t)))
        elif kind == "zigzag":
            k = int(rng.integers(5, 16))
            xs = np.linspace(cx - L / 2, cx + L / 2, k)
            ys = cy + rng.uniform(0.1, 0.35) * cv.h * np.where(np.arange(k) % 2 == 0, 1, -1)
            pts = list(zip(xs, ys))
        else:
            th = t * rng.uniform(4, 10) * math.pi
            r = t * rng.uniform(0.15, 0.4) * cv.h
            pts = list(zip(cx + r * np.cos(th), cy + r * np.sin(th)))
        wd = float(rng.uniform(1.0, 2.5))
        style = S.stroke_style(rng)
        _polyline(cv, rng, pts, wd, S.colour(rng, pal, cv.background, cv.text_colour), "scribbles2", "scribbles", kw.get("where", "under"),
                  1.0, style, {})
    return dict(kind=kind, n=n, palette=pal)


EXTRA_UNDER = ["pattern", "caption", "scribbles2"]
EXTRA_OVER = ["occluders", "cuts"]
EXTRA_POST = ["swirl", "invert", "grain", "jpeg", "pixelate", "resample", "quantize"]
