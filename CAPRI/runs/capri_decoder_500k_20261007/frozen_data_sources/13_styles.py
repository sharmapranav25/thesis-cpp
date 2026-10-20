"""Colour palettes and stroke styles. All sampling goes through a numpy Generator, so results are seed-deterministic."""
import colorsys

import numpy as np

PALETTES = ("random", "pastel", "muted", "dark", "grey", "contrast", "near_background", "saturated", "shades", "match_text")


def lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def colour(rng, palette, background=(255, 255, 255), text_colour=None):
    """One RGB colour from a named palette (relative to the background where that matters). "match_text" = the text's own
    colour (+- a little), so interference cannot be removed by colour; needs text_colour, else falls back to "contrast"."""
    if palette == "match_text":
        if text_colour is None:
            palette = "contrast"
        else:
            return tuple(int(np.clip(c + rng.integers(-12, 13), 0, 255)) for c in text_colour)
    if palette == "random":
        return tuple(int(v) for v in rng.integers(0, 256, 3))
    if palette in ("pastel", "muted", "dark", "saturated"):
        h = rng.random()
        s, v = {"pastel": (rng.uniform(0.15, 0.45), rng.uniform(0.85, 1.0)),
                "muted": (rng.uniform(0.1, 0.35), rng.uniform(0.35, 0.7)),
                "dark": (rng.uniform(0.0, 0.6), rng.uniform(0.05, 0.35)),
                "saturated": (rng.uniform(0.7, 1.0), rng.uniform(0.6, 1.0))}[palette]
        return tuple(int(round(x * 255)) for x in colorsys.hsv_to_rgb(h, s, v))
    if palette == "grey":
        g = int(rng.integers(60, 230))
        return (g, g, g)
    if palette == "contrast":
        lo, hi = (0, 100) if lum(background) > 128 else (160, 256)
        return tuple(int(v) for v in rng.integers(lo, hi, 3))
    if palette == "shades":  # same hue as the background, different lightness/saturation (monochrome schemes)
        h, sat, v = colorsys.rgb_to_hsv(*[x / 255 for x in background])
        sat = float(np.clip(sat + rng.uniform(-0.12, 0.12), 0, 1))
        v = float(np.clip(v * rng.uniform(0.45, 1.35), 0.08, 1))
        return tuple(int(round(x * 255)) for x in colorsys.hsv_to_rgb(h, sat, v))
    if palette == "near_background":
        return tuple(int(np.clip(b + rng.integers(-45, 46), 0, 255)) for b in background)
    raise ValueError(palette)


def palette_choice(rng, allowed=PALETTES):
    return str(rng.choice(list(allowed)))


def stroke_style(rng):
    """clean | rough (jittered vertices) | dashed; antialias on/off."""
    return dict(kind=str(rng.choice(["clean", "clean", "rough", "dashed"])), antialias=bool(rng.random() < 0.7))


def jitter_polyline(pts, rng, amount):
    if amount <= 0:
        return pts
    p = np.asarray(pts, float)
    return [tuple(x) for x in p + rng.normal(0, amount, p.shape)]


def dashes(pts, rng, on=6.0, off=4.0):
    """Split a polyline into dash segments of roughly on/off length (pixels)."""
    p = np.asarray(pts, float)
    out, cur, acc, drawing = [], [tuple(p[0])], 0.0, True
    for a, b in zip(p[:-1], p[1:]):
        seg = np.linalg.norm(b - a)
        t = 0.0
        while t < seg:
            lim = on if drawing else off
            step = min(lim - acc, seg - t)
            t += step
            acc += step
            q = a + (b - a) * (t / max(seg, 1e-9))
            if drawing:
                cur.append(tuple(q))
            if acc >= lim - 1e-9:
                if drawing and len(cur) > 1:
                    out.append(cur)
                cur = [tuple(q)]
                acc, drawing = 0.0, not drawing
    if drawing and len(cur) > 1:
        out.append(cur)
    return out


FALLBACK_FONTS = ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")


def font(path, size):
    """TrueType font at `size` px: `path` if given, else a common system font, else PIL's scalable default."""
    from PIL import ImageFont
    for p in ([path] if path else []) + list(FALLBACK_FONTS):
        try:
            return ImageFont.truetype(p, max(4, int(size)))
        except OSError:
            continue
    return ImageFont.load_default(size=max(4, int(size)))


def with_lum_distance(c, background, dl):
    """Shift colour c (keeping its hue) so its luminance differs from the background's by about dl."""
    lb, lc = lum(background), lum(c)
    target = lb + dl if (lb + dl <= 255 and (lc >= lb or lb - dl < 0)) else lb - dl
    return tuple(int(v) for v in np.clip(np.array(c, float) + (target - lc), 0, 255))
