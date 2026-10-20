"""Composition: a spec (list of effects with levels) -> drawn in three passes around the caller's text.

    spec = {"background": [r, g, b] | "random",
            "effects": [{"name": "blocks", "level": 6}, {"name": "grid", "level": 3}, {"name": "wave", "level": 4}, ...]}

  draw_under(cv, spec, rng)            backgrounds + interference marked "under"
  (caller draws the text: e.g. a real CAPTCHA library, or draw_text() here for standalone use)
  apply_glyph(cv, spec, rng, ids)      effects on the glyph pixels (shadow, outline, hollow, bold, fill texture...)
  draw_over(cv, spec, rng, text_box)   strike-through / noise / occluders / cuts over the text
  apply_post(cv, spec, rng)            warps and image-level degradations on everything

sample_spec(rng, ...) draws a balanced spec: each effect independently on with probability p_on, level uniform 1..10.
"""
import numpy as np
from PIL import ImageFont

from captcha_effects import styles as S
from captcha_effects.canvas import Canvas
from captcha_effects.effects import REGISTRY, BACKGROUND_EFFECTS, INTERFERENCE_EFFECTS, OVER_EFFECTS
from captcha_effects.warp import POST_EFFECTS
from captcha_effects.extras import EXTRA_UNDER, EXTRA_OVER, EXTRA_POST
from captcha_effects.glyph import GLYPH_EFFECTS
from captcha_effects.labels import style_labels

UNDER = BACKGROUND_EFFECTS + INTERFERENCE_EFFECTS + EXTRA_UNDER
OVER = OVER_EFFECTS + EXTRA_OVER
POST = POST_EFFECTS + EXTRA_POST
ALL_EFFECTS = UNDER + GLYPH_EFFECTS + OVER + POST
# effects drawn by default sampling; "caption" stays available but is not sampled (no head: unlabelled extra text would
# confuse the text reader; user decision 2026-10-03)
NOT_SAMPLED = ("caption",)
SAMPLED = [n for n in ALL_EFFECTS if n not in NOT_SAMPLED]


def new_canvas(w, h, spec, rng):
    bg = spec.get("background", "random")
    if bg == "random":
        bg = S.colour(rng, str(rng.choice(["pastel", "pastel", "grey", "muted", "dark", "random"])), (255, 255, 255))
        if rng.random() < 0.35:
            bg = (255, 255, 255)
    return Canvas(int(w), int(h), tuple(int(v) for v in bg))


def _run(cv, spec, rng, names, **extra):
    out = []
    for e in spec.get("effects", []):
        if e["name"] in names and e.get("level", 0) > 0:
            fx = REGISTRY[e["name"]]
            p = fx["fn"](cv, int(e["level"]), rng, **{**e.get("style", {}), **extra})
            out.append(dict(name=e["name"], head=fx["head"], level_requested=int(e["level"]), params=p,
                            style=style_labels(e["name"], p)))
    return out


def draw_under(cv, spec, rng, font_paths=(), text_colour=None):
    """text_colour: the colour the text will have, if known in advance (enables match_text under the text)."""
    if text_colour is not None:
        cv.text_colour = tuple(text_colour)
    return _run(cv, spec, rng, UNDER, font_paths=font_paths)


def apply_glyph(cv, spec, rng, ids=None):
    return _run(cv, spec, rng, GLYPH_EFFECTS, ids=ids)


def draw_over(cv, spec, rng, text_box=None):
    return _run(cv, spec, rng, OVER, text_box=text_box)


def apply_post(cv, spec, rng):
    return _run(cv, spec, rng, POST)


def draw_text(cv, text, rng, font_path=None, colour=None):
    """Plain text for standalone use and calibration sheets (the corpus uses real CAPTCHA libraries instead)."""
    size = int(cv.h * 0.55)
    font = S.font(font_path, size * cv.supersample)
    col = colour or S.colour(rng, "contrast", cv.background)
    cv.text_colour = tuple(col)
    x0 = 0.1 * cv.w
    cov = cv.mask(lambda d, s: d.text((x0 * s, (cv.h - size) / 2 * s), text, fill=255, font=font))
    cv.blend(cov, col, "text", "text", "text", 1.0, dict(text=text))
    ys, xs = np.nonzero(cov >= 0.5)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())) if len(xs) else None


def sample_spec(rng, p_on=None, levels=range(1, 11),
                exclusive=(("gradient", "texture", "pattern"), ("wave", "morph", "perspective", "swirl"),
                           ("hollow", "bold", "thin"), ("pixelate", "resample"), ("lines", "bands"),
                           ("scribbles", "scribbles2"))):
    """Balanced random spec. p_on: {effect: probability} (default 0.3 each). Mutually exclusive groups keep at most one."""
    p_on = p_on or {}
    chosen = [n for n in SAMPLED if rng.random() < p_on.get(n, 0.15)]
    for grp in exclusive:
        hit = [n for n in chosen if n in grp]
        for n in hit[1:]:
            chosen.remove(n)
    return dict(background="random", effects=[dict(name=n, level=int(rng.choice(list(levels)))) for n in chosen])


EXCLUSIVE_K = (("gradient", "texture", "pattern"), ("wave", "morph", "perspective", "swirl"), ("hollow", "bold", "thin"),
               ("blur", "pixelate", "resample"), ("lines", "bands"), ("scribbles", "scribbles2"))


def sample_spec_k(rng, k, weights=None, levels=range(1, 11), exclusive=EXCLUSIVE_K):
    """Exactly k effects, drawn without replacement with probability proportional to `weights` ({effect: w}, default
    1), never two from the same exclusive group (at most one of blur / pixelate / low resolution). Level uniform."""
    names = list(SAMPLED)
    w = np.array([float((weights or {}).get(n, 1.0)) for n in names])
    chosen = []
    while len(chosen) < k and w.sum() > 0:
        i = int(rng.choice(len(names), p=w / w.sum()))
        n = names[i]
        chosen.append(n)
        for grp in exclusive:
            if n in grp:
                for m in grp:
                    w[names.index(m)] = 0.0
        w[i] = 0.0
    return dict(background="random", effects=[dict(name=n, level=int(rng.choice(list(levels)))) for n in chosen])
