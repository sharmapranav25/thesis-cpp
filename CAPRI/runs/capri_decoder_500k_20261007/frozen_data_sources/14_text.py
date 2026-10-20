"""Text renderer: draws CAPTCHA text glyph by glyph from real font files, with exact per-glyph labels.

Each character is rasterised on its own (supersampled), transformed (x-scale, slant/shear, rotation), and composited
onto the canvas as its own labelled op (head "text"). So for every glyph we know: the character, its pixel mask and box,
the font (key, family, Google Fonts category, stroke class, split), the variable-font axes used, size, rotation, slant,
x-scale, colour; plus measured stroke width. Text-level properties (spacing, overlap, rotation spread, jitter, size
variation, baseline curve, two-layer, font mixing, colours) are measured from the placed glyphs.

    spec = sample_text_spec(rng, ...)            # or write one by hand
    res = draw_text_spec(cv, spec, rng, bank)    # -> dict(text, glyphs=[...], text_box, ids, measures)

FontBank reads fonts/font_meta.json (tools/build_font_meta.py). Fonts are OFL; split "train" / "heldout" is kept.
"""
import json
import math
import string
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from captcha_effects import styles as S

DIGITS, UPPER, LOWER = string.digits, string.ascii_uppercase, string.ascii_lowercase
CHARSETS = dict(digits=DIGITS, upper=UPPER, lower=LOWER, upper_digits=UPPER + DIGITS, lower_digits=LOWER + DIGITS,
                mixed=UPPER + LOWER + DIGITS, letters=UPPER + LOWER)
CONFUSABLE_FREE = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"
LAYOUTS = ("line", "curve", "two_layer")
VOWELS = "aeiouAEIOU"


# ------------------------------------------------------------------ fonts

class FontBank:
    def __init__(self, meta_path=None):
        meta_path = Path(meta_path or Path(__file__).resolve().parents[1] / "fonts" / "font_meta.json")
        self.fonts = json.loads(meta_path.read_text(encoding="utf-8"))
        self.by_key = {f["key"]: f for f in self.fonts}

    def pool(self, split="train", category=None, caseless=False):
        return [f for f in self.fonts if f["split"] == split and (category is None or f["category"] == category)
                and (caseless or not f["caps_only"])]


@lru_cache(maxsize=512)
def _font(path, size_px):
    return ImageFont.truetype(path, int(size_px))


def _font_with_axes(meta, size_px, weight=None, width=None):
    f = ImageFont.truetype(meta["path"], int(size_px))  # fresh object: variation state is per object
    if meta["axes"] and (weight is not None or width is not None):
        vals = []
        for ax in meta["axes"]:
            if ax["tag"] == "wght" and weight is not None:
                vals.append(float(np.clip(weight, ax["min"], ax["max"])))
            elif ax["tag"] == "wdth" and width is not None:
                vals.append(float(np.clip(width, ax["min"], ax["max"])))
            else:
                vals.append(ax["default"])
        try:
            f.set_variation_by_axes(vals)
        except Exception:  # noqa: BLE001 - fall back to the default instance
            pass
    return f


# ------------------------------------------------------------------ one glyph

def glyph_mask(meta, ch, size_px, ss=4, weight=None, width_axis=None, stroke=0, x_scale=1.0, slant=0.0, angle=0.0):
    """Supersampled glyph -> transformed -> downsampled coverage mask (float 0..1) and its anchor offset."""
    f = _font_with_axes(meta, size_px * ss, weight, width_axis)
    l, t, r, b = f.getbbox(ch, stroke_width=stroke * ss)
    pad = 4 * ss + int(size_px * ss * 0.3)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    im = Image.new("L", (max(1, w), max(1, h)), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), ch, font=f, fill=255, stroke_width=stroke * ss, stroke_fill=255)
    if x_scale != 1.0 or slant != 0.0:
        # output x = sx * (x_in - w/2) + slant * (y - h/2) + nw/2  (horizontal scale about the centre, then shear)
        # inverse:  x_in = (x_out - nw/2 - slant * (y - h/2)) / sx + w/2
        sx = max(0.3, x_scale)
        nw = int(math.ceil(w * sx + abs(slant) * h)) + 2
        a, b = 1 / sx, -slant / sx
        c = (-nw / 2 + slant * h / 2) / sx + w / 2
        im = im.transform((nw, h), Image.AFFINE, (a, b, c, 0, 1, 0), resample=Image.BICUBIC)
    if angle:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    W, H = max(1, round(im.width / ss)), max(1, round(im.height / ss))
    m = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255.0
    ys, xs = np.nonzero(m > 0.05)
    if len(xs) == 0:
        return None
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return m


def stroke_width_px(mask):
    """Mean stroke width of a glyph mask: 2 * area / perimeter (perimeter = boundary pixels)."""
    b = mask >= 0.5
    if b.sum() == 0:
        return 0.0
    pad = np.pad(b, 1)
    inner = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    per = (b & ~inner).sum()
    return float(2 * b.sum() / max(1, per))


# ------------------------------------------------------------------ text spec

def make_text(rng, length, charset="mixed", pronounceable=False, caps_only=False):
    alpha = CHARSETS[charset]
    if caps_only:
        alpha = "".join(sorted(set(c.upper() if c.isalpha() else c for c in alpha)))
    if pronounceable:
        vs = [c for c in alpha if c in VOWELS]
        cs = [c for c in alpha if c.isalpha() and c not in VOWELS]
        if vs and cs:
            start = bool(rng.random() < 0.5)
            return "".join(str(rng.choice(cs if (i % 2 == 0) == start else vs)) for i in range(length))
    return "".join(str(rng.choice(list(alpha))) for _ in range(length))


def sample_text_spec(rng, bank, split="train", length=None, charset=None, layout=None, font_mix=None):
    """A random text spec covering the full range of text features."""
    n = int(length or rng.integers(2, 9))
    charset = charset or str(rng.choice(list(CHARSETS), p=[0.12, 0.14, 0.14, 0.12, 0.12, 0.28, 0.08]))
    layout = layout or str(rng.choice(LAYOUTS, p=[0.8, 0.12, 0.08]))
    mix = font_mix if font_mix is not None else int(rng.choice([1, 1, 1, 2, 3]))
    pool = bank.pool(split, caseless=charset in ("upper", "upper_digits", "digits"))
    fonts = [f["key"] for f in rng.choice(pool, size=min(mix, len(pool)), replace=False)]
    caps = all(bank.by_key[k]["caps_only"] for k in fonts)
    text = make_text(rng, n, charset, pronounceable=bool(rng.random() < 0.1), caps_only=caps)
    spec = dict(
        text=text, charset=charset, layout=layout, fonts=fonts,
        size=float(rng.uniform(0.4, 0.75)),                            # glyph em size / canvas height
        size_jitter=0.0 if rng.random() < 0.5 else float(rng.uniform(0.05, 0.45)),
        weight=float(rng.uniform(0, 1)) if rng.random() < 0.7 else None,  # 0..1 of the variable wght axis (if any)
        bold_stroke=int(rng.choice([0, 0, 0, 1, 2])),                 # synthetic emboldening (px)
        x_scale=1.0 if rng.random() < 0.6 else float(rng.uniform(0.6, 1.45)),
        slant=0.0 if rng.random() < 0.6 else float(rng.uniform(-0.4, 0.4)),
        rotation=0.0 if rng.random() < 0.4 else float(rng.uniform(3, 45)),   # max |angle| per glyph (deg)
        jitter=0.0 if rng.random() < 0.5 else float(rng.uniform(0.02, 0.25)), # vertical jitter / canvas height
        spacing=float(rng.choice([rng.uniform(0.0, 0.12), rng.uniform(-0.35, 0.0), rng.uniform(0.15, 1.0)],
                                 p=[0.5, 0.25, 0.25])),                     # gap / glyph height (negative = overlap)
        curve=float(rng.uniform(0.03, 0.2)) if layout == "curve" else 0.0,
        multicolour=bool(rng.random() < 0.3), palette=str(rng.choice(["contrast", "dark", "saturated", "random", "grey"])))
    return spec


FONT_CATEGORIES = ("Sans Serif", "Serif", "Display", "Handwriting", "Monospace")


def weight_target(level):
    """weight level 0..10 -> stroke_rel (inverse of font_heads)."""
    return 0.03 + 0.025 * level


def probe_weight(meta, size_px, weight_frac=None, stroke=0, chars="Hno"):
    """stroke_rel as the labels measure it (stroke width / the glyph's own height, mean over glyphs) for one font
    setting, on the characters that will be drawn."""
    wa = [a for a in meta["axes"] if a["tag"] == "wght"]
    weight = None if weight_frac is None or not wa else wa[0]["min"] + weight_frac * (wa[0]["max"] - wa[0]["min"])
    ms = [glyph_mask(meta, c, size_px, 2, weight=weight, stroke=stroke) for c in chars]
    ms = [m for m in ms if m is not None]
    if not ms:
        return None
    return float(np.mean([stroke_width_px(m) / max(1.0, m.shape[0]) for m in ms]))


def fit_weight(meta, size_px, target, chars="Hno"):
    """Font setting (weight_frac, bold_stroke) whose measured stroke_rel is closest to `target` -> (frac, stroke, err)."""
    wa = [a for a in meta["axes"] if a["tag"] == "wght"]
    cands = []
    if wa:
        lo, hi = probe_weight(meta, size_px, 0.0, 0, chars), probe_weight(meta, size_px, 1.0, 0, chars)
        if lo is None or hi is None:
            return None
        cands += [(0.0, 0, lo), (1.0, 0, hi)]
        if lo < target < hi and hi > lo:
            f = (target - lo) / (hi - lo)
            v = probe_weight(meta, size_px, f, 0, chars)
            if v is not None:
                cands.append((f, 0, v))
        if target > hi:
            for st in (1, 2):
                v = probe_weight(meta, size_px, 1.0, st, chars)
                if v is not None:
                    cands.append((1.0, st, v))
    else:
        for st in (0, 1, 2):
            v = probe_weight(meta, size_px, None, st, chars)
            if v is not None:
                cands.append((None, st, v))
            if v is not None and v >= target:
                break
    if not cands:
        return None
    f, st, v = min(cands, key=lambda c: abs(c[2] - target))
    return f, st, abs(v - target)


def x_height_level(f):
    return None if f.get("x_height") is None else int(np.clip(round((f["x_height"] - 0.45) / 0.055), 0, 10))


def balanced_text_spec(rng, bank, canvas_h, split="train", tries=5, **kw):
    """sample_text_spec, then the font side is drawn to uniform targets (same attention as the effects):
    category uniform over FONT_CATEGORIES, weight level 0..10 uniform (font, wght axis and synthetic bold chosen by
    probing), width level 0..10 uniform (x-scale solved per font), slant level 0..10 uniform, italic in ~25%.
    Targets are stored in spec['font_targets']; labels are still measured from the drawn glyphs."""
    spec = sample_text_spec(rng, bank, split=split, **kw)
    cat = str(rng.choice(FONT_CATEGORIES))
    t_w, t_wd, t_sl = int(rng.integers(0, 11)), int(rng.integers(0, 11)), int(rng.integers(0, 11))
    want_italic = bool(rng.random() < 0.25)
    caseless = spec["charset"] in ("upper", "upper_digits", "digits")
    pool = bank.pool(split, category=cat, caseless=caseless)
    ital = [f for f in pool if f.get("italic")]
    if want_italic and ital:
        pool = ital
    elif not want_italic:
        pool = [f for f in pool if not f.get("italic")] or pool
    # x-height: uniform over the levels with >= 20 fonts in the table (intrinsic to the font; rarer levels would repeat
    # a handful of fonts); candidates near the target level, falling back to the whole category pool
    if not hasattr(bank, "_xh_levels"):
        from collections import Counter
        cnt = Counter(x_height_level(f) for f in bank.fonts if f["split"] == "train")
        bank._xh_levels = sorted(k for k, v in cnt.items() if k is not None and v >= 20)
    t_xh = int(rng.choice(bank._xh_levels))
    near = [f for f in pool if x_height_level(f) is not None and abs(x_height_level(f) - t_xh) <= 0]
    pool = near if len(near) >= 3 else [f for f in pool if x_height_level(f) is not None and abs(x_height_level(f) - t_xh) <= 1] or pool
    size_px = spec["size"] * canvas_h
    chars = spec["text"][:3]
    wr = 0.38 + 0.077 * t_wd
    best = None
    for f in rng.choice(pool, size=min(tries, len(pool)), replace=False):
        xs = wr / (f.get("width_rel") or 0.8)
        x_err = abs(float(np.clip(xs, 0.6, 1.45)) - xs)
        fw = fit_weight(f, size_px, weight_target(t_w), chars)
        if fw is None:
            continue
        err = fw[2] / 0.025 + x_err / 0.1  # in weight levels + x-scale tenths
        if best is None or err < best[0]:
            best = (err, f, fw, float(np.clip(xs, 0.6, 1.45)))
    if best is None:
        return spec
    _, f, (frac, stroke, _), xs = best
    fonts = [f["key"]]
    if len(spec["fonts"]) > 1:  # font mixture: more fonts of the same category
        others = [g["key"] for g in bank.pool(split, category=cat, caseless=caseless) if g["key"] != f["key"]]
        fonts += [str(k) for k in rng.choice(others, size=min(len(spec["fonts"]) - 1, len(others)), replace=False)]
    caps = all(bank.by_key[k]["caps_only"] for k in fonts)
    if caps and any(c.islower() for c in spec["text"]):
        spec["text"] = spec["text"].upper()
    slant = 0.0 if t_sl == 0 else float(rng.choice([-1, 1])) * float(rng.uniform(t_sl - 0.5, t_sl + 0.5)) * 0.04
    spec.update(fonts=fonts, weight=frac, bold_stroke=int(stroke), x_scale=xs, slant=slant,
                font_targets=dict(category=cat, weight=t_w, width=t_wd, slant=t_sl, italic=want_italic, x_height=t_xh))
    return spec


# ------------------------------------------------------------------ drawing

def draw_text_spec(cv, spec, rng, bank, ss=4, colour=None):
    """Draw spec['text'] glyph by glyph onto the canvas. Returns per-glyph labels and measured text properties."""
    text = spec["text"]
    n = len(text)
    fonts = spec["fonts"]
    base_px = spec["size"] * cv.h
    min_dl = float(spec.get("min_contrast", 60.0))  # text stays distinguishable from the background (luminance)

    def text_colour():
        c = S.colour(rng, spec.get("palette", "contrast"), cv.background)
        if abs(S.lum(c) - S.lum(cv.background)) < min_dl:
            c = S.with_lum_distance(c, cv.background, float(rng.uniform(min_dl, 180.0)))
        return c
    col0 = colour or text_colour()
    cv.text_colour = tuple(col0)
    glyphs = []
    for i, ch in enumerate(text):
        meta = bank.by_key[fonts[i % len(fonts)] if len(fonts) > 1 else fonts[0]]
        if spec.get("font_assign") == "random" and len(fonts) > 1:
            meta = bank.by_key[str(rng.choice(fonts))]
        size = base_px * (1 + spec["size_jitter"] * float(rng.uniform(-1, 1)))
        weight = None
        if spec.get("weight") is not None:
            wa = [a for a in meta["axes"] if a["tag"] == "wght"]
            if wa:
                weight = wa[0]["min"] + spec["weight"] * (wa[0]["max"] - wa[0]["min"])
        angle = float(rng.uniform(-1, 1)) * spec["rotation"]
        m = glyph_mask(meta, ch, size, ss, weight=weight, stroke=spec.get("bold_stroke", 0), x_scale=spec["x_scale"],
                       slant=spec["slant"], angle=angle)
        if m is None:
            continue
        glyphs.append(dict(index=i, char=ch, mask=m, font=meta["key"], family=meta["family"], category=meta["category"],
                           stroke_class=meta["stroke"], split=meta["split"], italic=bool(meta.get("italic", False)),
                           x_height_font=meta.get("x_height"), width_font=meta.get("width_rel"),
                           size_px=round(size, 2), angle=round(angle, 2),
                           weight_axis=None if weight is None else round(weight, 1), stroke=spec.get("bold_stroke", 0),
                           x_scale=spec["x_scale"], slant=spec["slant"]))
    if not glyphs:
        return None
    # ---- layout: positions
    hs = [g["mask"].shape[0] for g in glyphs]
    ref_h = float(np.median(hs))
    rows = [glyphs]
    if spec["layout"] == "two_layer" and len(glyphs) >= 4:
        k = (len(glyphs) + 1) // 2
        rows = [glyphs[:k], glyphs[k:]]
    row_widths = []
    for row in rows:
        wsum = sum(g["mask"].shape[1] for g in row) + spec["spacing"] * ref_h * (len(row) - 1)
        row_widths.append(wsum)
    scale = min(1.0, 0.92 * cv.w / max(row_widths), 0.92 * cv.h / (ref_h * (len(rows) + 0.3 * (len(rows) - 1))))
    if scale < 1.0:  # too big for the canvas: shrink every glyph
        for g in glyphs:
            m = g["mask"]
            nh, nw = max(1, int(round(m.shape[0] * scale))), max(1, int(round(m.shape[1] * scale)))
            g["mask"] = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((nw, nh), Image.BOX), np.float32) / 255
            g["size_px"] = round(g["size_px"] * scale, 2)
        ref_h *= scale
        row_widths = [w * scale for w in row_widths]
    total_h = ref_h * len(rows) + (0.15 * ref_h if len(rows) > 1 else 0)
    y_top = (cv.h - total_h) / 2 + float(rng.uniform(-0.1, 0.1)) * cv.h * (1 - total_h / cv.h)
    ids = []
    for r_i, row in enumerate(rows):
        x = (cv.w - row_widths[r_i]) / 2 + float(rng.uniform(-0.3, 0.3)) * max(0.0, cv.w - row_widths[r_i]) / 2
        if len(rows) > 1:
            x += (r_i - 0.5) * 0.25 * ref_h  # stagger the two layers so they interlock
        base_y = y_top + r_i * (ref_h * 0.85)
        ph = float(rng.uniform(0, 2 * math.pi))
        per = float(rng.uniform(0.6, 1.4)) * cv.w
        for g in row:
            m = g["mask"]
            h, w = m.shape
            cx = x + w / 2
            g["jitter_off"] = spec["jitter"] * cv.h * float(rng.uniform(-1, 1))
            g["curve_off"] = spec["curve"] * cv.h * math.sin(2 * math.pi * cx / per + ph)
            y = base_y + (ref_h - h) / 2 + g["jitter_off"] + g["curve_off"]
            y = float(np.clip(y, 0, max(0, cv.h - h)))  # glyphs stay inside the canvas vertically
            g["x0"], g["y0"] = x, y
            x += w + spec["spacing"] * ref_h
            col = text_colour() if spec.get("multicolour") else col0
            g["colour"] = [int(c) for c in col]
            cov = np.zeros((cv.h, cv.w), np.float32)
            xi, yi = int(round(g["x0"])), int(round(g["y0"]))
            ys, xs = max(0, yi), max(0, xi)
            ye, xe = min(cv.h, yi + h), min(cv.w, xi + w)
            if ye > ys and xe > xs:
                cov[ys:ye, xs:xe] = m[ys - yi:ye - yi, xs - xi:xe - xi]
            g["clipped_fraction"] = round(1.0 - float(cov.sum()) / max(1e-6, float(m.sum())), 4)
            g["stroke_width_px"] = round(stroke_width_px(m), 3)
            g["stroke_rel"] = round(g["stroke_width_px"] / max(1.0, h), 4)
            params = {k: v for k, v in g.items() if k != "mask"}
            g["id"] = cv.blend(cov, col, "text", "text", "text", 1.0, params)
            ids.append(g["id"])
    # ---- final per-glyph boxes from the label map (after later glyphs may overlap earlier ones)
    for g in glyphs:
        if "id" not in g:
            continue
        ys, xs = np.nonzero(cv.opaque == g["id"])
        g["box"] = None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        g["visible_px"] = int(len(xs))
    placed = [g for g in glyphs if g.get("box")]
    tb = (min(g["box"][0] for g in placed), min(g["box"][1] for g in placed), max(g["box"][2] for g in placed),
          max(g["box"][3] for g in placed)) if placed else None
    return dict(text="".join(g["char"] for g in glyphs if "id" in g), spec=spec, ids=ids, text_box=tb,
                glyphs=[{k: v for k, v in g.items() if k != "mask"} for g in glyphs if "id" in g],
                measures=measure_text(glyphs, cv, spec))


def measure_text(glyphs, cv, spec):
    """Text-level properties measured from the placed glyphs (not copied from the spec)."""
    g = [x for x in glyphs if "x0" in x]
    if not g:
        return {}
    hs = np.array([x["mask"].shape[0] for x in g], float)
    ref = float(np.median(hs))
    boxes = [(x["x0"], x["y0"], x["x0"] + x["mask"].shape[1], x["y0"] + x["mask"].shape[0]) for x in g]
    gaps, overl = [], []
    row_break = (len(g) + 1) // 2 - 1 if spec.get("layout") == "two_layer" and len(g) >= 4 else -1
    for pair_index, (a, b) in enumerate(zip(boxes[:-1], boxes[1:])):
        if pair_index == row_break:
            continue  # the last glyph of row 1 is not adjacent to the first glyph of row 2
        gaps.append((b[0] - a[2]) / ref)
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        overl.append(iw * ih / max(1.0, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))))
    cy = np.array([(b[1] + b[3]) / 2 for b in boxes])
    xs = np.array([(b[0] + b[2]) / 2 for b in boxes])
    # geometric offsets actually applied to the placed glyphs (exact; positions are clamped to the canvas first)
    curve = float(np.ptp([x["curve_off"] for x in g]) / ref) if len(g) > 1 else 0.0
    jit = float(np.std([x["jitter_off"] for x in g]) / ref) if len(g) > 1 else 0.0
    sizes = np.array([x["size_px"] for x in g], float)
    return dict(length=len(g), curve=round(curve, 4), mean_gap=round(float(np.mean(gaps)), 4) if gaps else 0.0,
                slant=round(float(np.mean([abs(x["slant"]) for x in g])), 4),
                x_scale=round(float(np.mean([x["x_scale"] for x in g])), 4),
                overlap=round(float(np.mean(overl)), 4) if overl else 0.0,
                rotation_mean_abs=round(float(np.mean([abs(x["angle"]) for x in g])), 3),
                jitter=round(jit, 4),
                size_variation=round(float(np.std(sizes) / max(1e-6, np.mean(sizes))), 4),
                fonts=len({x["font"] for x in g}), colours=len({tuple(x["colour"]) for x in g}),
                layout=spec["layout"], max_clipped=round(max(x["clipped_fraction"] for x in g), 4),
                stroke_rel=round(float(np.mean([x["stroke_rel"] for x in g])), 4))


# ------------------------------------------------------------------ labels (heads) for the text side

# measured value -> 0..10 (value at level 10 in brackets); the measures are taken from the placed glyphs
TEXT_SCALES = dict(per_char_rotation=("rotation_mean_abs", 30.0), vertical_jitter=("jitter", 0.3),
                   size_variation=("size_variation", 0.25), text_on_curve=("curve", 0.8), slant=("slant", 0.4),
                   overlap_tight_spacing=("overlap", 0.45), wide_spacing=("mean_gap", 1.0))
WEIGHT_CLASSES = [(0.09, "thin"), (0.14, "regular"), (0.2, "bold"), (9.9, "heavy")]


def weight_class(stroke_rel):
    return next(name for lim, name in WEIGHT_CLASSES if stroke_rel < lim)


def font_heads(g):
    """Per-glyph font features on 0..10 scales.
    Ranges = the 2nd-98th percentile over the 2,185 fonts of the table:
    weight: measured stroke width / glyph height (0.03 -> 0 ... 0.28 -> 10), includes synthetic emboldening;
    width: the font's own n-advance / cap height times the applied x-scale (0.38 -> 0 ... 1.15 -> 10);
    x_height: the font's x-height / cap height (0.45 -> 0 ... 1.0 -> 10);
    slant_level: |applied shear| (0.4 -> 10); italic: the font is a true italic."""
    w = float(np.clip((g["stroke_rel"] - 0.03) / 0.025, 0, 10))
    wr = (g.get("width_font") or 0.8) * g["x_scale"]
    xh = g.get("x_height_font")
    return dict(weight=int(round(w)), width=int(np.clip(round((wr - 0.38) / 0.077), 0, 10)),
                x_height=None if xh is None else int(np.clip(round((xh - 0.45) / 0.055), 0, 10)),
                slant_level=int(np.clip(round(abs(g["slant"]) / 0.04), 0, 10)), italic=bool(g.get("italic", False)))


def text_labels(res):
    """Head labels for one drawn text: presence + 0-10 level for the text-geometry heads, content heads, style classes,
    and per-glyph slot labels (char, group, font, category, stroke class, weight class, box, angle, colour)."""
    m, spec = res["measures"], res["spec"]
    t = res["text"]
    heads = {}
    for h, (key, top) in TEXT_SCALES.items():
        v = max(0.0, float(m.get(key, 0.0)))
        if h == "overlap_tight_spacing":
            v = max(v, max(0.0, -float(m.get("mean_gap", 0.0))) * 0.45)  # gaps below zero are crowding too
        lv = int(np.clip(round(10 * v / top), 0, 10))
        heads[h] = dict(present=int(lv > 0), level=lv, value=round(v, 4))
    xs = float(m.get("x_scale", 1.0))
    heads["condensed"] = dict(present=int(xs < 0.9), level=int(np.clip(round(10 * (1 - xs) / 0.35), 0, 10)) if xs < 1 else 0, value=xs)
    heads["expanded"] = dict(present=int(xs > 1.1), level=int(np.clip(round(10 * (xs - 1) / 0.4), 0, 10)) if xs > 1 else 0, value=xs)
    heads["font_mixture"] = dict(present=int(m.get("fonts", 1) > 1), level=None, value=m.get("fonts", 1))
    heads["multicolor_text"] = dict(present=int(m.get("colours", 1) > 1), level=None, value=m.get("colours", 1))
    heads["two_layer"] = dict(present=int(m.get("layout") == "two_layer"), level=None, value=None)
    d, u, lo = any(c.isdigit() for c in t), any(c.isupper() for c in t), any(c.islower() for c in t)
    heads["digits_only"] = dict(present=int(d and not u and not lo), level=None)
    heads["uppercase_only"] = dict(present=int(u and not lo), level=None)
    heads["lowercase_only"] = dict(present=int(lo and not u), level=None)
    heads["mixed_case_alnum"] = dict(present=int(u and lo), level=None)
    heads["length_short_le4"] = dict(present=int(len(t) <= 4), level=None)
    heads["length_long_ge7"] = dict(present=int(len(t) >= 7), level=None)
    cats = sorted({g["category"] for g in res["glyphs"]})
    style = dict(layout=m.get("layout"), charset=spec.get("charset"), categories=cats,
                 weight_class=weight_class(float(m.get("stroke_rel", 0.12))))
    slots = [dict(char=g["char"], group="digit" if g["char"].isdigit() else "upper" if g["char"].isupper() else "lower",
                  font=g["font"], family=g["family"], category=g["category"], stroke_class=g["stroke_class"],
                  weight_class=weight_class(g["stroke_rel"]), **font_heads(g), box=g["box"], angle=g["angle"],
                  colour=g["colour"], size_px=g["size_px"]) for g in res["glyphs"]]
    if slots:  # per-image font summary (what the controller uses to pick fonts for a target family)
        from collections import Counter
        style["dominant_category"] = Counter(s_["category"] for s_ in slots).most_common(1)[0][0]
        style["dominant_stroke_class"] = Counter(s_["stroke_class"] for s_ in slots).most_common(1)[0][0]
        for k in ("weight", "width", "x_height", "slant_level"):
            style[f"median_{k}"] = int(np.median([s_[k] for s_ in slots]))
        style["italic_any"] = any(s_["italic"] for s_ in slots)
    return dict(text=t, length=len(t), heads=heads, style=style, slots=slots,
                usable=bool(m.get("max_clipped", 0) <= 0.2))
