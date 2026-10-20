"""0-10 intensity from the *visible* amount (never the requested settings).

For every effect, levels 0..10 are drawn on blank canvases over many seeds (with a line of text for over-text effects) and
the visible amount is measured:
  under/over layer effects : visible pixels of the effect's head / canvas pixels (from the label maps)
  fields (gradient/texture): luminance spread p95 - p5 of the drawn field
  post / glyph effects     : the measured amount recorded by the op (displacement px, blur sigma, noise sigma, JPEG
                             100-quality, block size, colours removed, changed fraction of glyph or canvas pixels)
Styles are random per draw, so the raw per-level medians need not be monotone. The scale is the isotonic (monotone
non-decreasing) fit of the medians, made strictly increasing; bin edges sit halfway between neighbouring fitted medians.
`level_of(table, effect, value)` = the bin a measured value falls in (0 when nothing is visible): a larger visible amount
never gets a lower level. An effect passes when level 0 measures 0, its knob drives the visible amount (Spearman rank
correlation between requested level and measured value >= 0.7) and the fitted scale spans >= 2x from level 1 to 10.

  python -m captcha_effects.calibrate --out calibration.json --seeds 24 --sheet sheets/
"""
import argparse
import json
from pathlib import Path

import numpy as np

from captcha_effects.canvas import Canvas
from captcha_effects.effects import REGISTRY
from captcha_effects.render import ALL_EFFECTS, draw_text

SIZES = [(200, 60), (240, 80), (324, 81), (200, 50), (300, 90)]


def isotonic(y):
    """Pool-adjacent-violators: monotone non-decreasing fit of y."""
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(0, i - 1)
        else:
            i += 1
    out = []
    for v, n in blocks:
        out += [v] * n
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def measure_effect(name, level, seed, size=(200, 60), font_path=None):
    from captcha_effects.measure import visible_amount
    rng = np.random.default_rng(seed)
    cv = Canvas(size[0], size[1], (255, 255, 255))
    fx = REGISTRY[name]
    tb = None
    if fx["where"] != "under":
        tb = draw_text(cv, "AbC3k", rng, font_path, colour=(0, 0, 0))
    kw = dict(text_box=tb) if fx["where"] == "over" else {}
    p = fx["fn"](cv, level, rng, **kw) if level else {}
    v = visible_amount(name, cv.records, cv.opaque, cv.translucent, cv.background, p.get("measured"))
    return (v if level else 0.0), cv


def calibrate(seeds=24, font_path=None):
    out = {}
    for name in ALL_EFFECTS:
        med, lv_all, val_all = [], [], []
        for level in range(11):
            vals = [measure_effect(name, level, 1000 * level + s, SIZES[s % len(SIZES)], font_path)[0] for s in range(seeds)]
            med.append(float(np.median(vals)))
            lv_all += [level] * len(vals)
            val_all += vals
        fit = isotonic(med[1:])
        span = max(fit[-1], 1e-9) * 1e-3
        fit = [max(v, fit[0] + span * k) if k else v for k, v in enumerate(fit)]
        for k in range(1, len(fit)):
            fit[k] = max(fit[k], fit[k - 1] + span)  # strictly increasing
        edges = [fit[0] / 2] + [(a + b) / 2 for a, b in zip(fit[:-1], fit[1:])]
        rho = spearman(np.array(lv_all[seeds:]), np.array(val_all[seeds:]))  # levels 1..10
        ok = med[0] == 0 and rho >= 0.7 and fit[-1] >= 2 * max(fit[0], 1e-9)
        out[name] = dict(head=REGISTRY[name]["head"], where=REGISTRY[name]["where"], raw_medians=med, fitted=[0.0] + fit,
                         edges=edges, spearman=round(rho, 3), passed=bool(ok))
    return out


def level_of(table, name, value):
    """Bin of a measured visible amount on the effect's monotone scale (0 = nothing visible)."""
    if value <= 0:
        return 0
    return int(max(1, np.searchsorted(table[name]["edges"], value, side="right")))


def sheet(path, name, font_path=None, levels=(0, 2, 5, 8, 10), styles=4):
    from PIL import Image, ImageDraw
    W, H = 240, 80
    S = Image.new("RGB", (len(levels) * (W + 6) + 80, styles * (H + 6) + 26), "white")
    d = ImageDraw.Draw(S)
    d.text((6, 6), name, fill="black")
    for j, lv in enumerate(levels):
        d.text((80 + j * (W + 6), 6), f"level {lv}", fill="black")
        for i in range(styles):
            _, cv = measure_effect(name, lv, 7000 + 97 * i + lv, (W, H), font_path)
            S.paste(Image.fromarray(cv.image()), (80 + j * (W + 6), 22 + i * (H + 6)))
    S.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--font")
    ap.add_argument("--sheet")
    a = ap.parse_args()
    t = calibrate(a.seeds, a.font)
    Path(a.out).write_text(json.dumps(t, indent=1), encoding="utf-8")
    for n, e in t.items():
        print(f"{n:12s} {'PASS' if e['passed'] else 'FAIL'} rho {e['spearman']:.2f} fitted "
              + " ".join(f"{v:.3g}" for v in e["fitted"]))
    if a.sheet:
        Path(a.sheet).mkdir(parents=True, exist_ok=True)
        for n in ALL_EFFECTS:
            sheet(Path(a.sheet) / f"{n}.png", n, a.font)


if __name__ == "__main__":
    main()
