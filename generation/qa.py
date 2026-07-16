"""QA / verification (pipeline step 6) — the recognizer-free "does it work" check.

1. panel(): a visual A/B grid, REAL (top block) vs SYNTH (bottom block), per scheme.
2. discriminate(): the VLM DISCRIMINATOR — show the VLM shuffled real/synth pairs and
   ask which is the real captcha. Accuracy near 50% => synth is convincing (a
   recognizer-free Turing-style test; the only success metric, since we train nothing).
"""
from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

from PIL import Image, ImageDraw

import pipeline

HERE = Path(__file__).resolve().parent


def _reals(scheme, k, rng):
    ps = pipeline._list_images(pipeline.SCHEMES[scheme]["dir"])
    rng.shuffle(ps)
    return ps[:k]


def _synths(scheme, k, rng, out_dir=None):
    d = Path(out_dir) if out_dir else HERE / "out" / scheme / "images"
    ps = sorted(glob.glob(str(d / "*.png")))
    rng.shuffle(ps)
    return ps[:k]


def panel(scheme, out=None, cols=8, rng=None, out_dir=None):
    rng = rng or random.Random(0)
    reals = _reals(scheme, cols, rng)
    synth = _synths(scheme, cols, rng, out_dir)
    if not reals or not synth:
        print(f"[{scheme}] missing images (real={len(reals)} synth={len(synth)})"); return None
    cw, ch = 200, 60
    pad, hdr = 6, 18
    W = cols * (cw + pad) + pad
    H = 2 * (ch + pad) + 2 * hdr + pad
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)

    def row(paths, y0):
        for i, p in enumerate(paths):
            try:
                im = Image.open(p).convert("RGB").resize((cw, ch))
            except Exception:
                im = Image.new("RGB", (cw, ch), "gray")
            canvas.paste(im, (pad + i * (cw + pad), y0))

    d.text((pad, 2), f"{scheme}  —  REAL", fill="black")
    row(reals, hdr)
    d.text((pad, hdr + ch + pad + 2), f"{scheme}  —  SYNTH", fill="black")
    row(synth, 2 * hdr + ch + pad)
    out = Path(out) if out else HERE / "out" / f"qa_{scheme}.png"
    canvas.save(out)
    print(f"[{scheme}] panel -> {out}")
    return out


def discriminate(scheme, trials=20, rng=None, out_dir=None):
    """VLM discriminator accuracy: 0.5 == synth indistinguishable from real."""
    import vlm
    rng = rng or random.Random(1)
    reals = _reals(scheme, trials, rng)
    synth = _synths(scheme, trials, rng, out_dir)
    n = min(len(reals), len(synth), trials)
    correct = 0
    instr = ("One of these two images is a REAL captcha, the other is a computer-"
             "generated FAKE. Which is the REAL one? Reply with exactly one character: "
             "1 or 2.")
    for i in range(n):
        real_first = rng.random() < 0.5
        a = reals[i] if real_first else synth[i]
        b = synth[i] if real_first else reals[i]
        out = vlm._generate(instr, [Image.open(a), Image.open(b)], max_new_tokens=4)
        pick_first = out.strip().startswith("1")
        guessed_real = a if pick_first else b
        if guessed_real == reals[i]:
            correct += 1
    acc = correct / n if n else 0.0
    print(f"[{scheme}] VLM discriminator: {correct}/{n} = {acc:.0%} "
          f"(50% = indistinguishable; >70% = synth still tells)")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=list(pipeline.SCHEMES))
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--discriminate", action="store_true")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.panel or not args.discriminate:
        panel(args.scheme, out_dir=args.out_dir)
    if args.discriminate:
        discriminate(args.scheme, trials=args.trials, out_dir=args.out_dir)
