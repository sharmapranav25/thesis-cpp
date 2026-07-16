"""Make a side-by-side real-vs-synth comparison panel per scheme.

Layout: top row = 6 random real images, bottom row = 6 random synth images.
Saved to notebooks/<scheme>_compare.png.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image


SCHEMES = [
    {
        "name": "courier_market",
        "real": Path("E:/pranav/Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)"),
        "canvas": (200, 60),
    },
    {
        "name": "city",
        "real": Path("E:/pranav/City (live)-20260503T045450Z-3-001/City (live)"),
        "canvas": (324, 81),
    },
    {
        "name": "cryptbb_unsolved",
        "real": Path("E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023"),
        "canvas": (200, 60),
    },
    {
        "name": "joey_retitling",
        "real": Path("E:/pranav/JoeyRetitling_extracted/JoeyRetitling"),
        "canvas": (200, 60),
    },
]

DATA_ROOT = Path("E:/pranav/auto_synth/data")
NB_ROOT = Path("E:/pranav/auto_synth/notebooks")
N_PER_ROW = 6


def make_panel(real_dir: Path, synth_dir: Path, out: Path, canvas: tuple[int, int],
               n_per_row: int = N_PER_ROW, seed: int = 0):
    rng = random.Random(seed)
    real_files = sorted(p for p in real_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    synth_files = sorted(p for p in synth_dir.iterdir() if p.suffix.lower() == ".png")
    if not real_files or not synth_files:
        print(f"[compare] empty: real={len(real_files)} synth={len(synth_files)}")
        return
    real_pick = rng.sample(real_files, min(n_per_row, len(real_files)))
    synth_pick = rng.sample(synth_files, min(n_per_row, len(synth_files)))

    W, H = canvas
    pad = 6
    label_h = 22
    panel_w = n_per_row * (W + pad) + pad
    panel_h = label_h + (H + pad) * 2 + pad + label_h
    panel = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))

    # Labels
    from PIL import ImageDraw
    d = ImageDraw.Draw(panel)
    d.text((pad, 2), "REAL", fill=(0, 0, 0))
    d.text((pad, label_h + H + pad + 2), "SYNTH", fill=(0, 0, 0))

    # Real row
    for i, f in enumerate(real_pick):
        try:
            im = Image.open(f).convert("RGB").resize(canvas, Image.LANCZOS)
            x = pad + i * (W + pad)
            y = label_h
            panel.paste(im, (x, y))
        except Exception:
            pass

    # Synth row
    for i, f in enumerate(synth_pick):
        try:
            im = Image.open(f).convert("RGB").resize(canvas, Image.LANCZOS)
            x = pad + i * (W + pad)
            y = label_h + H + pad + label_h
            panel.paste(im, (x, y))
        except Exception:
            pass

    panel.save(out)
    print(f"[compare] saved {out}")


def main():
    for sc in SCHEMES:
        synth_dir = DATA_ROOT / sc["name"] / "synth_pool"
        out = NB_ROOT / f"{sc['name']}_compare.png"
        if not synth_dir.exists():
            print(f"[compare] skip {sc['name']}: no synth_pool")
            continue
        make_panel(sc["real"], synth_dir, out, sc["canvas"])


if __name__ == "__main__":
    main()
