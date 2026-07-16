"""Side-by-side bg comparison panel: original / CC-mask bg / SAM-mask bg.

Per-scheme panel saved to notebooks/<scheme>_bg_compare.png.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw


SCHEMES = [
    {"name": "courier_market", "real": Path("E:/pranav/Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)"), "canvas": (200, 60)},
    {"name": "city", "real": Path("E:/pranav/City (live)-20260503T045450Z-3-001/City (live)"), "canvas": (324, 81)},
    {"name": "cryptbb_unsolved", "real": Path("E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023"), "canvas": (200, 60)},
    {"name": "joey_retitling", "real": Path("E:/pranav/JoeyRetitling_extracted/JoeyRetitling"), "canvas": (200, 60)},
]

DATA_ROOT = Path("E:/pranav/auto_synth/data")
NB_ROOT = Path("E:/pranav/auto_synth/notebooks")
N_PER_ROW = 6


def _png_files(d: Path):
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def make_panel(real_dir: Path, cc_bg_dir: Path, sam_bg_dir: Path,
               out: Path, canvas: tuple[int, int],
               n_per_row: int = N_PER_ROW, seed: int = 0):
    rng = random.Random(seed)
    real_files = _png_files(real_dir)
    cc_files = _png_files(cc_bg_dir)
    sam_files = _png_files(sam_bg_dir)
    if not real_files or not sam_files:
        print(f"[bg_compare] missing inputs at {real_dir} / {sam_bg_dir}")
        return

    # Match by filename so we see same image processed by both methods
    cc_by_name = {f.name: f for f in cc_files}
    sam_by_name = {f.name: f for f in sam_files}
    common = sorted(set(cc_by_name) & set(sam_by_name))
    if len(common) < n_per_row:
        print(f"[bg_compare] only {len(common)} common files")
    pick_names = rng.sample(common, min(n_per_row, len(common)))

    real_by_name = {f.name: f for f in real_files}

    W, H = canvas
    pad = 6
    label_h = 22
    panel_w = n_per_row * (W + pad) + pad
    panel_h = (label_h + (H + pad)) * 3 + pad
    panel = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
    d = ImageDraw.Draw(panel)

    rows = [
        ("REAL", [real_by_name.get(n) for n in pick_names]),
        ("CC-mask + LaMa bg", [cc_by_name.get(n) for n in pick_names]),
        ("SAM-mask + LaMa bg", [sam_by_name.get(n) for n in pick_names]),
    ]
    for row_idx, (label, files) in enumerate(rows):
        y0 = row_idx * (label_h + H + pad) + pad
        d.text((pad, y0), label, fill=(0, 0, 0))
        for i, f in enumerate(files):
            if f is None:
                continue
            try:
                im = Image.open(f).convert("RGB").resize(canvas, Image.LANCZOS)
                x = pad + i * (W + pad)
                y = y0 + label_h
                panel.paste(im, (x, y))
            except Exception:
                pass

    panel.save(out)
    print(f"[bg_compare] saved {out}")


def main():
    for sc in SCHEMES:
        cc_dir = DATA_ROOT / sc["name"] / "bg_lama"
        sam_dir = DATA_ROOT / sc["name"] / "bg_lama_sam"
        out = NB_ROOT / f"{sc['name']}_bg_compare.png"
        make_panel(sc["real"], cc_dir, sam_dir, out, sc["canvas"])


if __name__ == "__main__":
    main()
