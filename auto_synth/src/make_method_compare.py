"""Three-row comparison panel per scheme: REAL vs CC-discovery synth vs OCR-discovery synth.

Lets you eyeball which discovery method produced more realistic output.
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


def make_panel(real_dir: Path, cc_dir: Path, ocr_dir: Path, out: Path,
               canvas: tuple[int, int], n_per_row: int = N_PER_ROW, seed: int = 0):
    rng = random.Random(seed)
    real_files = _png_files(real_dir)
    cc_files = _png_files(cc_dir)
    ocr_files = _png_files(ocr_dir)

    if not real_files:
        print(f"[compare3] no real images at {real_dir}")
        return

    real_pick = rng.sample(real_files, min(n_per_row, len(real_files)))
    cc_pick = rng.sample(cc_files, min(n_per_row, len(cc_files))) if cc_files else []
    ocr_pick = rng.sample(ocr_files, min(n_per_row, len(ocr_files))) if ocr_files else []

    W, H = canvas
    pad = 6
    label_h = 22
    panel_w = n_per_row * (W + pad) + pad
    panel_h = (label_h + (H + pad)) * 3 + pad
    panel = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
    d = ImageDraw.Draw(panel)

    rows = [
        ("REAL", real_pick),
        ("CC-based synth", cc_pick),
        ("OCR-based synth", ocr_pick),
    ]
    for row_idx, (label, files) in enumerate(rows):
        y0 = row_idx * (label_h + H + pad) + pad
        d.text((pad, y0), label, fill=(0, 0, 0))
        for i, f in enumerate(files):
            try:
                im = Image.open(f).convert("RGB").resize(canvas, Image.LANCZOS)
                x = pad + i * (W + pad)
                y = y0 + label_h
                panel.paste(im, (x, y))
            except Exception:
                pass

    panel.save(out)
    print(f"[compare3] saved {out}")


def main():
    for sc in SCHEMES:
        cc_dir = DATA_ROOT / sc["name"] / "synth_pool"
        ocr_dir = DATA_ROOT / f"{sc['name']}_ocr" / "synth_pool"
        out = NB_ROOT / f"{sc['name']}_method_compare.png"
        make_panel(sc["real"], cc_dir, ocr_dir, out, sc["canvas"])


if __name__ == "__main__":
    main()
