"""Visual smoke test: run cleaner on N synth + N real images, save side-by-side
grids (raw | saliency | canny | mask | cleaned) so a human can verify chars
survive cleaning before we commit to bulk processing 100k images."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import Cleaner, CleanerConfig


def _to_rgb_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr)


def _row(samples: list[Image.Image], headers: list[str], pad: int = 4) -> Image.Image:
    """Lay out [hdr | img1 | img2 | ... | imgN] side by side."""
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    if not samples:
        return Image.new("RGB", (1, 1), (255, 255, 255))
    w, h = samples[0].size
    label_h = 16
    canvas = Image.new("RGB", (w * len(samples) + pad * (len(samples) + 1), h + label_h),
                       (235, 235, 235))
    d = ImageDraw.Draw(canvas)
    for i, (im, hdr) in enumerate(zip(samples, headers)):
        x = pad + i * (w + pad)
        d.text((x, 0), hdr, fill=(0, 0, 0), font=font)
        canvas.paste(im.resize((w, h)), (x, label_h))
    return canvas


def _stack(rows: list[Image.Image], pad: int = 4) -> Image.Image:
    if not rows:
        return Image.new("RGB", (1, 1), (255, 255, 255))
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows) + pad * (len(rows) - 1)
    canvas = Image.new("RGB", (w, h), (210, 210, 210))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + pad
    return canvas


def main(synth_dir: Path, real_dir: Path, n: int, out_path: Path, scale: int) -> None:
    cfg = CleanerConfig()
    cleaner = Cleaner(cfg)

    synth_paths = sorted(synth_dir.glob("*.png"))[:n]
    real_paths = sorted(p for p in real_dir.glob("*.png")
                        if not p.name.startswith("_"))[:n]

    rows: list[Image.Image] = []
    for label, paths in (("SYNTH", synth_paths), ("REAL", real_paths)):
        for p in paths:
            img = Image.open(p).convert("RGB")
            cleaned, dbg = cleaner.clean_pil(img)
            samples = [
                img,
                _to_rgb_pil(dbg["saturation"]),
                _to_rgb_pil(dbg["darkness"]),
                _to_rgb_pil(dbg["canny"]),
                _to_rgb_pil(dbg["mask"]),
                cleaned,
            ]
            if scale > 1:
                samples = [s.resize((s.width * scale, s.height * scale), Image.NEAREST)
                           for s in samples]
            row = _row(samples, [f"{label}: {p.name}", "sat", "dark", "canny", "mask", "cleaned"])
            rows.append(row)

    grid = _stack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"saved {out_path}  ({grid.size})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--real", type=Path, default=Path("data/raw"))
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("data/_cleaner_inspect/grid.png"))
    args = ap.parse_args()
    main(args.synth, args.real, args.n, args.out, args.scale)
