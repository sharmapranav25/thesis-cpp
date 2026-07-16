"""Render synthetic CryptBB-style captchas with known text.

Produces 200x60 RGBA images mimicking the visual style of real CryptBB captchas:
white background, multi-color crosshatch grid, colored scribbles, per-character
random font/color/rotation. Labels are written to labels.csv.

The synthetic images need only be *structurally* similar to real captchas; the
CycleGAN in Phase 2 closes the remaining style gap.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


CANVAS_W, CANVAS_H = 200, 60
CHARSET = string.ascii_letters + string.digits  # [a-zA-Z0-9], 62 chars
CAPTCHA_LEN = 5

FONT_CANDIDATES = [
    "arial.ttf", "arialbd.ttf", "arialbi.ttf", "ariblk.ttf",
    "calibri.ttf", "calibrib.ttf", "calibriz.ttf",
    "cambriab.ttf", "cambriaz.ttf",
    "Candara.ttf", "Candarab.ttf", "Candaraz.ttf",
    "comic.ttf", "comicbd.ttf",
    "consola.ttf", "consolab.ttf",
    "cour.ttf", "courbd.ttf",
    "georgia.ttf", "georgiab.ttf", "georgiaz.ttf",
    "impact.ttf",
    "lucon.ttf",
    "tahoma.ttf", "tahomabd.ttf",
    "times.ttf", "timesbd.ttf", "timesbi.ttf",
    "trebuc.ttf", "trebucbd.ttf",
    "verdana.ttf", "verdanab.ttf", "verdanaz.ttf",
    "BOOKOS.TTF", "BOOKOSB.TTF",
    "GARA.TTF", "GARABD.TTF",
    "PALA.TTF", "PALAB.TTF",
]
WIN_FONTS = Path("C:/Windows/Fonts")


def _load_fonts() -> list[Path]:
    found = [WIN_FONTS / name for name in FONT_CANDIDATES if (WIN_FONTS / name).exists()]
    if not found:
        raise RuntimeError("no fonts found; check FONT_CANDIDATES against C:/Windows/Fonts")
    return found


def _saturated_color(rng: random.Random) -> tuple[int, int, int]:
    """Pick a vivid color by sampling HSV with high S/V then converting."""
    import colorsys
    h = rng.random()
    s = rng.uniform(0.6, 1.0)
    v = rng.uniform(0.4, 0.85)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _darken(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in rgb)  # type: ignore


def _faded_color_arr(np_rng: np.random.Generator, n: int) -> np.ndarray:
    """Vectorised pastel-ish colors via HSV. Returns (n, 3) uint8."""
    h = np_rng.random(n)
    s = np_rng.uniform(0.25, 0.75, n)
    v = np_rng.uniform(0.55, 0.92, n)
    # HSV -> RGB inline (avoid heavy imports)
    i = (h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i_mod = i % 6
    r = np.where(i_mod == 0, v, np.where(i_mod == 1, q, np.where(i_mod == 2, p, np.where(i_mod == 3, p, np.where(i_mod == 4, t, v)))))
    g = np.where(i_mod == 0, t, np.where(i_mod == 1, v, np.where(i_mod == 2, v, np.where(i_mod == 3, q, np.where(i_mod == 4, p, p)))))
    b = np.where(i_mod == 0, p, np.where(i_mod == 1, p, np.where(i_mod == 2, t, np.where(i_mod == 3, v, np.where(i_mod == 4, v, q)))))
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _draw_speckle(canvas: Image.Image, rng: random.Random) -> None:
    """Dense fine multi-colored pixel-level speckle/static noise."""
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    arr = np.array(canvas)  # H, W, 4
    h, w = arr.shape[:2]
    density = rng.uniform(0.30, 0.55)
    n = int(h * w * density)
    ys = np_rng.integers(0, h, n)
    xs = np_rng.integers(0, w, n)
    colors = _faded_color_arr(np_rng, n)
    arr[ys, xs, :3] = colors
    canvas.paste(Image.fromarray(arr))


def _draw_grid(canvas: Image.Image, rng: random.Random) -> None:
    """Fine colored grid forming square cells (~12-20px)."""
    draw = ImageDraw.Draw(canvas)
    spacing = rng.randint(13, 19)
    base_alpha = rng.randint(80, 150)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    # color slightly varies per line for the multi-color faint look
    for x in range(rng.randint(0, spacing - 1), CANVAS_W + 1, spacing):
        c = tuple(int(v) for v in _faded_color_arr(np_rng, 1)[0])
        draw.line([(x, 0), (x, CANVAS_H)], fill=c + (base_alpha,), width=1)
    for y in range(rng.randint(0, spacing - 1), CANVAS_H + 1, spacing):
        c = tuple(int(v) for v in _faded_color_arr(np_rng, 1)[0])
        draw.line([(0, y), (CANVAS_W, y)], fill=c + (base_alpha,), width=1)


def _draw_blocks(canvas: Image.Image, rng: random.Random) -> None:
    """Scattered pastel filled rectangles (block-patch style, like real_24/30/40)."""
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    n_blocks = rng.randint(8, 16)
    for _ in range(n_blocks):
        w = rng.randint(20, 45)
        h = rng.randint(18, 38)
        x = rng.randint(-10, CANVAS_W - 5)
        y = rng.randint(-8, CANVAS_H - 5)
        c = tuple(int(v) for v in _faded_color_arr(np_rng, 1)[0])
        alpha = rng.randint(70, 130)
        draw.rectangle([x, y, x + w, y + h], fill=c + (alpha,))
    canvas.alpha_composite(overlay)


def _draw_background(canvas: Image.Image, rng: random.Random) -> None:
    """CryptBB-style background. Three observed styles in real data:
      A) speckle-only (~40%)
      B) grid lines + speckle (~30%)
      C) block patches + speckle (~30%)
    Speckle is always present; grid and blocks are mutually exclusive overlays
    drawn under the speckle."""
    r = rng.random()
    if r < 0.30:
        _draw_grid(canvas, rng)
    elif r < 0.60:
        _draw_blocks(canvas, rng)
    _draw_speckle(canvas, rng)
    if rng.random() < 0.10:
        _draw_scribbles(canvas, rng)


def _draw_scribbles(canvas: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(canvas)
    n_scribbles = rng.randint(1, 3)
    for _ in range(n_scribbles):
        color = _saturated_color(rng) + (rng.randint(80, 160),)
        n_pts = rng.randint(4, 7)
        pts = []
        x = rng.randint(0, CANVAS_W)
        y = rng.randint(0, CANVAS_H)
        for _ in range(n_pts):
            pts.append((x, y))
            x = max(0, min(CANVAS_W, x + rng.randint(-18, 18)))
            y = max(0, min(CANVAS_H, y + rng.randint(-12, 12)))
        draw.line(pts, fill=color, width=1)


def _render_char(ch: str, font_path: Path, size: int, rng: random.Random) -> Image.Image:
    """Render a single char with a 'chromatic replica' shadow: the same glyph drawn
    just behind in a slightly different shade, then the main fill on top.
    Mimics CryptBB-style character styling."""
    font = ImageFont.truetype(str(font_path), size)
    pad = size
    tile = Image.new("RGBA", (size + 2 * pad, size + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)

    fill = _saturated_color(rng)
    # Shadow shade: usually a darker variant of the same hue (chromatic),
    # occasionally a slight hue-shifted variant.
    if rng.random() < 0.8:
        shadow = _darken(fill, rng.uniform(0.45, 0.7))
    else:
        shadow = _saturated_color(rng)  # different-hue echo, rarer
    # Small offset in a random direction; magnitude scaled to font size.
    mag = max(2, size // 14)
    dx = rng.choice([-1, 1]) * rng.randint(mag - 1, mag + 1)
    dy = rng.choice([-1, 1]) * rng.randint(mag - 1, mag + 1)

    # Draw shadow first, then main on top.
    d.text((pad + dx, pad + dy), ch, font=font, fill=shadow + (255,))
    d.text((pad, pad), ch, font=font, fill=fill + (255,))

    bbox = tile.getbbox()
    if bbox is None:
        return tile
    tile = tile.crop(bbox)
    angle = rng.uniform(-28, 28)
    tile = tile.rotate(angle, resample=Image.BICUBIC, expand=True)
    return tile


def _render_one(text: str, fonts: list[Path], rng: random.Random,
                draw_bg: bool = True) -> Image.Image:
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255, 255, 255, 255))
    if draw_bg:
        _draw_background(canvas, rng)

    # Render each char, then place at evenly-spaced anchors with jitter.
    # Keep size modest so rotated tiles fit within the 60px height; we also
    # clamp final paste positions below to guarantee no edge-clipping.
    char_imgs = []
    for ch in text:
        size = rng.randint(34, 42)
        font_path = rng.choice(fonts)
        tile = _render_char(ch, font_path, size, rng)
        # If rotated tile is taller than canvas, scale it down proportionally
        # so it fits — preserves whole character visibility.
        max_h = CANVAS_H - 4
        if tile.height > max_h:
            scale = max_h / tile.height
            new_w = max(1, int(tile.width * scale))
            tile = tile.resize((new_w, max_h), Image.BICUBIC)
        char_imgs.append(tile)

    # 5 anchor x-centers, with margin pulled inward so even the widest rotated
    # tile fits fully on canvas.
    n = len(char_imgs)
    margin_x = 6
    usable_w = CANVAS_W - 2 * margin_x
    stride = usable_w / n
    for i, im in enumerate(char_imgs):
        cx = int(margin_x + (i + 0.5) * stride + rng.randint(-2, 2))
        cy = CANVAS_H // 2 + rng.randint(-2, 2)
        x = cx - im.width // 2
        y = cy - im.height // 2
        # Hard clamp to keep entire tile on canvas — no edge clipping.
        x = max(0, min(CANVAS_W - im.width, x))
        y = max(0, min(CANVAS_H - im.height, y))
        canvas.alpha_composite(im, (x, y))

    return canvas


def main(out_dir: Path, n: int, seed: int, preview: int, draw_bg: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fonts = _load_fonts()
    print(f"loaded {len(fonts)} fonts (draw_bg={draw_bg})")

    rng = random.Random(seed)
    labels_path = out_dir / "labels.csv"

    if preview > 0:
        prev_dir = out_dir / "_preview"
        prev_dir.mkdir(exist_ok=True)
        thumbs = []
        for i in range(preview):
            text = "".join(rng.choices(CHARSET, k=CAPTCHA_LEN))
            img = _render_one(text, fonts, rng, draw_bg=draw_bg)
            img.save(prev_dir / f"{i:04d}_{text}.png")
            thumbs.append(img.convert("RGB"))
        # 5xN grid montage
        cols = 5
        rows = (preview + cols - 1) // cols
        grid = Image.new("RGB", (CANVAS_W * cols, CANVAS_H * rows), (255, 255, 255))
        for i, im in enumerate(thumbs):
            grid.paste(im, ((i % cols) * CANVAS_W, (i // cols) * CANVAS_H))
        grid.save(out_dir / "_preview_montage.png")
        print(f"preview saved to {prev_dir} and montage at {out_dir / '_preview_montage.png'}")
        return

    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for i in tqdm(range(n), desc="rendering"):
            text = "".join(rng.choices(CHARSET, k=CAPTCHA_LEN))
            img = _render_one(text, fonts, rng, draw_bg=draw_bg)
            fn = f"{i:07d}.png"
            img.save(out_dir / fn)
            writer.writerow([fn, text])
    print(f"wrote {n} samples to {out_dir} and labels to {labels_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default="data/synthetic")
    ap.add_argument("-n", type=int, default=100_000, help="number to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0,
                    help="if >0, render this many preview samples and a montage; skip bulk generation")
    ap.add_argument("--no-bg", action="store_true",
                    help="render chars on plain white only (skip background drawing)")
    args = ap.parse_args()
    main(args.out, args.n, args.seed, args.preview, draw_bg=not args.no_bg)
