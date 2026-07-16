"""Compose a synthetic captcha from a per-scheme char bank + bg pool.

For each render:
  1. Sample N chars from `available_charset()` (uniform).
  2. For each char, sample a real RGBA crop (build_bank.py output) +
     small geometric jitter (rotation, scale).
  3. Normalize each crop to a target glyph height ≈ scheme's char height.
  4. Place crops at evenly-spaced cx positions with per-char x/y jitter.
  5. Sample a bg from the LaMa pool, light color/brightness/contrast jitter +
     small translation.
  6. Alpha-composite chars over bg, paint 1-px black border, return RGB.

The resulting image goes through `recognize`-quality detection naturally
because the char silhouettes come from real captchas of the same scheme.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.synth_v1.font_bank import CharBank
from src.synth_v1.scheme import SchemeSpec

# Target glyph height as a fraction of image H. Crops are scaled to this
# before placement; ~70% leaves visible margin top/bottom and matches what
# real CryptBB chars look like at 200×60.
TARGET_GLYPH_HFRAC = 0.72

# Per-char placement jitter (px) on top of even spacing.
PCJ_X = 3.0
PCJ_Y = 2.0


def _jitter_bg(bg: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.7:
        bg = ImageEnhance.Color(bg).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.7:
        bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.7:
        bg = ImageEnhance.Contrast(bg).enhance(rng.uniform(0.9, 1.1))
    dx = rng.randint(-10, 10)
    dy = rng.randint(-3, 3)
    if dx or dy:
        arr = np.array(bg)
        arr = np.roll(arr, (dy, dx), axis=(0, 1))
        bg = Image.fromarray(arr)
    return bg


def _normalize_height(im: Image.Image, target_h: int) -> Image.Image:
    if im.height == target_h:
        return im
    new_w = max(4, int(round(im.width * (target_h / max(1, im.height)))))
    return im.resize((new_w, target_h), Image.BICUBIC)


def random_text(rng: random.Random, charset: str, n: int) -> str:
    return "".join(rng.choice(charset) for _ in range(n))


class Composer:
    def __init__(self, scheme: SchemeSpec, bank: CharBank, bg_dir: Path = None,
                 seed: int = 0):
        bg_dir = Path(bg_dir) if bg_dir else scheme.bg_dir
        self.bgs = [Image.open(p).convert("RGB")
                    for p in sorted(bg_dir.glob("*.png"))]
        if not self.bgs:
            raise RuntimeError(f"no bgs in {bg_dir}")
        self.bank = bank
        self.scheme = scheme
        self.rng = random.Random(seed)
        # Filter scheme.charset down to what the bank actually has,
        # so random_text() never asks for an unsampleable character.
        avail = set(bank.available_charset())
        self.charset = "".join(c for c in scheme.charset if c in avail)
        if len(self.charset) < 8:
            raise RuntimeError(
                f"bank only covers {len(self.charset)} chars; "
                f"too sparse to compose. Have: {self.charset!r}"
            )

    def render_one(self) -> tuple[Image.Image, str, list[list[int]]]:
        s = self.scheme
        text = random_text(self.rng, self.charset, s.n_chars)

        target_h = int(round(s.H * TARGET_GLYPH_HFRAC))
        glyphs: list[Image.Image] = []
        for ch in text:
            crop = self.bank.sample(ch, self.rng)
            crop = _normalize_height(crop, target_h)
            glyphs.append(crop)

        # bg
        bg = self.rng.choice(self.bgs).copy()
        if bg.size != (s.W, s.H):
            bg = bg.resize((s.W, s.H))
        bg = _jitter_bg(bg, self.rng)
        canvas = bg.convert("RGBA")

        # placement: even cx spacing, per-char x/y jitter, vertical center
        spacing = s.W / s.n_chars
        bboxes: list[list[int]] = []
        for i, g in enumerate(glyphs):
            cx = spacing * (i + 0.5) + self.rng.uniform(-PCJ_X, PCJ_X)
            cy = s.H / 2 + self.rng.uniform(-PCJ_Y, PCJ_Y)
            px = int(round(cx - g.width / 2))
            py = int(round(cy - g.height / 2))
            canvas.alpha_composite(g, (max(0, px), max(0, py)))
            x0 = max(0, px); y0 = max(0, py)
            x1 = min(s.W - 1, px + g.width); y1 = min(s.H - 1, py + g.height)
            bboxes.append([x0, y0, x1, y1])

        out = canvas.convert("RGB")
        arr = np.array(out)
        arr[0, :] = 0; arr[-1, :] = 0; arr[:, 0] = 0; arr[:, -1] = 0
        return Image.fromarray(arr), text, bboxes


if __name__ == "__main__":
    from src.synth_v1.scheme import get
    s = get("cryptbb")
    bank = CharBank(s.bank_dir, min_count=5)
    comp = Composer(s, bank, seed=0)
    out_dir = Path(__file__).resolve().parents[2] / "mvp_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        im, lbl, bb = comp.render_one()
        im.save(out_dir / f"_synth_v1_smoke_{i}_{lbl}.png")
    print(f"saved 8 smoke samples to {out_dir}")
