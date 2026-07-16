"""ParametricGenerator: difficulty-knob synth captcha producer.

One public method: `render(t: float, n: int) -> list[(image, label, bboxes)]`.
Internally:
  1. interpolate(EASY, HARD, t) → RendererConfig
  2. render_one(config) for each of n samples → text-only RGBA + bboxes
  3. sample a real bg + jitter (color/brightness/contrast + small translation)
  4. alpha-composite text over bg
  5. paint a 1-px black border (matches real DarkBay)

Backgrounds are loaded once at init from `data/darkbay_bg_lama/` (1000 PNGs).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auto_tune.difficulty import EASY, interpolate, load_hard
from src.auto_tune.render_darkbay_v2 import (
    H, W, load_fonts_for_config, random_text, render_one,
)
from src.auto_tune.renderer_config import RendererConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BG_DIR = ROOT / "data" / "darkbay_bg_lama"


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


class ParametricGenerator:
    def __init__(self, bg_dir: Optional[Path] = None,
                 hard: Optional[RendererConfig] = None, seed: int = 0):
        bg_dir = Path(bg_dir) if bg_dir else DEFAULT_BG_DIR
        self.bgs = [Image.open(p).convert("RGB")
                    for p in sorted(bg_dir.glob("*.png"))]
        if not self.bgs:
            raise RuntimeError(f"no backgrounds in {bg_dir}")
        self.easy = EASY
        self.hard = hard or load_hard()
        self.rng = random.Random(seed)
        self._fonts_cache: dict[tuple, dict] = {}

    def _config_for(self, t: float) -> RendererConfig:
        return interpolate(self.easy, self.hard, t)

    def _fonts_for(self, config: RendererConfig):
        # cache by (font_indices, size_min, size_max) — load_fonts_for_config is
        # cheap but not free, and difficulty interpolation often produces the
        # same key across batches.
        key = (tuple(config.font_indices), config.size_min, config.size_max)
        if key not in self._fonts_cache:
            self._fonts_cache[key] = load_fonts_for_config(config)
        return self._fonts_cache[key]

    def render_sample(self, t: float) -> tuple[Image.Image, str, list[list[int]]]:
        return self.render_sample_with_mask(t)[:3]

    def render_sample_with_mask(self, t: float) -> tuple[Image.Image, str, list[list[int]], np.ndarray]:
        """Same as render_sample but also returns the text alpha mask
        (HxW float in [0, 1]) — the exact glyph silhouette before compositing.
        """
        config = self._config_for(t)
        fonts = self._fonts_for(config)
        text = random_text(self.rng)
        text_im, bboxes = render_one(text, config, fonts, self.rng)
        alpha = np.array(text_im.split()[-1]).astype(np.float32) / 255.0  # HxW in [0,1]

        bg = self._jitter(self.rng.choice(self.bgs).copy())
        if bg.size != (W, H):
            bg = bg.resize((W, H))
        composed = Image.alpha_composite(bg.convert("RGBA"), text_im).convert("RGB")
        arr = np.array(composed)
        arr[0, :] = 0; arr[-1, :] = 0; arr[:, 0] = 0; arr[:, -1] = 0
        return Image.fromarray(arr), text, bboxes, alpha

    def _jitter(self, bg: Image.Image) -> Image.Image:
        return _jitter_bg(bg, self.rng)

    def render(self, t: float, n: int) -> list[tuple[Image.Image, str, list[list[int]]]]:
        return [self.render_sample(t) for _ in range(n)]
