"""RealBgGenerator: high-t bg path that preserves real wallpaper texture.

Pipeline:
  1. Sample a real captcha (full RGB).
  2. Render text-only RGBA with current config.
  3. Build a tight inpaint mask covering ONLY where our chars will go.
  4. cv2.INPAINT_TELEA to fill that small region with real-bg context.
  5. Alpha-composite our synth text on top of the inpainted real captcha.

Result: real wallpaper texture everywhere outside the text region, our
labeled text in the text region. No GAN, no training, no LaMa flattening.

Trade-off vs ParametricGenerator (with LaMa bg pool):
  + Real wallpaper texture preserved (the missing piece for "looks real").
  - Each real captcha can be reused, but the underlying bg distribution is
    only 1000 unique reals — less variance than LaMa pool over time.
  - Inpaint region must cover the original captcha's text too (otherwise the
    real text bleeds through). We dilate the inpaint mask to be safe.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auto_tune.render_darkbay_v2 import (
    H, W, load_fonts_for_config, random_text, render_one,
)
from src.auto_tune.renderer_config import RendererConfig
from src.auto_tune.difficulty import EASY, interpolate, load_hard
from src.generator.parametric import ParametricGenerator, _jitter_bg

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_DIR = Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled")

# Threshold for "this pixel was changed by LaMa = original captcha had text-styling here."
# Used to mask out the original real captcha's text region before our synth text goes on top.
ORIG_TEXT_DIFF_THR = 50
ORIG_TEXT_DILATE = 2


class RealBgGenerator(ParametricGenerator):
    """Generator using real captchas as bg substrate (instead of LaMa pool).

    Falls back to ParametricGenerator's LaMa pool at low t (for the easy
    end of the curriculum, where flat bgs are appropriate). At high t,
    real-captcha bgs with local inpainting kick in.

    `bg_switch_t` is the t at which we transition from LaMa to real bgs.
    """
    def __init__(self, real_dir: Optional[Path] = None,
                 lama_bg_dir: Optional[Path] = None,
                 bg_switch_t: float = 0.5, seed: int = 0):
        super().__init__(bg_dir=lama_bg_dir, seed=seed)
        real_dir = Path(real_dir) if real_dir else DEFAULT_REAL_DIR
        self.real_paths = sorted(real_dir.glob("*.png"))
        if not self.real_paths:
            raise RuntimeError(f"no real captchas in {real_dir}")
        self.lama_dir = self.bgs  # parent loaded these into RAM as Image objects
        self.bg_switch_t = bg_switch_t
        # Lazy-load real captchas as np arrays on first request to amortize
        # the up-front cost; cached after.
        self._real_cache: list[np.ndarray] = []
        # Pre-compute LaMa-vs-real diff masks so we know each real's original
        # text region (to mask out before overlaying our text).
        self._orig_text_masks: list[np.ndarray] = []
        self._init_real_pool()

    def _init_real_pool(self) -> None:
        # Match real captchas to their LaMa output by filename.
        from src.refine.data_bg import text_mask_from_diff
        from src.generator.parametric import DEFAULT_BG_DIR
        lama_dir = DEFAULT_BG_DIR
        for p in self.real_paths:
            real_pil = Image.open(p).convert("RGB")
            if real_pil.size != (W, H):
                real_pil = real_pil.resize((W, H))
            real = np.array(real_pil)
            lama_path = lama_dir / p.name
            if lama_path.exists():
                lama_pil = Image.open(lama_path).convert("RGB")
                if lama_pil.size != (W, H):
                    lama_pil = lama_pil.resize((W, H))
                lama = np.array(lama_pil)
                orig_text = text_mask_from_diff(real, lama)
            else:
                orig_text = np.zeros((H, W), dtype=np.float32)
            self._real_cache.append(real)
            self._orig_text_masks.append(orig_text)

    @staticmethod
    def _build_inpaint_mask(bboxes: list[list[int]], orig_text: np.ndarray,
                            dilate: int = 4) -> np.ndarray:
        """Combine our planned char bboxes with the real captcha's original
        text region; dilate; return uint8 mask suitable for cv2.inpaint."""
        m = np.zeros((H, W), dtype=np.uint8)
        for x0, y0, x1, y1 in bboxes:
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(W, x1); y1 = min(H, y1)
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = 255
        m = np.maximum(m, (orig_text > 0).astype(np.uint8) * 255)
        if dilate > 0:
            kernel = np.ones((3, 3), np.uint8)
            m = cv2.dilate(m, kernel, iterations=dilate)
        return m

    def render_sample(self, t: float) -> tuple[Image.Image, str, list[list[int]]]:
        if t < self.bg_switch_t:
            # easy end: keep parametric path with LaMa bg
            return super().render_sample(t)

        # high t: real-bg path
        config = self._config_for(t)
        fonts = self._fonts_for(config)
        text = random_text(self.rng)
        text_im, bboxes = render_one(text, config, fonts, self.rng)

        idx = self.rng.randrange(len(self._real_cache))
        real_bg = self._real_cache[idx].copy()
        orig_text = self._orig_text_masks[idx]
        inpaint_mask = self._build_inpaint_mask(bboxes, orig_text, dilate=4)
        real_bg = cv2.inpaint(real_bg, inpaint_mask, inpaintRadius=3,
                              flags=cv2.INPAINT_TELEA)

        # Light jitter to add some variation since we're reusing reals.
        bg_pil = Image.fromarray(real_bg)
        bg_pil = _jitter_bg(bg_pil, self.rng)

        composed = Image.alpha_composite(bg_pil.convert("RGBA"), text_im).convert("RGB")
        arr = np.array(composed)
        arr[0, :] = 0; arr[-1, :] = 0; arr[:, 0] = 0; arr[:, -1] = 0
        return Image.fromarray(arr), text, bboxes


if __name__ == "__main__":
    gen = RealBgGenerator(seed=0)
    print(f"loaded {len(gen._real_cache)} real captchas + {len(gen._orig_text_masks)} masks")
    for t in (0.3, 0.7, 1.0):
        im, label, bb = gen.render_sample(t)
        out_path = ROOT / "mvp_results" / f"_realbg_t{t}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path)
        print(f"  t={t}: {label}  saved {out_path.name}")
