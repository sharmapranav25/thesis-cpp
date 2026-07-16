"""Compose (pipeline step 5, real-crop path) — spell OUR chosen text with REAL glyph
crops (exact font by construction) composited onto a generated background.

This is the chosen glyph strategy: recreation-from-libraries gives poor font fidelity
(DINOv2 is only coarse and no generic font matches the exotic scheme faces), so we
paste actual extracted glyph pixels. The background is php-GENERATED (clean, no ghost
residue) rather than a reused real plate. Labels are correct by construction because
WE pick the text and the bank maps each crop to a character.

Glyph bank + alpha keying reused from auto_synth (vendored as _glyph_bank.py).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from _glyph_bank import GlyphBank


def to_alpha_smart(glyph: Image.Image, tol: int = 36) -> Image.Image:
    """Alpha-key by the crop's ACTUAL background colour (its 2px border median),
    not just pure white — kills the rectangular halo auto_synth's white_thresh left
    on crops whose background is off-white/grey. Erodes 1px to shave anti-alias edges."""
    rgb = np.asarray(glyph.convert("RGB")).astype(np.int16)
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[:2].reshape(-1, 3), rgb[-2:].reshape(-1, 3),
                             rgb[:, :2].reshape(-1, 3), rgb[:, -2:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(-1))
    alpha = (dist > tol).astype(np.uint8) * 255
    try:
        import cv2
        alpha = cv2.erode(alpha, np.ones((2, 2), np.uint8), iterations=1)
        alpha = cv2.medianBlur(alpha, 3)   # drop speckle
    except Exception:
        pass
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), "RGBA")


def _scale_to_height(rgba: Image.Image, h: int) -> Image.Image:
    if rgba.height <= 0:
        return rgba
    s = h / rgba.height
    return rgba.resize((max(4, int(rgba.width * s)), h), Image.LANCZOS)


def compose_one(bank: GlyphBank, bg: Image.Image, text: str, rng: random.Random,
                target_h_frac: float = 0.72, overlap: float = 0.12) -> Image.Image:
    """Composite real glyphs spelling `text` onto `bg`. Layout = centred L->R slots
    with y-jitter and mild overlap (real captchas overlap; auto_synth's even slots
    were too regular)."""
    W, H = bg.size
    canvas = bg.convert("RGBA")
    th = int(H * target_h_frac)
    glyphs = []
    for ch in text:
        g = bank.render_char(ch, seed=rng.randint(0, 2 ** 31 - 1))
        glyphs.append(_scale_to_height(to_alpha_smart(g), rng.randint(int(th * 0.9), th)))
    gap = max(1, int(th * 0.06))
    advance = [g.width + gap - int(g.width * overlap * rng.random()) for g in glyphs]
    total = sum(advance[:-1]) + (glyphs[-1].width if glyphs else 0)
    x = max(2, (W - total) // 2 + rng.randint(-4, 4))
    for i, g in enumerate(glyphs):
        y = int((H - g.height) / 2 + rng.randint(-3, 3))
        canvas.alpha_composite(g, (max(0, min(W - g.width, x)), max(0, y)))
        x += advance[i]
    return canvas.convert("RGB")


def bank_for(scheme_dir: str | Path) -> GlyphBank:
    return GlyphBank(Path(scheme_dir))


def sample_text(bank: GlyphBank, length: int, rng: random.Random) -> str:
    return "".join(rng.choice(bank.observed_chars) for _ in range(length))
