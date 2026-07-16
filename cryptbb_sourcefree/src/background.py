"""JOB B — recreate the background (GOAL.md). No text, ever.

Two sources, mixable:
  procedural  — CryptBB-like pastel tile mosaic + white speckle + thin colored
                lines. Guaranteed label-clean (no ghost glyphs) and fully under
                our control. The safe default.
  lama        — sample the LaMa-extracted real background pool. These ARE the
                real backgrounds, but some retain ghost-letter fragments, so we
                FILTER (drop ones with too much dark/structured content) before
                use to avoid injecting phantom letters into labeled data.

Keeping B separate from the glyph GAN means the letter-preserving anchors only
protect the glyphs, and the GAN never invents a background from scratch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


# ---- procedural CryptBB-like background --------------------------------------
def procedural_bg(W: int, H: int, rng: np.random.Generator) -> np.ndarray:
    """Return an (H,W,3) uint8 background resembling CryptBB's canvas: a near-white
    field with a DENSE mosaic of pastel squares + fine speckle (matched to the real
    look — prominent tiles, faint/optional thin lines)."""
    base = rng.integers(235, 256, size=3).astype(np.float32)
    img = np.ones((H, W, 3), np.float32) * base

    # dense pastel tile mosaic — square-ish, clearly visible pastels
    n_tiles = int(rng.integers(35, 70))
    for _ in range(n_tiles):
        ts = int(rng.integers(7, 20))
        tw = ts + int(rng.integers(-2, 3)); th = ts + int(rng.integers(-2, 3))
        x0 = int(rng.integers(0, max(1, W - tw))); y0 = int(rng.integers(0, max(1, H - th)))
        # pastel = high value, moderate saturation
        col = rng.integers(175, 245, size=3).astype(np.float32)
        a = rng.uniform(0.25, 0.65)
        img[y0:y0+th, x0:x0+tw] = (1 - a) * img[y0:y0+th, x0:x0+tw] + a * col

    # fine white speckle (salt) — a strong CryptBB trait
    salt = rng.random((H, W)) < rng.uniform(0.03, 0.10)
    img[salt] = 255
    if rng.random() < 0.4:
        pepper = rng.random((H, W)) < rng.uniform(0.003, 0.012)
        img[pepper] = rng.integers(150, 210)

    # occasional faint thin line (real has a few; keep sparse & pale)
    if rng.random() < 0.5:
        from PIL import ImageDraw
        pim = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(pim)
        for _ in range(int(rng.integers(1, 3))):
            col = tuple(int(v) for v in rng.integers(150, 230, size=3))
            pts = [(int(rng.integers(0, W)), int(rng.integers(0, H))) for _ in range(2)]
            d.line(pts, fill=col, width=1)
        img = np.asarray(pim, dtype=np.float32)

    img += rng.normal(0, rng.uniform(1, 4), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def dark_block_bg(W: int, H: int, rng: np.random.Generator) -> np.ndarray:
    """DarkBay-style canvas: overlapping muted dark color rectangles (grey/purple/
    teal/brown blocks) — no text."""
    base = rng.integers(40, 90, size=3).astype(np.float32)
    img = np.ones((H, W, 3), np.float32) * base
    for _ in range(int(rng.integers(14, 34))):
        tw = int(rng.integers(10, 60)); th = int(rng.integers(8, H))
        x0 = int(rng.integers(0, max(1, W - tw))); y0 = int(rng.integers(0, max(1, H - th)))
        col = rng.integers(35, 130, size=3).astype(np.float32)
        a = rng.uniform(0.3, 0.8)
        img[y0:y0+th, x0:x0+tw] = (1 - a) * img[y0:y0+th, x0:x0+tw] + a * col
    img += rng.normal(0, rng.uniform(2, 7), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---- LaMa background pool (filtered) -----------------------------------------
def _has_ghost_glyph(a: np.ndarray, ink_thresh: int = 150,
                     max_blob: int = 120) -> bool:
    """Reject a LaMa background whose LARGEST coherent dark blob (in the central
    glyph band) exceeds max_blob px — that's a residual ghost letter or a heavy
    scribble. Clean tile+speckle backgrounds have only small components. Tuned so
    ~124/200 CryptBB LaMa bgs (the clean ones) survive."""
    from scipy import ndimage
    H, W = a.shape[:2]
    gray = a.mean(2)
    band = np.zeros_like(gray, bool)
    band[int(H * 0.12):int(H * 0.88)] = True
    ink = (gray < ink_thresh) & band
    lab, n = ndimage.label(ink)
    if n == 0:
        return False
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    return bool(sizes.max() >= max_blob)


class LamaBGPool:
    def __init__(self, bg_dir: str | Path, canvas: tuple[int, int],
                 dark_thresh: int = 85, max_dark_frac: float = 0.15):
        """Load and FILTER the LaMa bg pool: reject backgrounds that still contain
        a ghost letter (largest-dark-blob test) OR excessive dark content."""
        self.W, self.H = canvas
        self.imgs: list[np.ndarray] = []
        kept = dropped = 0
        for p in sorted(Path(bg_dir).glob("*.png")):
            im = Image.open(p).convert("RGB").resize((self.W, self.H), Image.BILINEAR)
            a = np.asarray(im)
            dark_frac = float((a.mean(2) < dark_thresh).mean())
            if dark_frac <= max_dark_frac and not _has_ghost_glyph(a):
                self.imgs.append(a); kept += 1
            else:
                dropped += 1
        self.kept, self.dropped = kept, dropped

    def __len__(self):
        return len(self.imgs)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.imgs[int(rng.integers(0, len(self.imgs)))].copy()


# ---- compositing -------------------------------------------------------------
def composite(glyph_rgba: Image.Image, bg_rgb: np.ndarray) -> np.ndarray:
    """Alpha-composite the glyph layer over an (H,W,3) uint8 background."""
    H, W = bg_rgb.shape[:2]
    if glyph_rgba.size != (W, H):
        glyph_rgba = glyph_rgba.resize((W, H), Image.BILINEAR)
    bg = Image.fromarray(bg_rgb).convert("RGBA")
    bg.alpha_composite(glyph_rgba)
    return np.asarray(bg.convert("RGB"))
