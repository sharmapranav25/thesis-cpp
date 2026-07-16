"""PHP-GD captcha distortion family — the domain-knowledge prior.

Most text captchas are PHP-GD renders from a small known family (Securimage,
Gregwar/Captcha, cool-php-captcha, phpBB, MyBB, the imagettftext + sine-warp +
noise-line recipe). Our plain PIL compositing was OFF-family, which is a big part
of the synth->real gap on exotic-font schemes. This module replicates the shared
GD primitives so our source-free synth looks like *a PHP captcha* (the thing the
real one actually is) — without using any specific scheme's captcha.php or fonts.

Primitives (all applied to a composited RGB image, label unchanged since WE chose
the text):
  sine_warp   - the signature captcha wobble (elastic x/y sine displacement)
  add_lines   - random thin straight noise lines (GD imageline)
  add_arcs    - random arcs/curves (GD imagearc)
  add_dots    - random dot/speckle noise

`PROFILES` holds named parameter sets ("php_generic" etc.); a scheme config
selects one. Later we can add a "php_render" profile that shells to real PHP libs.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def sine_warp(arr: np.ndarray, rng: np.random.Generator,
              amp_x=(0.0, 2.5), amp_y=(1.0, 4.0), periods=(1.0, 2.5)) -> np.ndarray:
    """Elastic sine displacement (GD-style wave). arr: (H,W,3) uint8."""
    from scipy.ndimage import map_coordinates
    H, W = arr.shape[:2]
    ax = rng.uniform(*amp_x); ay = rng.uniform(*amp_y)
    fx = rng.uniform(*periods) / H; fy = rng.uniform(*periods) / W
    px = rng.uniform(0, 2 * np.pi); py = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:H, 0:W]
    # column shift depends on row (vertical wobble), row shift depends on col
    new_x = xx + ax * np.sin(2 * np.pi * fx * yy + px)
    new_y = yy + ay * np.sin(2 * np.pi * fy * xx + py)
    coords = np.stack([new_y.ravel(), new_x.ravel()])
    out = np.empty_like(arr)
    for c in range(3):
        out[..., c] = map_coordinates(arr[..., c], coords, order=1, mode="nearest").reshape(H, W)
    return out


def add_lines(pil: Image.Image, rng: np.random.Generator, n=(1, 4), width=(1, 2),
              gray=False) -> Image.Image:
    d = ImageDraw.Draw(pil); W, H = pil.size
    for _ in range(int(rng.integers(*n))):
        col = (0, 0, 0) if gray and rng.random() < 0.5 else \
              (255, 255, 255) if gray else tuple(int(v) for v in rng.integers(0, 200, 3))
        p = [(int(rng.integers(0, W)), int(rng.integers(0, H))) for _ in range(2)]
        d.line(p, fill=col, width=int(rng.integers(*width)))
    return pil


def add_arcs(pil: Image.Image, rng: np.random.Generator, n=(0, 3)) -> Image.Image:
    d = ImageDraw.Draw(pil); W, H = pil.size
    for _ in range(int(rng.integers(*n))):
        x0 = int(rng.integers(-W // 2, W)); y0 = int(rng.integers(-H, H))
        x1 = x0 + int(rng.integers(W // 3, W)); y1 = y0 + int(rng.integers(H // 2, H * 2))
        col = tuple(int(v) for v in rng.integers(0, 200, 3))
        d.arc([x0, y0, x1, y1], int(rng.integers(0, 360)), int(rng.integers(0, 360)),
              fill=col, width=1)
    return pil


def add_dots(pil: Image.Image, rng: np.random.Generator, density=(0.0, 0.03)) -> Image.Image:
    arr = np.asarray(pil).copy()
    H, W = arr.shape[:2]
    m = rng.random((H, W)) < rng.uniform(*density)
    arr[m] = rng.integers(0, 255, size=(m.sum(), 3))
    return Image.fromarray(arr)


PROFILES = {
    # generic PHP-GD text-captcha look: moderate wobble + a few noise lines/dots.
    "php_generic": dict(warp=True, amp_x=(0.0, 2.5), amp_y=(1.0, 4.0), periods=(1.0, 2.5),
                        lines=(1, 4), arcs=(0, 3), dots=(0.0, 0.02), gray_lines=False),
    # heavier wobble (Securimage-ish)
    "php_wavy": dict(warp=True, amp_x=(1.0, 3.5), amp_y=(2.0, 6.0), periods=(1.5, 3.0),
                     lines=(2, 6), arcs=(1, 4), dots=(0.0, 0.03), gray_lines=False),
    # dark/mono schemes (DarkBay-ish): gray noise lines, mild warp
    "php_mono": dict(warp=True, amp_x=(0.0, 2.0), amp_y=(1.0, 3.0), periods=(1.0, 2.0),
                     lines=(1, 4), arcs=(0, 2), dots=(0.0, 0.015), gray_lines=True),
    "none": None,
}


def apply_profile(pil: Image.Image, rng: np.random.Generator, profile: str) -> Image.Image:
    """Apply a named PHP-GD distortion profile to a composited RGB image."""
    p = PROFILES.get(profile)
    if p is None:
        return pil
    arr = np.asarray(pil.convert("RGB"))
    if p["warp"]:
        arr = sine_warp(arr, rng, p["amp_x"], p["amp_y"], p["periods"])
    out = Image.fromarray(arr)
    out = add_lines(out, rng, p["lines"], gray=p["gray_lines"])
    out = add_arcs(out, rng, p["arcs"])
    out = add_dots(out, rng, p["dots"])
    return out
