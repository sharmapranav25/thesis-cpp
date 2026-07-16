"""Difficulty axis: interpolate between EASY and HARD RendererConfig.

`interpolate(easy, hard, t)` returns a RendererConfig whose numeric knobs are
linearly interpolated and whose discrete knobs (font set) grow with t.

The hard anchor is loaded from the saved DINOv2 search winner; the easy anchor
is hand-set to a clean baseline (no rotation, single font, no boxes, no jitter).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auto_tune.renderer_config import RendererConfig

ROOT = Path(__file__).resolve().parents[2]
HARD_CONFIG_PATH = ROOT / "configs" / "darkbay_renderer.json"

EASY = RendererConfig(
    font_indices=[2],          # Cambria Bold — clean serif, single font
    size_min=25, size_max=27,  # narrow size range
    rot_max=0.0,
    box_prob=0.0,
    box_color_min=42, box_color_max=70,      # unused at box_prob=0; mirror hard
    dark_text_min=20, dark_text_max=46,      # mirror hard
    light_text_min=237, light_text_max=243,  # unused at box_prob=0; mirror hard
    box_pad_min=0, box_pad_max=1,            # mirror hard
    pos_jitter_x=0.0, pos_jitter_y=0.0,
    spacing_mult=1.0,
)


def load_hard() -> RendererConfig:
    return RendererConfig.from_json(HARD_CONFIG_PATH)


_INT_FIELDS = (
    "size_min", "size_max",
    "box_color_min", "box_color_max",
    "dark_text_min", "dark_text_max",
    "light_text_min", "light_text_max",
    "box_pad_min", "box_pad_max",
)
_FLOAT_FIELDS = (
    "rot_max", "box_prob",
    "pos_jitter_x", "pos_jitter_y",
    "spacing_mult",
)


def interpolate(easy: RendererConfig, hard: RendererConfig, t: float) -> RendererConfig:
    t = max(0.0, min(1.0, t))
    out: dict = {}
    for f in _FLOAT_FIELDS:
        ev = getattr(easy, f)
        hv = getattr(hard, f)
        out[f] = ev + (hv - ev) * t
    for f in _INT_FIELDS:
        ev = getattr(easy, f)
        hv = getattr(hard, f)
        out[f] = int(round(ev + (hv - ev) * t))

    # discrete font_indices: grow from easy's set to hard's set as t rises.
    extras = [i for i in hard.font_indices if i not in easy.font_indices]
    # how many extras to include scales with t (rounded up so any t > 0 adds one)
    import math
    n_extras = math.ceil(len(extras) * t) if t > 0 else 0
    out["font_indices"] = sorted(set(easy.font_indices) | set(extras[:n_extras]))

    if out["size_min"] > out["size_max"]:
        out["size_min"], out["size_max"] = out["size_max"], out["size_min"]

    return RendererConfig(**out)


if __name__ == "__main__":
    hard = load_hard()
    print(f"EASY: {EASY}")
    print(f"HARD: {hard}")
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        cfg = interpolate(EASY, hard, t)
        print(f"\nt={t}:")
        for k, v in vars(cfg).items():
            print(f"  {k} = {v}")
