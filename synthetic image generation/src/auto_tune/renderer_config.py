"""Renderer parameter dataclass + random sampler.

Defines all knobs the auto-tuner can vary. Sampler returns a RendererConfig
drawn uniformly from the search ranges below. Adding a new knob = one field
+ one sampler line; the renderer reads from the dataclass directly.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RendererConfig:
    # font selection: indices into the FONT_POOL list (defined in render_darkbay_v2)
    font_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # per-char font size range (px)
    size_min: int = 22
    size_max: int = 28

    # per-char rotation range (degrees)
    rot_max: float = 12.0  # symmetric: rot ∈ [-rot_max, +rot_max]

    # box decoration probability (per char)
    box_prob: float = 0.5

    # gray-value range for the dark box fill
    box_color_min: int = 60
    box_color_max: int = 95

    # gray-value range for plain (no-box) text
    dark_text_min: int = 40
    dark_text_max: int = 90

    # gray-value range for boxed (white-on-dark) text
    light_text_min: int = 230
    light_text_max: int = 255

    # box padding around the char (px, asymmetric jitter)
    box_pad_min: int = 1
    box_pad_max: int = 3

    # per-char position jitter (px)
    pos_jitter_x: float = 2.0
    pos_jitter_y: float = 2.0

    # char center spacing (multiplier of W/N_CHARS; 1.0 = even, <1 = overlap, >1 = wider)
    spacing_mult: float = 1.0

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "RendererConfig":
        return cls(**json.loads(path.read_text()))


def sample_random(rng: random.Random) -> RendererConfig:
    """Draw one config uniformly from the search space."""
    size_lo = rng.randint(18, 26)
    size_hi = rng.randint(size_lo + 2, size_lo + 10)
    box_lo = rng.randint(40, 80)
    box_hi = rng.randint(box_lo + 5, min(120, box_lo + 40))
    dark_lo = rng.randint(20, 60)
    dark_hi = rng.randint(dark_lo + 10, min(110, dark_lo + 50))
    light_lo = rng.randint(200, 240)
    light_hi = rng.randint(light_lo + 5, 255)
    box_pad_lo = rng.randint(0, 3)
    box_pad_hi = rng.randint(box_pad_lo + 1, box_pad_lo + 5)
    return RendererConfig(
        font_indices=sorted(rng.sample([0, 1, 2, 3], k=rng.randint(1, 4))),
        size_min=size_lo,
        size_max=size_hi,
        rot_max=rng.uniform(0.0, 25.0),
        box_prob=rng.uniform(0.0, 1.0),
        box_color_min=box_lo,
        box_color_max=box_hi,
        dark_text_min=dark_lo,
        dark_text_max=dark_hi,
        light_text_min=light_lo,
        light_text_max=light_hi,
        box_pad_min=box_pad_lo,
        box_pad_max=box_pad_hi,
        pos_jitter_x=rng.uniform(0.0, 5.0),
        pos_jitter_y=rng.uniform(0.0, 5.0),
        spacing_mult=rng.uniform(0.85, 1.10),
    )
