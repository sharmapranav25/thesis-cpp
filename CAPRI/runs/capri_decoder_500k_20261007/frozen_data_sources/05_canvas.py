"""Layered canvas with exact per-pixel labels.

Every drawing operation is rasterised once as a coverage mask (0..1), blended into the RGB image, and recorded:
  * `opaque` id map: the op's id where its coverage >= 0.5 and its opacity >= 0.5 (latest op wins);
  * `translucent` id map: the same for ops with opacity < 0.5;
  * `records`: one dict per op (id, effect, head, layer, params, colour, opacity).
Ids continue from `first_id`, so a canvas can be stacked on an earlier stage's labels.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Canvas:
    w: int
    h: int
    background: tuple = (255, 255, 255)
    first_id: int = 1
    supersample: int = 4
    img: np.ndarray = field(init=False)
    opaque: np.ndarray = field(init=False)
    translucent: np.ndarray = field(init=False)
    records: list = field(default_factory=list)
    text_colour: tuple = None   # set by the caller when known (enables the match_text palette)

    def __post_init__(self):
        self.img = np.zeros((self.h, self.w, 3), np.float32) + np.array(self.background, np.float32)
        self.opaque = np.zeros((self.h, self.w), np.uint32)
        self.translucent = np.zeros((self.h, self.w), np.uint32)
        self.next_id = int(self.first_id)

    @classmethod
    def from_arrays(cls, img, opaque=None, translucent=None, next_id=1):
        """Start from an existing image (and its label maps, if any)."""
        h, w = img.shape[:2]
        c = cls(w, h, tuple(int(v) for v in np.median(img.reshape(-1, 3), 0)), first_id=next_id)
        c.img = img.astype(np.float32).copy()
        if opaque is not None:
            c.opaque = opaque.astype(np.uint32).copy()
        if translucent is not None:
            c.translucent = translucent.astype(np.uint32).copy()
        return c

    # ------------------------------------------------------------------ rasterising
    def mask(self, draw_fn, antialias=True):
        """Coverage mask from a PIL drawing function f(draw, scale). Supersampled when antialias=True."""
        s = self.supersample if antialias else 1
        m = Image.new("L", (self.w * s, self.h * s), 0)
        draw_fn(ImageDraw.Draw(m), s)
        if s > 1:
            m = m.resize((self.w, self.h), Image.BOX)
        return np.asarray(m, np.float32) / 255.0

    def record(self, **rec):
        cid = self.next_id
        self.next_id += 1
        rec["id"] = cid
        self.records.append(rec)
        return cid

    def blend(self, cov, colour, effect, head, layer, opacity=1.0, params=None, colour_field=None):
        """Blend `colour` (or an H x W x 3 `colour_field`) through coverage `cov` at `opacity`; label it."""
        cid = self.record(effect=effect, head=head, layer=layer, params=params or {},
                          colour=None if colour is None else [int(c) for c in colour], opacity=round(float(opacity), 3))
        a = (np.clip(cov, 0, 1) * opacity)[..., None]
        src = colour_field if colour_field is not None else np.array(colour, np.float32)
        self.img = self.img * (1 - a) + src * a
        sel = cov >= 0.5
        (self.translucent if opacity < 0.5 else self.opaque)[sel] = cid
        return cid

    def apply_field(self, field_rgb, effect, head, layer, opacity=1.0, params=None):
        """Whole-canvas colour field (gradient, texture): labelled over every pixel."""
        return self.blend(np.ones((self.h, self.w), np.float32), None, effect, head, layer, opacity, params, colour_field=field_rgb)

    # ------------------------------------------------------------------ results
    def image(self):
        return self.img.clip(0, 255).round().astype(np.uint8)

    def visible_pixels(self):
        """Visible pixels per head (opaque or translucent id map) in the current canvas."""
        ids = np.where(self.opaque > 0, self.opaque, self.translucent)
        u, c = np.unique(ids, return_counts=True)
        vis = dict(zip(u.tolist(), c.tolist()))
        out = {}
        for r in self.records:
            out[r["head"]] = out.get(r["head"], 0) + vis.get(r["id"], 0)
        return out
