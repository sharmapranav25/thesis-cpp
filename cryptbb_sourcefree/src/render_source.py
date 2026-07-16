"""Neutral source renderer — JOB 0 of the factory.

Turns a random string we CHOSE into a text image + binary glyph mask. The label
is the string we typed, so every output is labeled by construction (GOAL.md's
iron rule). Uses a GENERIC font (DejaVu), never MyBB fonts.

Two products:
  render_layer() -> (glyph_rgba, mask, label)  transparent glyph layer to
                    composite onto any background (the factory path).
  render()       -> (rgb_on_white, mask, label) flattened on white, for the
                    reader cold-start / read-back checks.

Styling we control (so labels stay exact): per-char color, per-char rotation,
size/spacing jitter. The GAN later refines these toward the real font look. A
`neutral=True` flag renders dark glyphs on white (clean) for the reader.

Scheme-general: canvas / charset / length come from a config entry.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    r"E:/pranav/model 1/.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
    r"E:/pranav/model 1/.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
    r"E:/pranav/model 1/.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono-Bold.ttf",
    r"C:/Windows/Fonts/arialbd.ttf",
    r"C:/Windows/Fonts/arial.ttf",
]


def _available_fonts() -> list[str]:
    return [f for f in _FONT_CANDIDATES if Path(f).is_file()]


DEFAULT_FONTS = _available_fonts()


def _rand_color(rng: random.Random) -> tuple[int, int, int]:
    """A vivid, saturated captcha color (CryptBB glyphs are strongly colored).
    Sample in HSV: any hue, high saturation, mid-low value → solid readable ink."""
    import colorsys
    h = rng.random()
    s = rng.uniform(0.55, 1.0)
    v = rng.uniform(0.45, 0.85)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


class SourceRenderer:
    def __init__(
        self,
        canvas: tuple[int, int],          # (W, H) native, from config
        charset: str,
        len_range: tuple[int, int],
        fonts: list[str] | None = None,
        seed: int | None = None,
        max_rotation: float = 15.0,     # ~real CryptBB; excessive rotation hurts transfer
        fixed_len: int | None = None,
        glyph_style: str = "vivid",     # "vivid" (colored) | "bw" (black/white)
    ):
        self.W, self.H = canvas
        self.charset = charset
        self.lo, self.hi = len_range
        self.fixed_len = fixed_len          # if set, always render this length
        self.glyph_style = glyph_style
        self.fonts = fonts or DEFAULT_FONTS
        self.max_rot = max_rotation
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        self.rng = random.Random(seed)

    def _font(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (path, size)
        if key not in self._font_cache:
            self._font_cache[key] = ImageFont.truetype(path, size)
        return self._font_cache[key]

    def random_text(self) -> str:
        n = self.fixed_len if self.fixed_len is not None else self.rng.randint(self.lo, self.hi)
        return "".join(self.rng.choice(self.charset) for _ in range(n))

    def render_layer(self, text: str | None = None, neutral: bool = False):
        """Render glyphs onto a transparent RGBA layer; return (rgba, mask, text).
        neutral=True -> dark glyphs, no rotation (clean, for the reader)."""
        if text is None:
            text = self.random_text()

        rgba = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        mask = Image.new("L", (self.W, self.H), 0)

        size = self.rng.randint(int(self.H * 0.60), int(self.H * 0.82))
        # one font per word (mix across words gives font variety)
        font_path = self.rng.choice(self.fonts)
        font = self._font(font_path, size)

        # measure per-char advance
        widths = []
        for ch in text:
            box = font.getbbox(ch)
            widths.append(max(box[2] - box[0], size // 3))
        gap = self.rng.randint(2, 6)
        total_w = sum(widths) + gap * (len(text) - 1)
        x = max(2, (self.W - total_w) // 2 + self.rng.randint(-4, 4))

        for i, ch in enumerate(text):
            # render single glyph on its own tight canvas
            pad = size // 2
            gi = Image.new("RGBA", (widths[i] + pad * 2, self.H + pad * 2), (0, 0, 0, 0))
            gm = Image.new("L", gi.size, 0)
            gd = ImageDraw.Draw(gi)
            gmd = ImageDraw.Draw(gm)
            box = font.getbbox(ch)
            gy = (gi.size[1] - (box[3] - box[1])) // 2 - box[1]
            gx = pad - box[0]
            if neutral:
                v = self.rng.randint(0, 55)
                color = (v, v, v, 255)
            elif self.glyph_style == "bw":
                # DarkBay: black or white glyphs (high contrast on dark bg)
                if self.rng.random() < 0.5:
                    v = self.rng.randint(0, 45)
                else:
                    v = self.rng.randint(210, 255)
                color = (v, v, v, 255)
            else:
                color = (*_rand_color(self.rng), 255)
            gd.text((gx, gy), ch, font=font, fill=color)
            gmd.text((gx, gy), ch, font=font, fill=255)

            if not neutral and self.max_rot > 0:
                ang = self.rng.uniform(-self.max_rot, self.max_rot)
                gi = gi.rotate(ang, expand=True, resample=Image.BICUBIC)
                gm = gm.rotate(ang, expand=True, resample=Image.BICUBIC)

            # paste position (center the glyph cell around running x)
            px = x - pad
            py = (self.H - gi.size[1]) // 2 + self.rng.randint(-3, 3)
            rgba.alpha_composite(gi, (px, py))
            mask.paste(gm, (px, py), gm)
            x += widths[i] + gap

        return rgba, mask, text

    def render(self, text: str | None = None, neutral: bool = True):
        """Flatten a (neutral by default) render on white -> (RGB, mask, text)."""
        rgba, mask, text = self.render_layer(text, neutral=neutral)
        bg = Image.new("RGBA", (self.W, self.H), (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB"), mask, text


def _panel(renderer: SourceRenderer, out_path: Path, rows: int = 4, cols: int = 5) -> None:
    pad = 4
    cellw, cellh = renderer.W, renderer.H * 2 + pad
    W = cols * (cellw + pad) + pad
    H = rows * (cellh + pad) + pad
    sheet = Image.new("RGB", (W, H), (40, 40, 40))
    for r in range(rows):
        for c in range(cols):
            img, mask, label = renderer.render(neutral=False)
            x0 = pad + c * (cellw + pad)
            y0 = pad + r * (cellh + pad)
            sheet.paste(img, (x0, y0))
            sheet.paste(mask.convert("RGB"), (x0, y0 + renderer.H + pad))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"wrote panel {out_path}")


if __name__ == "__main__":
    import argparse
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    ap.add_argument("--scheme", default="cryptbb")
    ap.add_argument("--panel", type=Path, default=Path("data/qa_panels/m0_source_renders.png"))
    args = ap.parse_args()

    corpus = yaml.safe_load(args.config.read_text())
    s = corpus["schemes"][args.scheme]
    r = SourceRenderer(canvas=tuple(s["canvas"]), charset=s["charset"],
                       len_range=tuple(s["len_range"]), seed=0)
    for _ in range(10):
        _, _, lbl = r.render()
        print("render:", lbl)
    _panel(r, args.panel)
