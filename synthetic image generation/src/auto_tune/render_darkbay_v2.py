"""Config-driven DarkBay text renderer.

Same idea as src/composite/render_darkbay_text.py but reads all knobs from
a RendererConfig instead of hardcoding them. Used by the DINOv2 search loop
so each candidate config can render its own batch without rewriting code.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auto_tune.renderer_config import RendererConfig

CHARSET = "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ"  # no 'I'
N_CHARS = 6
W, H = 200, 50

WIN_FONTS = "C:/Windows/Fonts"
FONT_POOL = [
    "timesbd.ttf",      # 0: Times New Roman Bold (serif)
    "georgiab.ttf",     # 1: Georgia Bold (serif)
    "cambriab.ttf",     # 2: Cambria Bold (serif)
    "consolab.ttf",     # 3: Consolas Bold (mono)
]


def load_fonts_for_config(config: RendererConfig) -> dict[int, list[ImageFont.FreeTypeFont]]:
    fonts: dict[int, list[ImageFont.FreeTypeFont]] = {}
    for sz in range(config.size_min, config.size_max + 1):
        fonts[sz] = []
        for idx in config.font_indices:
            fname = FONT_POOL[idx]
            path = Path(WIN_FONTS) / fname
            if path.exists():
                fonts[sz].append(ImageFont.truetype(str(path), sz))
    return fonts


def random_text(rng: random.Random) -> str:
    return "".join(rng.choice(CHARSET) for _ in range(N_CHARS))


def _render_char(canvas: Image.Image, ch: str, cx: float, cy: float,
                 font: ImageFont.FreeTypeFont, rng: random.Random,
                 boxed: bool, config: RendererConfig) -> tuple[int, int, int, int]:
    angle = rng.uniform(-config.rot_max, config.rot_max)
    bbox = font.getbbox(ch)
    cw = bbox[2] - bbox[0]
    chh = bbox[3] - bbox[1]
    pad = 8
    mini = Image.new("RGBA", (cw + 2 * pad, chh + 2 * pad), (0, 0, 0, 0))
    mdraw = ImageDraw.Draw(mini)
    if boxed:
        bcol = rng.randint(config.box_color_min, config.box_color_max)
        box_color = (bcol, bcol, bcol, 255)
        bx0 = pad - rng.randint(config.box_pad_min, config.box_pad_max)
        by0 = pad - rng.randint(config.box_pad_min, config.box_pad_max)
        bx1 = pad + cw + rng.randint(config.box_pad_min, config.box_pad_max)
        by1 = pad + chh + rng.randint(config.box_pad_min, config.box_pad_max)
        mdraw.rectangle([bx0, by0, bx1, by1], fill=box_color)
        tcol = rng.randint(config.light_text_min, config.light_text_max)
        text_color = (tcol, tcol, tcol, 255)
    else:
        tcol = rng.randint(config.dark_text_min, config.dark_text_max)
        text_color = (tcol, tcol, tcol, 255)
    mdraw.text((pad - bbox[0], pad - bbox[1]), ch, font=font, fill=text_color)
    rot = mini.rotate(angle, resample=Image.BICUBIC, expand=True)
    rw, rh = rot.size
    px = int(round(cx - rw / 2))
    py = int(round(cy - rh / 2))
    canvas.alpha_composite(rot, (px, py))
    return (max(0, px), max(0, py), min(W - 1, px + rw), min(H - 1, py + rh))


def main_cli() -> None:
    """Render N text-only RGBA samples + labels.csv + bboxes.jsonl using a saved config."""
    import argparse, csv, json, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = RendererConfig.from_json(args.config)
    print(f"loaded config from {args.config}")
    args.out.mkdir(parents=True, exist_ok=True)
    fonts = load_fonts_for_config(config)
    if not any(fonts.values()):
        raise RuntimeError("no fonts available")
    rng = random.Random(args.seed)
    t0 = time.time()
    with (args.out / "labels.csv").open("w", newline="") as flbl, \
         (args.out / "bboxes.jsonl").open("w") as fbb:
        w = csv.writer(flbl); w.writerow(["filename", "label"])
        for i in range(args.n):
            text = random_text(rng)
            im, bboxes = render_one(text, config, fonts, rng)
            fn = f"{i:07d}.png"
            im.save(args.out / fn)
            w.writerow([fn, text])
            fbb.write(json.dumps({"filename": fn, "label": text,
                                  "width": W, "height": H, "bboxes": bboxes}) + "\n")
            if (i + 1) % 5000 == 0 or (i + 1) == args.n:
                rate = (i + 1) / (time.time() - t0)
                eta = (args.n - i - 1) / rate if rate else 0
                print(f"  {i + 1}/{args.n}  rate={rate:.0f}/s  eta={eta:.0f}s")
    print(f"done; saved to {args.out}")


def render_one(text: str, config: RendererConfig,
               fonts_by_size: dict, rng: random.Random) -> tuple[Image.Image, list[list[int]]]:
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    spacing = (W / N_CHARS) * config.spacing_mult
    total_width = spacing * N_CHARS
    left_offset = (W - total_width) / 2
    bboxes = []
    image_boxed = rng.random() < config.box_prob
    for i, ch in enumerate(text):
        size = rng.randint(config.size_min, config.size_max)
        font = rng.choice(fonts_by_size[size])
        cx = left_offset + spacing * (i + 0.5) + rng.uniform(-config.pos_jitter_x, config.pos_jitter_x)
        cy = H / 2 + rng.uniform(-config.pos_jitter_y, config.pos_jitter_y)
        bb = _render_char(canvas, ch, cx, cy, font, rng, image_boxed, config)
        bboxes.append(list(bb))
    return canvas, bboxes


if __name__ == "__main__":
    main_cli()
