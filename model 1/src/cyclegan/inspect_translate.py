"""Load trained CycleGAN and visualize synth->real translations alongside real
captchas. Output: large image with rows = (synth | translated | real)."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torchvision import transforms as T
from PIL import Image

from .models import ResnetGenerator


def main(ckpt_path: Path, n: int, seed: int, scale: int, out_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    G = ResnetGenerator(3, 3, ngf=cfg["ngf"], n_blocks=cfg["n_blocks"]).to(device)
    G.load_state_dict(state["G_AB"])
    G.eval()

    rng = random.Random(seed)
    synth_dir = Path("data/synthetic")
    real_dir = Path("data/raw")
    synth_paths = rng.sample([p for p in synth_dir.glob("*.png")], n)
    real_paths = rng.sample(
        [p for p in real_dir.glob("captcha_screenshot_*.png")], n
    )

    tf = T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    synth_tensors = torch.stack([tf(Image.open(p)) for p in synth_paths]).to(device)
    with torch.inference_mode():
        translated = G(synth_tensors)
    translated = ((translated + 1) / 2).clamp(0, 1).cpu()

    W, H = 200, 60
    pad = 6
    cols = 3
    grid_w = W * cols + pad * (cols + 1)
    grid_h = (H + pad) * n + pad
    canvas = Image.new("RGB", (grid_w, grid_h), (235, 235, 235))

    to_pil = T.ToPILImage()
    for i in range(n):
        y = pad + i * (H + pad)
        canvas.paste(Image.open(synth_paths[i]).convert("RGB"), (pad, y))
        canvas.paste(to_pil(translated[i]), (pad * 2 + W, y))
        canvas.paste(Image.open(real_paths[i]).convert("RGB"), (pad * 3 + 2 * W, y))

    if scale > 1:
        canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.NEAREST)
    canvas.save(out_path)
    print(f"saved {out_path}  (cols: SYNTH | TRANSLATED | REAL)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/cyclegan/inspect.png"))
    args = ap.parse_args()
    main(args.ckpt, args.n, args.seed, args.scale, args.out)
