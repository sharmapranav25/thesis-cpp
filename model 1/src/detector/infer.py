"""Run a trained detector on a directory of images.

Loads CharBoxNet, predicts 5 boxes per image (sorted left-to-right by cx),
and either dumps boxes to JSONL (default) or to a numpy array. Used both
for char-CNN training (cropping with detector boxes for augmented training)
and for inference-time cropping in the ensemble.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.detector.model import CharBoxNet, sort_by_cx


def _to_rgb(im):
    return im.convert("RGB")


class ImageDirSimple(torch.utils.data.Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            and not p.name.startswith("_")
        )
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize((60, 200), antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i])), self.paths[i].name


def load_detector(ckpt_path: Path, device: str = "cuda") -> CharBoxNet:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CharBoxNet().to(device).eval()
    model.load_state_dict(state["model"])
    return model


@torch.inference_mode()
def predict_dir(model: CharBoxNet, src: Path, batch: int = 128):
    """Yields (filename, [5 x (x0, y0, x1, y1)]) in absolute pixel coords (200x60)."""
    ds = ImageDirSimple(src)
    loader = DataLoader(ds, batch_size=batch, num_workers=4, pin_memory=True)
    out_w, out_h = 200, 60
    for x, names in loader:
        x = x.cuda(non_blocking=True)
        out = model(x)
        out = sort_by_cx(out)
        out = out.cpu()
        for i, name in enumerate(names):
            boxes_px = []
            for b in out[i]:
                cx, cy, w, h = (v.item() for v in b)
                x0 = (cx - w / 2) * out_w
                y0 = (cy - h / 2) * out_h
                x1 = (cx + w / 2) * out_w
                y1 = (cy + h / 2) * out_h
                boxes_px.append([
                    max(0.0, x0), max(0.0, y0),
                    min(out_w - 1, x1), min(out_h - 1, y1),
                ])
            yield name, boxes_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/eval/_detector_boxes.jsonl"))
    args = ap.parse_args()

    model = load_detector(args.ckpt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as f:
        for name, boxes in predict_dir(model, args.src):
            f.write(json.dumps({"filename": name, "bboxes": boxes}) + "\n")
            n += 1
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
