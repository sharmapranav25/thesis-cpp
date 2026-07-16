"""Run a trained Model 3 PARSeq checkpoint over a directory of images.
Emits predictions.csv with (filename, prediction, conf_joint, conf_min, conf_mean) rows.

Forked from M2's infer.py but uses M3's `build_model` (PARSeqWithLossLog,
which has the same forward path as vanilla PARSeq — the loss-capture only
fires inside training_step).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

# Reuse M2's vendored PARSeq tree.
_M3_ROOT = Path(__file__).resolve().parents[2]
_PARSEQ_CANDIDATES = [
    _M3_ROOT.parent / "model 2" / "third_party" / "parseq",
    _M3_ROOT.parent / "model 1" / "third_party" / "parseq",
    _M3_ROOT / "third_party" / "parseq",
]
PARSEQ_ROOT = next((p for p in _PARSEQ_CANDIDATES if p.exists()), None)
if PARSEQ_ROOT is None:
    raise RuntimeError("Could not locate third_party/parseq")
sys.path.insert(0, str(PARSEQ_ROOT))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
import yaml

from src.parseq.train import build_model, _to_rgb


class ImageDir(Dataset):
    def __init__(self, root: Path, img_size: tuple[int, int]):
        self.paths = sorted(p for p in root.glob("*.png") if not p.name.startswith("_"))
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i])), self.paths[i].name


def main(ckpt_path: Path, src_dir: Path, out_csv: Path, cfg_path: Path, batch: int) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model = model.cuda().eval()

    ds = ImageDir(src_dir, tuple(cfg["img_size"]))
    loader = DataLoader(ds, batch_size=batch, num_workers=4, pin_memory=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction", "conf_joint", "conf_min", "conf_mean"])
        with torch.inference_mode():
            for imgs, names in tqdm(loader, desc="infer"):
                imgs = imgs.cuda(non_blocking=True)
                logits = model(imgs)
                preds, probs = model.tokenizer.decode(logits.softmax(-1))
                for name, p, prob in zip(names, preds, probs):
                    if len(p) == 0:
                        cj = cmin = cmean = 0.0
                    else:
                        char_probs = prob[: len(p)]
                        cj = float(char_probs.prod().item())
                        cmin = float(char_probs.min().item())
                        cmean = float(char_probs.mean().item())
                    writer.writerow([name, p, f"{cj:.6f}", f"{cmin:.6f}", f"{cmean:.6f}"])
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/eval/predictions.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/parseq_adaptive.yaml"))
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    main(args.ckpt, args.src, args.out, args.config, args.batch)
