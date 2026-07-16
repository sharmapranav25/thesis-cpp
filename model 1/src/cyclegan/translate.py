"""After CycleGAN training, push the entire synthetic pool through G_AB to
produce a 'real-styled' training set for PARSeq. Labels are carried over from
the synthetic labels.csv unchanged (the generator preserves text content)."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from .models import ResnetGenerator


def _to_rgb(im):
    return im.convert("RGB")


class SynthDir(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(root.glob("*.png"))
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        return self.tf(Image.open(p)), p.name


def main(ckpt_path: Path, synth_dir: Path, out_dir: Path, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    G = ResnetGenerator(3, 3, ngf=cfg["ngf"], n_blocks=cfg["n_blocks"]).to(device)
    G.load_state_dict(state["G_AB"])
    G.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = SynthDir(synth_dir)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    # carry labels.csv over verbatim — content is preserved
    src_labels = synth_dir / "labels.csv"
    if src_labels.exists():
        shutil.copy2(src_labels, out_dir / "labels.csv")
    else:
        print("WARN: no labels.csv in", synth_dir)

    with torch.inference_mode():
        for batch, names in tqdm(loader, desc="translate"):
            batch = batch.to(device, non_blocking=True)
            out = G(batch)
            out = (out + 1) / 2
            for img, name in zip(out, names):
                save_image(img, out_dir / name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--synth", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--out", type=Path, default=Path("data/translated"))
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    main(args.ckpt, args.synth, args.out, args.batch)
