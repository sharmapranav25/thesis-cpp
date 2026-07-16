"""Cold-start the legibility reader R0 (RESEARCH §4).

The off-the-shelf parseq_v2 reader was trained only on MyBB-styled captchas and
CANNOT read our neutral renders (0% — it hallucinates texture). So we bootstrap a
reader on OUR OWN renders. Source-free/label-free: text + labels are ours; we
never read a real captcha.

To be a useful L_rec anchor on the GAN's input distribution, R is trained on the
SAME styled composites the GAN restyles (colored, rotated glyphs on recreated
backgrounds) AND on clean neutral renders (so the M0 clean read-back passes and
the teacher signal is reliable). R is later refreshed on real G(x) during M2.

Init from parseq_v2 weights (encoder transfers) for fast convergence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.render_source import SourceRenderer
from src.background import procedural_bg, LamaBGPool, composite
from src.parseq_reader import ParseqReader, DEFAULT_CKPT, DEFAULT_CFG


class ReaderDS(Dataset):
    """Mix of clean neutral renders and styled composites (the GAN-input look)."""

    def __init__(self, renderer, img_size, length, lama_dir=None,
                 p_clean=0.4, p_procedural=0.5, seed=0):
        self.r = renderer
        self.H, self.W = img_size            # reader size (32,128)
        self.length = length
        self.p_clean = p_clean
        self.p_procedural = p_procedural
        self.seed = seed
        self._rng = None
        self.lama = LamaBGPool(lama_dir, (self.r.W, self.r.H)) if lama_dir else None
        self.tf = T.Compose([
            T.Resize((self.H, self.W), antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def _rng_(self):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng(self.seed * 1000 + wid + 1)
        return self._rng

    def __len__(self):
        return self.length

    def _bg(self, rng):
        if self.lama is None or rng.random() < self.p_procedural:
            return procedural_bg(self.r.W, self.r.H, rng)
        return self.lama.sample(rng)

    def __getitem__(self, i):
        rng = self._rng_()
        if rng.random() < self.p_clean:
            img, mask, label = self.r.render(neutral=True)      # clean white-bg
            pil = img
        else:
            glyph_rgba, mask, label = self.r.render_layer(neutral=False)
            comp = composite(glyph_rgba, self._bg(rng))
            # mild photometric jitter
            arr = comp.astype(np.float32) * rng.uniform(0.9, 1.1)
            arr += rng.normal(0, rng.uniform(0, 5), arr.shape)
            pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return self.tf(pil), label


def collate(b):
    xs, ys = zip(*b)
    return torch.stack(xs, 0), list(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    ap.add_argument("--scheme", default="cryptbb")
    ap.add_argument("--lama-dir", default="E:/pranav/auto_synth/data/cryptbb_unsolved/bg_lama")
    ap.add_argument("--out", type=Path, default=Path("data/ckpts/reader_r0.ckpt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init", default=str(DEFAULT_CKPT), help="init weights; 'none' for scratch")
    args = ap.parse_args()

    corpus = yaml.safe_load(args.config.read_text())
    s = corpus["schemes"][args.scheme]
    renderer = SourceRenderer(canvas=tuple(s["canvas"]), charset=s["charset"],
                              len_range=tuple(s["len_range"]), seed=0)

    init = None if str(args.init).lower() == "none" else args.init
    reader = ParseqReader(ckpt=init if init else DEFAULT_CKPT, cfg=DEFAULT_CFG, train=True)
    model = reader.model
    model.train()
    device = reader.device

    ds = ReaderDS(renderer, reader.img_size, length=args.steps * args.batch,
                  lama_dir=args.lama_dir)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=6,
                        collate_fn=collate, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    step = 0
    running = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        loss = model.training_step((imgs, labels), step)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        running += float(loss); step += 1
        if step % 100 == 0:
            print(f"step {step}/{args.steps}  loss {running/100:.4f}", flush=True)
            running = 0.0
        if step >= args.steps:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"saved reader R0 -> {args.out}")

    # clean read-back check (M0 gate metric) on neutral renders
    model.eval()
    ok = tot = 0
    with torch.inference_mode():
        buf, labs = [], []
        for _ in range(512):
            img, _, l = renderer.render(neutral=True)
            buf.append(ds.tf(img)); labs.append(l)
            if len(buf) == 64:
                x = reader.prep(torch.stack(buf).to(device))
                preds, _ = reader.tok.decode(model(x).softmax(-1))
                for p, gt in zip(preds, labs):
                    tot += 1; ok += int(p == gt)
                buf, labs = [], []
    print(f"clean read-back (greedy): {100*ok/tot:.2f}%  ({ok}/{tot})")


if __name__ == "__main__":
    main()
