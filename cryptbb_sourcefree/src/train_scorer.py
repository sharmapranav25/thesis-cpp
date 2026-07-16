"""M3 — train the scoring PARSeq on our synth pool, select by synth-val acc.

Why not reuse model 1's trainer directly: (1) its augmentation is tuned for CLEAN
MyBB synth and is far too heavy for our already-distorted GAN synth — it stalls
training (loss stuck ~3.3 vs ~0.9 with light aug); (2) it selects checkpoints by
PARSeq's internal AR `val_accuracy`, which underreports badly (the refine_iters
position-0 quirk) so it never saves good ckpts. We reuse the PARSeq *model*
(build_parseq) and its permutation training_step, but with light aug and clean
force-length synth-val selection (refine_iters=0). Real eval stays untouched
until final scoring.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import math
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

_MODEL1 = Path(r"E:/pranav/model 1")
for p in (str(_MODEL1), str(_MODEL1 / "third_party" / "parseq")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import yaml
import albumentations as A

from src.parseq.train import build_parseq, _to_rgb


def light_aug() -> A.Compose:
    """Mild photometric-only aug — our synth is already geometrically distorted,
    so we DON'T pile on rotation/shear/dropout (that stalls training). Just enough
    color/blur/noise jitter to help the synth→real domain shift."""
    return A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                 A.MotionBlur(blur_limit=3, p=1.0)], p=0.2),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
        A.Affine(rotate=(-2, 2), translate_percent={"x": (-0.02, 0.02), "y": (-0.03, 0.03)},
                 scale=(0.97, 1.03), border_mode=0, fill=255, p=0.4),
    ])


class PoolDS(Dataset):
    def __init__(self, root, rows, img_size, augment):
        self.root = Path(root); self.rows = rows; self.img_size = img_size
        self.aug = light_aug() if augment else None
        self.tf = T.Compose([T.Lambda(_to_rgb), T.Resize(img_size, antialias=True),
                             T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fn, lbl = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        if self.aug is not None:
            img = Image.fromarray(self.aug(image=np.array(img))["image"])
        return self.tf(img), lbl


def collate(b):
    xs, ys = zip(*b); return torch.stack(xs), list(ys)


def force_read(model, imgs, tok, n=5):
    specials = {tok.eos_id, tok.bos_id, tok.pad_id}
    itos = list(tok._itos)
    with torch.inference_mode():
        probs = model(imgs).softmax(-1).cpu()
    out = []
    for i in range(probs.shape[0]):
        s = []
        for t in range(n):
            r = probs[i, t].clone()
            for sp in specials:
                if sp < r.shape[0]:
                    r[sp] = -1.0
            s.append(itos[int(r.argmax())])
        out.append("".join(s))
    return out


def evaluate(model, loader, tok, device):
    model.eval(); model.model.refine_iters = 0
    ok = tot = 0
    for imgs, labs in loader:
        preds = force_read(model, imgs.to(device), tok, n=5)
        for p, l in zip(preds, labs):
            tot += 1; ok += int(p == l)
    model.model.refine_iters = 1
    model.train()
    return 100.0 * ok / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/parseq_pool.yaml"))
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--out", type=Path, default=Path("data/ckpts/parseq_pool/best.ckpt"))
    ap.add_argument("--init", default=None,
                    help="init weights from a ckpt (e.g. parseq_v2) for fast convergence; "
                         "diagnostic only — taints source-free purity (MyBB fonts)")
    ap.add_argument("--mae-init", default=None,
                    help="init PARSeq's ViT encoder from a source-free MAE checkpoint "
                         "(self-supervised on UNLABELED reals) — the font-wall lever")
    ap.add_argument("--eval-real", action="store_true",
                    help="ALSO score the 651 real each epoch (diagnostic; not used for selection)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = "cuda"
    root = Path(cfg["data_dir"])
    img_size = tuple(cfg["img_size"])

    rows = list(csv.reader(open(root / "labels.csv")))[1:]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(rows), generator=g).tolist()
    nval = int(len(rows) * args.val_frac)
    val_rows = [rows[i] for i in perm[:nval]]
    train_rows = [rows[i] for i in perm[nval:]]
    print(f"train={len(train_rows)} val={len(val_rows)}")

    tr = DataLoader(PoolDS(root, train_rows, img_size, True), batch_size=args.batch,
                    shuffle=True, num_workers=6, collate_fn=collate, pin_memory=True,
                    persistent_workers=True, drop_last=True)
    va = DataLoader(PoolDS(root, val_rows, img_size, False), batch_size=256,
                    shuffle=False, num_workers=4, collate_fn=collate)

    model = build_parseq(cfg).to(device)
    if args.mae_init:
        me = torch.load(args.mae_init, map_location=device, weights_only=False)["encoder"]
        missing, unexpected = model.model.encoder.load_state_dict(me, strict=False)
        print(f"[mae-init] loaded encoder from {args.mae_init} "
              f"(missing {len(missing)}, unexpected {len(unexpected)})")
    if args.init:
        st = torch.load(args.init, map_location=device, weights_only=False)
        sd = st["state_dict"] if "state_dict" in st else st
        own = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        for k, v in sd.items():   # partial-copy pos_queries
            if k in own and v.shape != own[k].shape and "pos_queries" in k:
                t = own[k].clone(); n = min(v.shape[1], t.shape[1]); t[:, :n] = v[:, :n]; keep[k] = t
        model.load_state_dict(keep, strict=False)
        print(f"[init] loaded {len(keep)}/{len(sd)} tensors from {args.init}")
    tok = model.tokenizer

    real_imgs = real_labs = None
    if args.eval_real:
        import csv as _csv
        ev = Path(r"E:/pranav/model 1/data/eval")
        rl = {r[0]: r[1] for r in list(_csv.reader(open(ev / "labels.csv")))[1:]}
        files = [f for f in rl if (ev / f).exists()]
        tfm = PoolDS(ev, [(f, rl[f]) for f in files], img_size, False)
        real_imgs = torch.stack([tfm[i][0] for i in range(len(files))])
        real_labs = [rl[f] for f in files]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.get("weight_decay", 0.0))
    total_steps = args.epochs * len(tr)
    warmup = int(0.05 * total_steps)

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best = -1.0
    step = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for imgs, labs in tr:
            loss = model.training_step((imgs.to(device), labs), step)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("gradient_clip_val", 5))
            opt.step(); sched.step()
            run += float(loss); step += 1
        vacc = evaluate(model, va, tok, device)
        racc = ""
        if real_imgs is not None:
            model.eval(); model.model.refine_iters = 0
            ok = 0
            for i in range(0, len(real_labs), 256):
                preds = force_read(model, real_imgs[i:i+256].to(device), tok, n=5)
                ok += sum(int(p == l) for p, l in zip(preds, real_labs[i:i+256]))
            model.model.refine_iters = 1; model.train()
            racc = f"  REAL-651 {100*ok/len(real_labs):.2f}%"
        print(f"epoch {ep:2d}  loss {run/len(tr):.3f}  synth-val(force-len) {vacc:.2f}%{racc}  "
              f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if vacc > best:
            best = vacc
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "synth_val": vacc}, args.out)
            print(f"  saved best -> {args.out} ({vacc:.2f}%)", flush=True)
    print(f"done. best synth-val {best:.2f}%  ckpt {args.out}")


if __name__ == "__main__":
    main()
