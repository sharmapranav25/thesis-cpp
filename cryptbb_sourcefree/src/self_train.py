"""Progressive self-training on the UNLABELED real pool (StrDA / GeeSolver spirit).

Bootstrap recognizer (trained on source-free synth, ideally MAE-initialized) ->
each round: pseudo-label the UNLABELED reals with TTA-consensus + confidence
filtering, add the confident ones to training (upweighted), fine-tune, repeat.
This adapts the recognizer to the real font WITHOUT human labels (pseudo-labels
only) and WITHOUT the scheme's fonts — scheme-general (swap the unlabeled folder).

GUARDRAIL: self-train on the UNLABELED pool (600 unsolved) ONLY. The 651 eval is
scored for REPORTING each round but NEVER used for pseudo-labels or selection.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

_MODEL1 = Path(r"E:/pranav/model 1")
for p in (str(_MODEL1), str(_MODEL1 / "third_party" / "parseq")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms as T
from PIL import Image
import yaml

from src.parseq.train import build_parseq, _to_rgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train_scorer import PoolDS, light_aug, collate, force_read


def load_labels(p):
    return [(r[0], r[1]) for r in list(csv.reader(open(p)))[1:]]


class RealDS(Dataset):
    """Reals with given (path,label) pseudo-labels + light aug."""
    def __init__(self, root, rows, img_size, augment=True):
        self.root = Path(root); self.rows = rows; self.img_size = img_size
        self.aug = light_aug() if augment else None
        self.tf = T.Compose([T.Lambda(_to_rgb), T.Resize(img_size, antialias=True),
                             T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        fn, lbl = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        if self.aug is not None:
            img = Image.fromarray(self.aug(image=np.array(img))["image"])
        return self.tf(img), lbl


# TTA transforms (post-normalization, spatially aligned so per-pos probs average)
def _tta(x):
    return [x, (x + 0.08).clamp(-1, 1), (x - 0.08).clamp(-1, 1),
            (x * 1.06).clamp(-1, 1), (x * 0.94).clamp(-1, 1)]


@torch.inference_mode()
def pseudo_label(model, tok, real_dir, img_size, device, conf_thr, n_len=5):
    """TTA-averaged force-len decode over all reals. Keep a captcha if every one
    of its n_len chars has averaged confidence >= conf_thr. Returns (rows, stats)."""
    model.eval(); model.model.refine_iters = 0
    paths = sorted(Path(real_dir).glob("*.png"))
    tf = T.Compose([T.Lambda(_to_rgb), T.Resize(img_size, antialias=True),
                    T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    itos = list(tok._itos); specials = {tok.eos_id, tok.bos_id, tok.pad_id}
    rows = []
    confs = []
    B = 128
    for i in range(0, len(paths), B):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+B]]).to(device)
        prob_sum = None
        for v in _tta(batch):
            pr = model(v).softmax(-1)
            prob_sum = pr if prob_sum is None else prob_sum + pr
        probs = (prob_sum / len(_tta(batch))).cpu()
        for j, p in enumerate(paths[i:i+B]):
            chars, cs = [], []
            for t in range(n_len):
                row = probs[j, t].clone()
                for sp in specials:
                    if sp < row.shape[0]:
                        row[sp] = -1.0
                idx = int(row.argmax()); chars.append(itos[idx]); cs.append(float(row[idx]))
            minc = min(cs)
            confs.append(minc)
            if minc >= conf_thr:
                rows.append((p.name, "".join(chars)))
    model.model.refine_iters = 1; model.train()
    return rows, (len(rows), len(paths), float(np.mean(confs)))


@torch.inference_mode()
def eval_real(model, tok, device, img_size):
    evd = Path(r"E:/pranav/model 1/data/eval")
    labels = {r[0]: r[1] for r in list(csv.reader(open(evd / "labels.csv")))[1:]}
    files = [f for f in labels if (evd / f).exists()]
    tf = T.Compose([T.Lambda(_to_rgb), T.Resize(img_size, antialias=True),
                    T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    model.eval(); model.model.refine_iters = 0
    ok = cc = ct = 0
    for i in range(0, len(files), 256):
        batch = torch.stack([tf(Image.open(evd / f).convert("RGB")) for f in files[i:i+256]]).to(device)
        preds = force_read(model, batch, tok, n=5)
        for p, f in zip(preds, files[i:i+256]):
            l = labels[f]; ok += int(p == l)
            for a, b in zip(p.ljust(len(l)), l):
                ct += 1; cc += int(a == b)
    model.model.refine_iters = 1; model.train()
    return 100 * ok / len(files), 100 * cc / ct, len(files)


def train_epochs(model, loaders_weights, opt, device, steps):
    """Round-robin over (loader, weight) — sample proportional to weight."""
    model.train()
    iters = [iter(l) for l, _ in loaders_weights]
    weights = np.array([w for _, w in loaders_weights], float); weights /= weights.sum()
    run = 0.0
    for st in range(steps):
        k = int(np.random.choice(len(iters), p=weights))
        try:
            imgs, labs = next(iters[k])
        except StopIteration:
            iters[k] = iter(loaders_weights[k][0]); imgs, labs = next(iters[k])
        loss = model.training_step((imgs.to(device), labs), st)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); run += float(loss)
    return run / max(1, steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/parseq_pool.yaml"))
    ap.add_argument("--corpus", type=Path, default=Path("configs/corpus.yaml"))
    ap.add_argument("--scheme", default="cryptbb")
    ap.add_argument("--bootstrap", default=None, help="starting recognizer ckpt (state_dict)")
    ap.add_argument("--mae-init", default=None)
    ap.add_argument("--bootstrap-epochs", type=int, default=8, help="synth-only warmup if no --bootstrap")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--round-steps", type=int, default=800)
    ap.add_argument("--conf-thr", type=float, default=0.9)
    ap.add_argument("--real-weight", type=float, default=0.5, help="sampling weight on pseudo-real vs synth")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("data/ckpts/self_train.ckpt"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    corpus = yaml.safe_load(args.corpus.read_text())
    real_dir = corpus["schemes"][args.scheme]["real_dir"]
    device = "cuda"; img_size = tuple(cfg["img_size"])

    model = build_parseq(cfg).to(device)
    tok = model.tokenizer
    if args.mae_init:
        me = torch.load(args.mae_init, map_location=device, weights_only=False)["encoder"]
        model.model.encoder.load_state_dict(me, strict=False)
        print(f"[mae-init] encoder loaded from {args.mae_init}", flush=True)

    # synth loader
    synth_rows = load_labels(Path(cfg["data_dir"]) / "labels.csv")
    synth_ds = PoolDS(cfg["data_dir"], synth_rows, img_size, augment=True)
    synth_loader = DataLoader(synth_ds, batch_size=args.batch, shuffle=True, num_workers=6,
                              collate_fn=collate, pin_memory=True, persistent_workers=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.bootstrap:
        st = torch.load(args.bootstrap, map_location=device, weights_only=False)
        model.load_state_dict(st["state_dict"] if "state_dict" in st else st)
        print(f"[bootstrap] loaded {args.bootstrap}", flush=True)
    else:
        print(f"[bootstrap] warmup {args.bootstrap_epochs} epochs on synth ...", flush=True)
        for ep in range(args.bootstrap_epochs):
            l = train_epochs(model, [(synth_loader, 1.0)], opt, device, len(synth_loader))
            if ep % 2 == 0 or ep == args.bootstrap_epochs - 1:
                w, c, n = eval_real(model, tok, device, img_size)
                print(f"  boot ep{ep}: loss {l:.3f}  REAL word {w:.2f}% char {c:.2f}%", flush=True)

    w, c, n = eval_real(model, tok, device, img_size)
    print(f"[round 0 / bootstrap] REAL word {w:.2f}%  char {c:.2f}%", flush=True)

    best = (w, 0)
    for rnd in range(1, args.rounds + 1):
        rows, (kept, total, mconf) = pseudo_label(model, tok, real_dir, img_size, device, args.conf_thr)
        print(f"[round {rnd}] pseudo-labeled {kept}/{total} reals (conf>={args.conf_thr}, mean-minconf {mconf:.3f})", flush=True)
        loaders = [(synth_loader, 1.0)]
        if kept >= 16:
            real_ds = RealDS(real_dir, rows, img_size, augment=True)
            real_loader = DataLoader(real_ds, batch_size=min(args.batch, kept), shuffle=True,
                                     num_workers=4, collate_fn=collate, pin_memory=True, drop_last=True)
            loaders.append((real_loader, args.real_weight))
        loss = train_epochs(model, loaders, opt, device, args.round_steps)
        w, c, n = eval_real(model, tok, device, img_size)
        print(f"[round {rnd}] loss {loss:.3f}  kept {kept}  REAL word {w:.2f}%  char {c:.2f}%", flush=True)
        # unsupervised selection: #kept pseudo-labels (grows as model adapts)
        if kept > best[1]:
            best = (w, kept)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "round": rnd, "kept": kept, "real_word": w}, args.out)
            print(f"  saved (kept={kept}, real_word={w:.2f}%) -> {args.out}", flush=True)
    print(f"done. best-by-kept real_word {best[0]:.2f}%", flush=True)


if __name__ == "__main__":
    main()
