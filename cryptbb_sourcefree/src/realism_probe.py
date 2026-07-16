"""Deterministic realism probe: how well does a REAL-captcha reader (parseq_v2,
which reads the 651 real at 84% char / 40% word) read OUR synth?

If a reader trained on real reads our synth as well as it reads real, the synth
matches the real distribution. This is a fast, no-training proxy for transfer
(symmetric to "train on synth → read real"). Higher = more real-like.
Reports word + char accuracy on a sample of a pool (labels known by construction).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.parseq_reader import ParseqReader, DEFAULT_CKPT, DEFAULT_CFG
from src.data import to_tensor_pm1


def char_acc(pred, gt):
    n = max(len(pred), len(gt))
    return sum(1 for i in range(n) if i < len(pred) and i < len(gt) and pred[i] == gt[i]), n


def probe(reader, items, load_fn, n=1000, seed=0):
    random.seed(seed); random.shuffle(items); items = items[:n]
    ok = tot = cc = ct = 0
    buf, labs = [], []
    for fn, lbl in items:
        buf.append(load_fn(fn)); labs.append(lbl)
        if len(buf) == 128:
            preds = reader.read(torch.stack(buf), force_len=5)
            for p, l in zip(preds, labs):
                tot += 1; ok += int(p == l)
                a, b = char_acc(p, l); cc += a; ct += b
            buf, labs = [], []
    if buf:
        preds = reader.read(torch.stack(buf), force_len=5)
        for p, l in zip(preds, labs):
            tot += 1; ok += int(p == l); a, b = char_acc(p, l); cc += a; ct += b
    return 100 * ok / tot, 100 * cc / ct, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True,
                    help="pool dirs (each with labels.csv) OR 'real' for the eval set")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    # parseq_v2 = a reader trained on real-ish (MyBB) captchas; refine_iters default 0
    reader = ParseqReader(ckpt=DEFAULT_CKPT, cfg=DEFAULT_CFG, max_label_length=5)

    for pool in args.pools:
        if pool == "real":
            evd = Path(r"E:/pranav/model 1/data/eval")
            labels = {r[0]: r[1] for r in list(csv.reader(open(evd / "labels.csv")))[1:]}
            items = [(f, labels[f]) for f in labels if (evd / f).exists()]
            load = lambda fn: to_tensor_pm1(Image.open(evd / fn).convert("RGB").resize((208, 64)))
        else:
            root = Path(pool)
            items = [(r[0], r[1]) for r in list(csv.reader(open(root / "labels.csv")))[1:]]
            load = lambda fn, root=root: to_tensor_pm1(Image.open(root / fn).convert("RGB").resize((208, 64)))
        w, c, tot = probe(reader, items, load, n=args.n)
        print(f"{pool:50s}  word {w:5.2f}%  char {c:5.2f}%  (n={tot})", flush=True)


if __name__ == "__main__":
    main()
