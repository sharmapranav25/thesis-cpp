"""Iteration 2 model checks (plan 1D). Uses the smoke cache runs/iter2_smoke3_20261003.

  PYTHONPATH=. python tests/iter2/test_model.py [--overfit]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from feature_bank.iter2 import data as D
from feature_bank.iter2 import losses as Lo
from feature_bank.iter2 import metrics as M
from feature_bank.iter2.model import Iter2Model, param_counts
from feature_bank.iter2.train import to_dev
from feature_bank.train.model import MaskedGroupNorm

SMOKE = "runs/iter2_smoke3_20261003/iter2_h96/smoke"
dev = torch.device("cuda")
R = {}


def batch(s, idx):
    return to_dev(D.collate([(s, i) for i in idx]), dev)


def test_shapes_grads(s, m):
    b = batch(s, range(8))
    o = m(b["image"], b["cols"])
    C = b["image"].shape[-1] // 8
    assert o["eff_presence"].shape == (8, 37) and o["eff_ordinal"].shape == (8, 37, 10)
    assert o["txt_presence"].shape == (8, 16) and o["ctc"].shape == (8, C * m.ctc_frames, 63)
    assert o["font_ordinal"].shape == (8, 4, 10) and o["length"].shape == (8, 7)
    g, _ = Lo.compute(o, b)
    sum(g.values()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    assert not missing, missing
    m.zero_grad()
    R["shapes_grads"] = "ok"


def test_branch_isolation(s, m):
    b = batch(s, range(8))
    o = m(b["image"], b["cols"])
    g, _ = Lo.compute(o, b)
    br = m.branches()
    for loss_names, must_zero in [(["eff_presence", "eff_ordinal"], "text"),
                                  (["ctc", "length", "txt_presence", "txt_ordinal", "font"], "effect")]:
        m.zero_grad()
        sum(g[k] for k in loss_names).backward(retain_graph=True)
        leak = [n for n, p in br[must_zero] if p.grad is not None and p.grad.abs().sum() > 0]
        stem = sum(float(p.grad.abs().sum()) for _, p in br["stem"] if p.grad is not None)
        assert not leak, (loss_names, leak[:5])
        assert stem > 0, "stem gets no gradient"
    m.zero_grad()
    R["branch_isolation"] = "ok (effect losses: no text-branch grad; text losses: no effect-branch grad; stem gets both)"


@torch.no_grad()
def test_padding(s, m):
    m.eval()
    idx = list(range(16))
    widths = s.ix["widths"][idx]
    narrow = int(np.argmin(widths))
    alone = batch(s, [narrow])
    together = batch(s, idx)
    o1 = m(alone["image"], alone["cols"])
    o2 = m(together["image"], together["cols"])
    diff = {}
    for k in ["eff_presence", "eff_ordinal", "txt_presence", "txt_ordinal", "length", "font_cat", "font_ordinal"]:
        diff[k] = float((o1[k][0].float() - o2[k][narrow].float()).abs().max())
    c = int(alone["cols"][0]) * m.ctc_frames
    diff["ctc"] = float((o1["ctc"][0, :c] - o2["ctc"][narrow, :c]).abs().max())
    worst = max(diff.values())
    assert worst < 1e-4, diff
    m.train()
    R["padding"] = f"max diff {worst:.2e}"


def test_receptive_field(m):
    """Stem: gradient of one stride-4 cell w.r.t. the input, GroupNorm statistics detached (local path only)."""
    m.eval()
    MaskedGroupNorm.detach_stats = True
    x = torch.rand(1, 3, 96, 256, device=dev, requires_grad=True)
    cols = torch.tensor([32], device=dev)
    f = m.cnn(x, cols)
    f[0, :, 12, 32].sum().backward()
    g = x.grad[0].abs().sum(0)
    ys, xs = torch.nonzero(g > 0, as_tuple=True)
    MaskedGroupNorm.detach_stats = False
    h, w = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    assert h <= 30 and w <= 30, (h, w)
    m.train()
    R["receptive_field"] = f"{h}x{w} px"


def test_overfit(s, steps=600, ctc_frames=1):
    torch.manual_seed(0)
    m = Iter2Model(dropout=0.0, drop_path=0.0, ctc_frames=ctc_frames).to(dev)
    uw = Lo.UncertaintyWeights().to(dev)
    opt = torch.optim.AdamW(list(m.parameters()) + list(uw.parameters()), lr=1e-3)
    idx = list(range(200))
    bs = [idx[i:i + 50] for i in range(0, 200, 50)]
    cached = [batch(s, b) for b in bs]
    t0 = time.time()
    for st in range(steps):
        b = cached[st % 4]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(b["image"], b["cols"])
        g, _ = Lo.compute(o, b)
        loss = sum(g.values())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    sub = D.Iter2Set(SMOKE)
    sub.ix = {k: v[:200] for k, v in sub.ix.items()}
    P, L = M.predict(m, [sub], dev)
    sc = M.scores(P, L, sub.meta["effect_heads"], sub.meta["text_heads"])
    R["overfit"] = dict(final_losses={k: round(float(v), 4) for k, v in g.items()}, summary=sc["summary"],
                        seconds=round(time.time() - t0))
    print(json.dumps(R, indent=1), flush=True)
    assert sc["summary"]["read_exact"] > 0.9 and sc["summary"]["eff_f1"] > 0.9, sc["summary"]


def main():
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=SMOKE)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--ctc-frames", type=int, default=1)
    a = ap.parse_args()
    SMOKE = a.cache
    s = D.Iter2Set(SMOKE)
    torch.manual_seed(0)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    m = Iter2Model(ctc_frames=a.ctc_frames).to(dev)
    R["params"] = param_counts(m)
    test_shapes_grads(s, m)
    test_branch_isolation(s, m)
    test_padding(s, m)
    test_receptive_field(m)
    if a.overfit:
        test_overfit(s, a.steps, a.ctc_frames)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    sys.exit(main())
