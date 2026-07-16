"""M0 gate: wiring + smoke test.

1. Source renderer produces (clean_img, mask, label) triples.
2. Real Y loader sees the 600 unlabeled CryptBB images.
3. Frozen PARSeq reader reads the CLEAN renders; must exceed 95% word acc
   (else L_rec later has no reliable teacher signal).

Prints a PASS/FAIL gate line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.render_source import SourceRenderer
from src.data import RealDataset, SourceDataset, source_collate
from src.parseq_reader import ParseqReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    ap.add_argument("--scheme", default="cryptbb")
    ap.add_argument("--work", default="64x208", help="HxW GAN working size")
    ap.add_argument("--n", type=int, default=256, help="renders to test reader on")
    ap.add_argument("--reader-ckpt", default="data/ckpts/reader_r0.ckpt",
                    help="cold-started R0 (source-free). 'v2' = off-the-shelf parseq_v2")
    args = ap.parse_args()

    H, W = (int(v) for v in args.work.lower().split("x"))
    corpus = yaml.safe_load(args.config.read_text())
    s = corpus["schemes"][args.scheme]

    # 1. renderer
    renderer = SourceRenderer(
        canvas=tuple(s["canvas"]), charset=s["charset"],
        len_range=tuple(s["len_range"]), seed=0,
    )
    src_ds = SourceDataset(renderer, (H, W), length=args.n)
    print(f"[1] renderer OK — {args.n} triples, canvas {tuple(s['canvas'])}, work {H}x{W}")

    # 2. real loader
    real_ds = RealDataset(s["real_dir"], (H, W))
    real_loader = DataLoader(real_ds, batch_size=16, shuffle=True)
    batch = next(iter(real_loader))
    print(f"[2] real Y loader OK — {len(real_ds)} unlabeled imgs, batch {tuple(batch.shape)}, "
          f"range [{batch.min():.2f},{batch.max():.2f}]")

    # 3. reader reads CLEAN neutral renders (the gate's intent)
    if args.reader_ckpt == "v2":
        reader = ParseqReader()
    else:
        reader = ParseqReader(ckpt=args.reader_ckpt)
    import torch
    from src.data import to_tensor_pm1
    cg = cf = total = 0
    buf, labs = [], []
    for _ in range(args.n):
        img, _, l = renderer.render(neutral=True)
        buf.append(to_tensor_pm1(img.resize((W, H)))); labs.append(l)
        if len(buf) == 64:
            xb = torch.stack(buf)
            greedy = reader.read(xb)
            forced = reader.read(xb, lengths=[len(x) for x in labs])
            for pg, pf, gt in zip(greedy, forced, labs):
                total += 1; cg += int(pg == gt); cf += int(pf == gt)
            buf, labs = [], []
    acc_g = 100.0 * cg / max(1, total)
    acc_f = 100.0 * cf / max(1, total)
    print(f"[3] reader read-back on clean renders: force-len {acc_f:.2f}%  greedy {acc_g:.2f}%  (n={total})")

    # Gate on force-length decode — the operative decode used downstream for
    # L_rec verify, pool verify, and final scoring (how model 1 reached 57%).
    gate = acc_f > 95.0
    print(f"\nM0 GATE: reader (force-len) >95% on clean renders -> "
          f"{'PASS' if gate else 'FAIL'} ({acc_f:.2f}%)")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
