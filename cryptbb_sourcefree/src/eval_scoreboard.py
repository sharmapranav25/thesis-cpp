"""M3 scoreboard — train PARSeq on the synth pool, score on the 651 real CryptBB.

Thin orchestrator over model 1's proven code:
  train : model 1/src/parseq/train.py  (our configs/parseq_pool.yaml)
  score : model 1/src/eval/case_decode.py  (force-length-5 decode) vs eval labels

Reports the headline word% and writes a row to notes/RESULTS.md context.
This file just shells the two steps so paths/venv are consistent; run with the
model-1 venv from the model-1 dir (its imports are package-relative).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

VENV = r"E:/pranav/model 1/.venv/Scripts/python.exe"
MODEL1 = Path(r"E:/pranav/model 1")
POOL_CFG = Path(r"E:/pranav/cryptbb_sourcefree/configs/parseq_pool.yaml")
EVAL_DIR = Path(r"E:/pranav/model 1/data/eval")
EVAL_LABELS = Path(r"E:/pranav/model 1/data/eval/labels.csv")
POOL_LABELS = Path(r"E:/pranav/cryptbb_sourcefree/data/synth_pool/cryptbb/labels.csv")
CKPT_DIR = Path(r"E:/pranav/cryptbb_sourcefree/data/ckpts/parseq_pool")


def run(cmd, cwd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train PARSeq on the pool")
    ap.add_argument("--score", action="store_true", help="score best ckpt on 651 real")
    ap.add_argument("--ckpt", default=None, help="explicit ckpt to score")
    args = ap.parse_args()

    if args.train:
        run([VENV, "-m", "src.parseq.train", "--config", POOL_CFG], cwd=MODEL1)

    if args.score:
        ckpt = args.ckpt
        if ckpt is None:
            cks = sorted(CKPT_DIR.glob("parseq-*.ckpt"))
            if not cks:
                print("no ckpt found; train first"); sys.exit(1)
            # pick highest val_accuracy in filename
            ckpt = max(cks, key=lambda p: float(p.stem.split("=")[-1]))
        print(f"scoring ckpt: {ckpt}", flush=True)
        run([VENV, "-m", "src.eval.case_decode",
             "--ckpt", ckpt, "--src", EVAL_DIR, "--labels", EVAL_LABELS,
             "--config", POOL_CFG, "--train-labels", POOL_LABELS,
             "--length-threshold", "0.5"], cwd=MODEL1)


if __name__ == "__main__":
    main()
