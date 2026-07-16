"""Download model checkpoints into <repo>/checkpoints/.

Usage:
    .venv\\Scripts\\python scripts\\fetch_checkpoints.py [--mobile-sam] [--craft]

Idempotent — skips files that already exist with non-zero size. Designed to
keep all weights on the same drive as the venv (no fall-through to ~/.cache).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints"

CHECKPOINTS = {
    "mobile_sam": {
        "filename": "mobile_sam.pt",
        "url": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        "expected_min_mb": 35.0,
    },
    "craft": {
        "filename": "craft_mlt_25k.pth",
        "url": "https://github.com/clovaai/CRAFT-pytorch/releases/download/v1.0/craft_mlt_25k.pth",
        "expected_min_mb": 18.0,
    },
}


def _fetch(name: str) -> int:
    info = CHECKPOINTS[name]
    out = CKPT_DIR / info["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > info["expected_min_mb"] * 1e6:
        size_mb = out.stat().st_size / 1e6
        print(f"  [skip] {info['filename']} already at {size_mb:.1f} MB")
        return 0
    print(f"  [fetch] {info['url']}")
    last_pct = -1
    def _hook(blocks, bs, total):
        nonlocal last_pct
        pct = int(blocks * bs * 100.0 / max(total, 1))
        if pct != last_pct and pct % 5 == 0:
            sys.stdout.write(f"\r    {pct}%")
            sys.stdout.flush()
            last_pct = pct
    try:
        urllib.request.urlretrieve(info["url"], out, _hook)
    except Exception as e:
        print(f"\n  [error] {e}")
        return 1
    size_mb = out.stat().st_size / 1e6
    print(f"\n  [done] {info['filename']} = {size_mb:.1f} MB")
    if size_mb < info["expected_min_mb"]:
        print(f"  [warn] file smaller than expected (got {size_mb:.1f} MB, expected ≥ {info['expected_min_mb']} MB)")
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download char_extraction model checkpoints to E: only.")
    p.add_argument("--mobile-sam", action="store_true", help="Fetch MobileSAM ViT-Tiny weights.")
    p.add_argument("--craft", action="store_true", help="Fetch CRAFT MLT weights.")
    p.add_argument("--all", action="store_true", help="Fetch every known checkpoint.")
    args = p.parse_args(argv)

    targets = []
    if args.all or args.mobile_sam:
        targets.append("mobile_sam")
    if args.all or args.craft:
        targets.append("craft")
    if not targets:
        targets = ["mobile_sam"]  # default — minimum to run universal baseline

    print(f"Checkpoint dir: {CKPT_DIR}")
    rc = 0
    for t in targets:
        if t not in CHECKPOINTS:
            print(f"  [skip] unknown checkpoint {t!r}")
            rc = 1
            continue
        print(f"[{t}]")
        rc |= _fetch(t)
    return rc


if __name__ == "__main__":
    sys.exit(main())
