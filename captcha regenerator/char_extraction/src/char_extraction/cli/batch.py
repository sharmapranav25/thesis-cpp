"""CLI: char-extract-batch --input-dir DIR --type T --out DIR

Run extraction over a directory of images. Sequential by default (avoids
torch/CUDA fork issues); use --workers > 1 for pure-CV methods.

Exit codes:
  0 — ≥95% of images produced ≥1 char
  1 — below threshold
  2 — hard setup error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from char_extraction.api import extract
from char_extraction.io_utils import load_image, write_extraction

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_images(root: Path) -> list:
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS])


def _process_one(args_tuple) -> dict:
    img_path, captcha_type, config_dir, out_dir, resume, method_override = args_tuple
    captcha_id = img_path.stem
    out_subdir = Path(out_dir) / captcha_id
    if resume and (out_subdir / "metadata.json").exists():
        return {"captcha_id": captcha_id, "skipped": True, "n_chars": -1}
    try:
        img = load_image(img_path)
        overrides = {"method": method_override} if method_override else None
        result = extract(
            image=img,
            captcha_type=captcha_type,
            config_dir=Path(config_dir) if config_dir else None,
            out_dir=None,
            captcha_id=captcha_id,
            overrides=overrides,
        )
        write_extraction(result, img, out_dir)
        return {"captcha_id": captcha_id, "n_chars": len(result.characters), "method": result.method}
    except Exception as e:
        return {"captcha_id": captcha_id, "error": f"{type(e).__name__}: {e}", "n_chars": 0}


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="char-extract-batch", description="Run char_extraction on a directory of images.")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--type", dest="captcha_type", default=None)
    p.add_argument("--out", dest="out_dir", required=True, type=Path)
    p.add_argument("--config-dir", type=Path, default=None)
    p.add_argument("--method", default=None, help="Override method (e.g. universal_sam).")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1; safe for torch methods).")
    p.add_argument("--limit", type=int, default=None, help="Process at most N images.")
    p.add_argument("--resume", action="store_true", help="Skip images that already have metadata.json.")
    p.add_argument("--summary", type=Path, default=None, help="Optional JSON summary path.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    try:
        images = _list_images(args.input_dir)
    except FileNotFoundError:
        print(json.dumps({"error": f"input dir not found: {args.input_dir}"}), file=sys.stderr)
        return 2
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(json.dumps({"error": "no images found"}), file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    items = [(p, args.captcha_type, args.config_dir, args.out_dir, args.resume, args.method) for p in images]

    results: list = []
    if args.workers <= 1:
        for it in tqdm(items, desc="extract"):
            results.append(_process_one(it))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_process_one, it) for it in items]
            for f in tqdm(as_completed(futures), total=len(futures), desc="extract"):
                results.append(f.result())

    n_total = len(results)
    n_processed = sum(1 for r in results if not r.get("skipped"))
    n_with_chars = sum(1 for r in results if r.get("n_chars", 0) > 0)
    n_errors = sum(1 for r in results if "error" in r)
    pass_rate = n_with_chars / max(n_processed, 1)
    runtime_s = time.perf_counter() - t0

    summary = {
        "input_dir": str(args.input_dir),
        "out_dir": str(args.out_dir),
        "captcha_type": args.captcha_type,
        "n_total": n_total,
        "n_processed": n_processed,
        "n_with_chars": n_with_chars,
        "n_errors": n_errors,
        "pass_rate": round(pass_rate, 4),
        "runtime_s": round(runtime_s, 2),
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    return 0 if pass_rate >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
