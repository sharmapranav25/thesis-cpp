"""Batch-run the masking pipeline on first N images per type."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from masking.api import extract                # noqa: E402
from masking.io_utils import load_config, write_result  # noqa: E402

SOURCES: dict[str, Path] = {
    "city":    Path(r"E:\pranav\City (live)-20260503T045450Z-3-001\City (live)"),
    "courier": Path(r"E:\pranav\Courier Market (live)-20260503T045531Z-3-001\Courier Market (live)"),
    "cryptbb": Path(r"E:\pranav\CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001\CryptBBCaptchas(unsolved)07272023"),
    "darkbay": Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled"),
}


def run_one_type(captcha_type: str, n: int, out_root: Path, configs_dir: Path, force_color: str) -> dict:
    src = SOURCES[captcha_type]
    images = sorted([p for p in src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])[:n]
    cfg = load_config(captcha_type, configs_dir)
    out_dir = out_root / captcha_type

    summary = {
        "type": captcha_type,
        "n": len(images),
        "path_a": 0,
        "path_b": 0,
        "stage25_iters": [],
        "coverage": [],
        "errors": [],
        "elapsed_s": 0.0,
    }

    t0 = time.time()
    for i, p in enumerate(images, start=1):
        try:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                summary["errors"].append({"file": p.name, "error": "cv2.imread returned None"})
                continue
            res = extract(img, captcha_id=p.stem, captcha_type=captcha_type, cfg=cfg, forced_color=force_color)
            write_result(res, out_dir, img)
            if res.path == "A":
                summary["path_a"] += 1
            else:
                summary["path_b"] += 1
            summary["stage25_iters"].append(res.stage_2_5["iterations"])
            if res.stage_2_5["final_coverage"] is not None:
                summary["coverage"].append(res.stage_2_5["final_coverage"])
            if i % 10 == 0:
                print(f"  [{captcha_type}] {i}/{len(images)} ({time.time()-t0:.1f}s)")
        except Exception as exc:
            summary["errors"].append({"file": p.name, "error": repr(exc)})
    summary["elapsed_s"] = time.time() - t0
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    ap.add_argument("--configs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    ap.add_argument("--force-color", default=None,
                    help="If set, skip Stage 0 and use this colour. Otherwise Qwen runs per image.")
    ap.add_argument("--types", nargs="+", default=list(SOURCES.keys()))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for t in args.types:
        print(f"=== {t} ===")
        s = run_one_type(t, args.n, args.out, args.configs_dir, args.force_color)
        summaries.append(s)
        print(f"  done {t}: A={s['path_a']} B={s['path_b']} errs={len(s['errors'])} t={s['elapsed_s']:.1f}s")

    summary_path = args.out / "batch_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
