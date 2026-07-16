"""CLI: char-extract --image PATH --type T --out DIR

Single-image runner. Exit codes:
  0 — success (mask non-empty, ≥1 char detected)
  1 — no chars detected
  2 — hard error (file missing, decode failure, unknown method, etc.)

Stdout is a one-line JSON summary regardless of --no-write. Designed to be
greppable from shell pipelines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from char_extraction.api import extract
from char_extraction.io_utils import load_image, write_extraction


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="char-extract", description="Extract characters from one CAPTCHA image.")
    p.add_argument("--image", required=True, type=Path, help="Path to image (PNG/JPG).")
    p.add_argument("--type", dest="captcha_type", default=None,
                   help="CAPTCHA type. Resolves a config file under --config-dir. "
                        "Omit to use the universal _default config.")
    p.add_argument("--out", dest="out_dir", type=Path, default=None,
                   help="Write the per-image output layout under this dir. Omit with --no-write.")
    p.add_argument("--config-dir", type=Path, default=None, help="Override config dir.")
    p.add_argument("--method", default=None,
                   help="Override the method name in the resolved config (e.g. universal_sam).")
    p.add_argument("--no-write", action="store_true", help="Skip filesystem output; just print summary.")
    p.add_argument("--id", dest="captcha_id", default=None, help="Override derived captcha_id (default = filename stem).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    t0 = time.perf_counter()
    try:
        if not args.image.exists():
            print(json.dumps({"error": f"file not found: {args.image}"}), file=sys.stderr)
            return 2
        img = load_image(args.image)
        if args.no_write or args.out_dir is None:
            out_dir = None
        else:
            out_dir = args.out_dir
        overrides = {"method": args.method} if args.method else None
        result = extract(
            image=img,
            captcha_type=args.captcha_type,
            config_dir=args.config_dir,
            out_dir=None,  # handle write below so we know paths
            captcha_id=args.captcha_id or args.image.stem,
            overrides=overrides,
        )
        if out_dir is not None:
            write_extraction(result, img, out_dir)
        runtime_ms = int((time.perf_counter() - t0) * 1000)
        summary = {
            "captcha_id": result.captcha_id,
            "captcha_type": result.captcha_type,
            "n_chars": len(result.characters),
            "method": result.method,
            "runtime_ms": runtime_ms,
            "image_size": list(result.image_size),
            "out": str((out_dir / result.captcha_id)) if out_dir else None,
        }
        print(json.dumps(summary))
        if args.verbose:
            for c in result.characters:
                print(f"  char[{c.index}] bbox={c.bbox} conf={c.confidence:.2f}", file=sys.stderr)
        return 0 if result.characters else 1
    except Exception as e:
        print(json.dumps({"error": str(e), "type": type(e).__name__}), file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
