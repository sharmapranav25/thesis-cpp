from __future__ import annotations

import argparse
from pathlib import Path

from ..api import extract
from ..io_utils import load_config, load_image_bgr, write_result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run masking pipeline on a single CAPTCHA image.")
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--type", required=True, dest="captcha_type")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--id", default=None, help="captcha_id (default: image stem)")
    ap.add_argument(
        "--configs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "configs",
    )
    ap.add_argument(
        "--force-color",
        choices=["red", "blue", "green", "unknown"],
        default=None,
        help="Skip Stage 0 and use this colour. 'unknown' forces Path B.",
    )
    args = ap.parse_args()

    cfg = load_config(args.captcha_type, args.configs_dir)
    image_bgr = load_image_bgr(args.image)
    captcha_id = args.id or args.image.stem

    result = extract(
        image_bgr,
        captcha_id=captcha_id,
        captcha_type=args.captcha_type,
        cfg=cfg,
        forced_color=args.force_color,
    )
    out_dir = write_result(result, args.out, image_bgr)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
