"""Driver: run Part C (LaMa bg extraction) for all schemes, then re-compose
synth captchas using real-extracted backgrounds.

Limits to 200 images per scheme to keep wall time bounded (~13 min total at
2 sec/image) — 200 unique bgs is plenty of variety for compose-time sampling.
"""
from __future__ import annotations

from pathlib import Path

from src.part_c.extract_bgs import main as run_part_c
from src.compose import main as run_compose


SCHEMES = [
    {
        "name": "courier_market",
        "images": Path("E:/pranav/Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)"),
        "canvas": (200, 60),
        "chars_per": 4,
    },
    {
        "name": "city",
        "images": Path("E:/pranav/City (live)-20260503T045450Z-3-001/City (live)"),
        "canvas": (324, 81),
        "chars_per": 5,
    },
    {
        "name": "cryptbb_unsolved",
        "images": Path("E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023"),
        "canvas": (200, 60),
        "chars_per": 5,
    },
    {
        "name": "joey_retitling",
        "images": Path("E:/pranav/JoeyRetitling_extracted/JoeyRetitling"),
        "canvas": (200, 60),
        "chars_per": 5,
    },
]

DATA_ROOT = Path("E:/pranav/auto_synth/data")
NB_ROOT = Path("E:/pranav/auto_synth/notebooks")
BG_LIMIT = 200


def main():
    for sc in SCHEMES:
        print("=" * 70)
        print(f"[part_c] scheme: {sc['name']}")
        print("=" * 70)
        bg_dir = DATA_ROOT / sc["name"] / "bg_lama"
        try:
            run_part_c(
                images_dir=sc["images"],
                out_dir=bg_dir,
                max_images=BG_LIMIT,
                dilate_px=4,
            )
        except Exception as e:
            print(f"[part_c] FAILED for {sc['name']}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Re-compose with real bgs
        try:
            sample_out = NB_ROOT / f"{sc['name']}_synth_with_bg"
            run_compose(
                scheme_dir=DATA_ROOT / sc["name"],
                out_dir=sample_out,
                n=24,
                chars_per=sc["chars_per"],
                canvas_w=sc["canvas"][0],
                canvas_h=sc["canvas"][1],
                bg_dir=bg_dir,
                seed=42,
            )
        except Exception as e:
            print(f"[part_c] re-compose FAILED for {sc['name']}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
