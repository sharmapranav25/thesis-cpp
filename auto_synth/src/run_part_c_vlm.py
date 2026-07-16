"""Driver: run VLM-driven (Qwen2.5-VL + SAM + LaMa) bg extraction for each scheme."""
from __future__ import annotations

from pathlib import Path

from src.part_c.extract_bgs_vlm import main as run_vlm


SCHEMES = [
    {"name": "courier_market", "images": Path("E:/pranav/Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)")},
    {"name": "city",            "images": Path("E:/pranav/City (live)-20260503T045450Z-3-001/City (live)")},
    {"name": "cryptbb_unsolved","images": Path("E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023")},
    {"name": "joey_retitling",  "images": Path("E:/pranav/JoeyRetitling_extracted/JoeyRetitling")},
]

DATA_ROOT = Path("E:/pranav/auto_synth/data")
LIMIT = 100   # 100/scheme to keep wall time bounded; raise to 200 if quality is good


def main():
    for sc in SCHEMES:
        print("=" * 70)
        print(f"[vlm_bg] scheme: {sc['name']}")
        print("=" * 70)
        out_dir = DATA_ROOT / sc["name"] / "bg_lama_vlm"
        try:
            run_vlm(
                images_dir=sc["images"],
                out_dir=out_dir,
                max_images=LIMIT,
                dilate_px=4,
            )
        except Exception as e:
            print(f"[vlm_bg] FAILED for {sc['name']}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
