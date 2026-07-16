"""Run vlm_alphabet for each scheme to derive a clean alphabet+length+glyph
bank from the Qwen2.5-VL predictions log."""
from __future__ import annotations

from pathlib import Path

from src.part_a.vlm_alphabet import main as run_one


SCHEMES = [
    {"name": "courier_market", "src": Path("E:/pranav/Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)")},
    {"name": "city",            "src": Path("E:/pranav/City (live)-20260503T045450Z-3-001/City (live)")},
    {"name": "cryptbb_unsolved","src": Path("E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023")},
    {"name": "joey_retitling",  "src": Path("E:/pranav/JoeyRetitling_extracted/JoeyRetitling")},
]

DATA_ROOT = Path("E:/pranav/auto_synth/data")


def main():
    for sc in SCHEMES:
        scheme_dir = DATA_ROOT / sc["name"]
        try:
            run_one(scheme_dir, sc["src"])
        except Exception as e:
            print(f"[vlm_alphabet] FAILED for {sc['name']}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
