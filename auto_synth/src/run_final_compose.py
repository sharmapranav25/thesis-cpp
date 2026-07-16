"""Driver: produce a downstream-usable synth pool per scheme.

For each scheme, generate a larger synth set (default 2000) using the
cluster->char mapping (if available) so labels are real characters. Saves
to data/<scheme>/synth_pool/.

Run after Part A, Part C, and run_mapping have finished.
"""
from __future__ import annotations

from pathlib import Path

from src.compose import main as run_compose


SCHEMES = [
    {"name": "courier_market", "canvas": (200, 60), "chars_per": 4, "n": 2000},
    {"name": "city",            "canvas": (324, 81), "chars_per": 5, "n": 2000},
    {"name": "cryptbb_unsolved","canvas": (200, 60), "chars_per": 5, "n": 2000},
    {"name": "joey_retitling",  "canvas": (200, 60), "chars_per": 5, "n": 2000},
]

DATA_ROOT = Path("E:/pranav/auto_synth/data")


def main():
    for sc in SCHEMES:
        print("=" * 70)
        print(f"[final_compose] {sc['name']}: n={sc['n']}, canvas={sc['canvas']}")
        print("=" * 70)
        scheme_dir = DATA_ROOT / sc["name"]
        bg_dir = scheme_dir / "bg_lama"
        out_dir = scheme_dir / "synth_pool"
        try:
            run_compose(
                scheme_dir=scheme_dir,
                out_dir=out_dir,
                n=sc["n"],
                chars_per=sc["chars_per"],
                canvas_w=sc["canvas"][0],
                canvas_h=sc["canvas"][1],
                bg_dir=bg_dir if bg_dir.exists() else None,
                seed=1234,
            )
        except Exception as e:
            print(f"[final_compose] FAILED for {sc['name']}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
