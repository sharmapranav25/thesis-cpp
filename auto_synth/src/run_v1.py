"""v1 driver: apply length+bank inference, refresh Courier bgs with underline
detection, re-compose all schemes with the leakage-aware sampler.

Assumes Part A (segment+embed+cluster) and OCR mapping have already been run.
This script:
  1. Calls infer_length per scheme -> length-mode JSON
  2. Calls infer_bank per scheme -> populates 'inferred_bank' in manifest
  3. Re-runs Part C ONLY for schemes that need it (Courier, due to wavy
     underline that the v0 mask missed)
  4. Re-runs compose with length modes + leakage-aware sampler
  5. Regenerates compare panels
"""
from __future__ import annotations

from pathlib import Path

from src.part_a.infer_length import infer_length_distribution, length_distribution_to_modes
from src.part_a.infer_bank import infer_bank
from src.part_c.extract_bgs import main as run_part_c
from src.compose import main as run_compose
from src.make_compare_panels import main as run_compare


SCHEMES = [
    {
        "name": "courier_market",
        "images": Path("E:/pranav/Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)"),
        "canvas": (200, 60),
        "redo_part_c": True,         # underline fix
    },
    {
        "name": "city",
        "images": Path("E:/pranav/City (live)-20260503T045450Z-3-001/City (live)"),
        "canvas": (324, 81),
        "redo_part_c": False,
    },
    {
        "name": "cryptbb_unsolved",
        "images": Path("E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023"),
        "canvas": (200, 60),
        "redo_part_c": False,
    },
    {
        "name": "joey_retitling",
        "images": Path("E:/pranav/JoeyRetitling_extracted/JoeyRetitling"),
        "canvas": (200, 60),
        "redo_part_c": False,
    },
]

DATA_ROOT = Path("E:/pranav/auto_synth/data")
NB_ROOT = Path("E:/pranav/auto_synth/notebooks")


def main():
    for sc in SCHEMES:
        print("=" * 70)
        print(f"[v1] {sc['name']}")
        print("=" * 70)
        scheme_dir = DATA_ROOT / sc["name"]

        # 1. Length inference
        print(f"[v1] step 1: infer length")
        try:
            counts = infer_length_distribution(sc["images"], max_images=500)
            modes = length_distribution_to_modes(counts)
            print(f"[v1] modes: {modes}")
        except Exception as e:
            print(f"[v1] length inference failed: {e}")
            modes = [(5, 1.0)]

        # 2. Bank inference
        print(f"[v1] step 2: infer bank")
        try:
            infer_bank(scheme_dir)
        except Exception as e:
            print(f"[v1] bank inference failed: {e}")
            import traceback; traceback.print_exc()

        # 3. Re-run Part C if needed (underline detection)
        if sc["redo_part_c"]:
            print(f"[v1] step 3: re-extract bgs with underline detection")
            try:
                run_part_c(
                    images_dir=sc["images"],
                    out_dir=scheme_dir / "bg_lama",
                    max_images=200,
                    dilate_px=4,
                )
            except Exception as e:
                print(f"[v1] Part C failed: {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"[v1] step 3: skipping Part C (no change needed)")

        # 4. Re-compose synth pool with leakage-aware sampler + length modes
        print(f"[v1] step 4: re-compose synth pool (n=2000)")
        try:
            run_compose(
                scheme_dir=scheme_dir,
                out_dir=scheme_dir / "synth_pool",
                n=2000,
                chars_per=int(modes[0][0]) if modes else 5,
                canvas_w=sc["canvas"][0],
                canvas_h=sc["canvas"][1],
                bg_dir=scheme_dir / "bg_lama",
                seed=2025,
                length_modes=modes,
            )
        except Exception as e:
            print(f"[v1] compose failed: {e}")
            import traceback; traceback.print_exc()

    # 5. Regenerate compare panels
    print("=" * 70)
    print("[v1] step 5: regenerate compare panels")
    print("=" * 70)
    try:
        run_compare()
    except Exception as e:
        print(f"[v1] compare panels failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
