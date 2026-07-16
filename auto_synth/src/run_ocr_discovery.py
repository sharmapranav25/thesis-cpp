"""Driver: run OCR-based discovery for each scheme, then re-compose synth
pools using the discovered alphabets + lengths. Compare panels regenerated.

Each scheme gets a separate output dir (`data/<scheme>_ocr/`) so we can
compare against the CC-based pipeline at `data/<scheme>/` without overwriting.
"""
from __future__ import annotations

from pathlib import Path

from src.part_a.ocr_discover import discover
from src.compose import main as run_compose


SCHEMES = [
    {
        "name": "courier_market",
        "images": Path("E:/pranav/Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)"),
        "canvas": (200, 60),
        "allowlist": "0123456789",
    },
    {
        "name": "city",
        "images": Path("E:/pranav/City (live)-20260503T045450Z-3-001/City (live)"),
        "canvas": (324, 81),
        "allowlist": "0123456789",
    },
    {
        "name": "cryptbb_unsolved",
        "images": Path("E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023"),
        "canvas": (200, 60),
        "allowlist": None,
    },
    {
        "name": "joey_retitling",
        "images": Path("E:/pranav/JoeyRetitling_extracted/JoeyRetitling"),
        "canvas": (200, 60),
        "allowlist": None,
    },
]

DATA_ROOT = Path("E:/pranav/auto_synth/data")


def main():
    for sc in SCHEMES:
        print("=" * 70)
        print(f"[ocr] {sc['name']}")
        print("=" * 70)
        out_dir = DATA_ROOT / f"{sc['name']}_ocr"
        try:
            manifest = discover(
                images_dir=sc["images"],
                out_dir=out_dir,
                max_images=0,
                conf_threshold=0.20,
                max_per_char=50,
                allowlist=sc["allowlist"],
            )
        except Exception as e:
            print(f"[ocr] discovery FAILED for {sc['name']}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Re-compose using OCR-discovered alphabet
        # Bg dir reused from the CC-based pipeline (scheme/bg_lama already exists)
        bg_dir = DATA_ROOT / sc["name"] / "bg_lama"
        try:
            length_modes = [(L, w) for L, w in manifest["inferred_length_modes"]]
            run_compose(
                scheme_dir=out_dir,
                out_dir=out_dir / "synth_pool",
                n=2000,
                chars_per=int(length_modes[0][0]) if length_modes else 5,
                canvas_w=sc["canvas"][0],
                canvas_h=sc["canvas"][1],
                bg_dir=bg_dir if bg_dir.exists() else None,
                seed=4242,
                length_modes=length_modes,
            )
        except Exception as e:
            print(f"[ocr] compose FAILED for {sc['name']}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
