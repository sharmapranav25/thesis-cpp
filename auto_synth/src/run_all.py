"""Driver: run Part A (segment + embed + cluster) and compose sample synth
captchas for each scheme. Configures per-scheme settings (canvas size, expected
chars per captcha, k for KMeans).

Run as a single overnight job:
    python -m src.run_all
"""
from __future__ import annotations

from pathlib import Path

from src.part_a.run import main as run_part_a
from src.compose import main as run_compose


SCHEMES = [
    {
        "name": "courier_market",
        "images": Path("E:/pranav/Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)"),
        "force_k": 18,           # digits-only-ish; allow some noise headroom
        "canvas": (200, 60),
        "chars_per": 4,
    },
    {
        "name": "city",
        "images": Path("E:/pranav/City (live)-20260503T045450Z-3-001/City (live)"),
        "force_k": 20,
        "canvas": (324, 81),
        "chars_per": 5,
    },
    {
        "name": "cryptbb_unsolved",
        "images": Path("E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023"),
        "force_k": 50,           # alphanumeric mixed-case, expect ~50 distinct chars
        "canvas": (200, 60),
        "chars_per": 5,
    },
    {
        "name": "joey_retitling",
        "images": Path("E:/pranav/JoeyRetitling_extracted/JoeyRetitling"),
        "force_k": 50,
        "canvas": (200, 60),
        "chars_per": 5,
    },
]

DATA_ROOT = Path("E:/pranav/auto_synth/data")
NB_ROOT = Path("E:/pranav/auto_synth/notebooks")


def main():
    NB_ROOT.mkdir(parents=True, exist_ok=True)
    for sc in SCHEMES:
        print("=" * 70)
        print(f"[run_all] scheme: {sc['name']}")
        print("=" * 70)
        out_dir = DATA_ROOT / sc["name"]
        try:
            run_part_a(
                scheme=sc["name"],
                images_dir=sc["images"],
                out_dir=out_dir,
                max_images=0,
                k_min=14,
                k_max=70,
                min_freq_frac=0.005,
                seed=0,
                force_k=sc["force_k"],
            )
        except Exception as e:
            print(f"[run_all] PART A FAILED for {sc['name']}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Compose sample synth captchas
        try:
            sample_out = NB_ROOT / f"{sc['name']}_synth_samples"
            run_compose(
                scheme_dir=out_dir,
                out_dir=sample_out,
                n=24,
                chars_per=sc["chars_per"],
                canvas_w=sc["canvas"][0],
                canvas_h=sc["canvas"][1],
                bg_dir=None,
                seed=0,
            )
        except Exception as e:
            print(f"[run_all] COMPOSE FAILED for {sc['name']}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
