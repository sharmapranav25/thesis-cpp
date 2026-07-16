"""SchemeSpec: per-captcha-scheme constants used by every v1 module.

Adding a new scheme = one new instance below + the scheme's data on disk
(unlabeled real samples + bg pool + char bank built by build_bank.py).

Only CryptBB is wired now because it's the only scheme with a working
detector + recognizer pair (both trained on CryptBB) and a clean disjoint
unlabeled pool. DarkBay/Courier/City need their own detector/recognizer
before bank-building can run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL1 = Path(r"E:\pranav\model 1")


@dataclass(frozen=True)
class SchemeSpec:
    name: str
    W: int
    H: int
    n_chars: int
    charset: str
    raw_dir: Path                    # unlabeled real samples (disjoint from eval)
    bg_dir: Path                     # LaMa-extracted bg pool
    bank_dir: Path                   # output of build_bank.py


CRYPTBB = SchemeSpec(
    name="cryptbb",
    W=200,
    H=60,
    n_chars=5,
    charset="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    raw_dir=MODEL1 / "data" / "raw",
    bg_dir=ROOT / "data" / "cryptbb_bg_lama",
    bank_dir=ROOT / "data" / "synth_v1" / "banks" / "cryptbb",
)


SCHEMES: dict[str, SchemeSpec] = {
    CRYPTBB.name: CRYPTBB,
}


def get(name: str) -> SchemeSpec:
    if name not in SCHEMES:
        raise KeyError(f"unknown scheme {name!r}; have {sorted(SCHEMES)}")
    return SCHEMES[name]
