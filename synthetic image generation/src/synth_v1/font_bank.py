"""CharBank: per-character RGBA crop sampler.

Loads a bank built by build_bank.py and exposes:
  - .sample(ch, rng) -> RGBA crop (PIL.Image), with optional small geometric jitter
  - .available_charset() -> str of chars with >= min_count crops in bank

Design notes:
  - Bank crops are RGBA; alpha is the char silhouette derived from the
    real-vs-LaMa diff mask. Some bg leaks at character edges — acceptable
    for v1; revisit if compose output shows visible bg fragments.
  - Geometric jitter only (small rotation, scale, optional flip-off). Color
    jitter happens later in compose.py / over the bg, not on the char itself —
    we want to preserve real per-char palette.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image


def _folder_to_char(name: str) -> str | None:
    """Inverse of build_bank.char_to_folder. Returns None for non-bank dirs."""
    if len(name) == 1 and name.isdigit():
        return name
    if name.startswith("L_") and len(name) == 3:
        return name[2]
    if name.startswith("U_") and len(name) == 3:
        return name[2]
    return None


class CharBank:
    def __init__(self, bank_dir: Path, min_count: int = 5):
        """`min_count` filters out chars with too few crops to sample reliably.
        These chars will not appear in available_charset() and sample() will
        raise on them — caller must check first."""
        self.bank_dir = Path(bank_dir)
        if not self.bank_dir.is_dir():
            raise FileNotFoundError(self.bank_dir)
        self._index: dict[str, list[Path]] = {}
        for sub in sorted(self.bank_dir.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_"):
                continue
            ch = _folder_to_char(sub.name)
            if ch is None:
                continue
            files = sorted(sub.glob("*.png"))
            if len(files) >= min_count:
                self._index[ch] = files
        if not self._index:
            raise RuntimeError(f"empty bank at {self.bank_dir} (min_count={min_count})")
        self.min_count = min_count

    @classmethod
    def from_meta(cls, bank_dir: Path, min_count: int = 5) -> "CharBank":
        return cls(bank_dir, min_count=min_count)

    def available_charset(self) -> str:
        return "".join(sorted(self._index.keys()))

    def counts(self) -> dict[str, int]:
        return {ch: len(v) for ch, v in self._index.items()}

    def has(self, ch: str) -> bool:
        return ch in self._index

    def sample(self, ch: str, rng: random.Random,
               jitter_rot: float = 4.0,
               jitter_scale: tuple[float, float] = (0.92, 1.10)) -> Image.Image:
        """Return an RGBA crop of `ch` with small rotation+scale jitter applied."""
        if ch not in self._index:
            raise KeyError(f"char {ch!r} not in bank (min_count={self.min_count})")
        path = rng.choice(self._index[ch])
        im = Image.open(path).convert("RGBA")
        # scale
        s = rng.uniform(*jitter_scale)
        if abs(s - 1.0) > 0.01:
            new_w = max(4, int(round(im.width * s)))
            new_h = max(4, int(round(im.height * s)))
            im = im.resize((new_w, new_h), Image.BICUBIC)
        # rotation (expand canvas so corners aren't cut)
        ang = rng.uniform(-jitter_rot, jitter_rot)
        if abs(ang) > 0.1:
            im = im.rotate(ang, resample=Image.BICUBIC, expand=True)
        return im


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.synth_v1.scheme import get
    s = get("cryptbb")
    bank = CharBank(s.bank_dir, min_count=5)
    print(f"available: {bank.available_charset()}")
    print(f"counts: {bank.counts()}")
    rng = random.Random(0)
    sample = bank.sample("M", rng)
    print(f"M sample size: {sample.size}")
