"""Compose synthetic captchas from Part A's alphabet + Part B's glyph bank.

Minimal v0: pick N random cluster ids from the alphabet, render each as a
glyph, paste onto a sampled background (or solid color if no bg provided),
add tiny jitter on positions. Outputs (image, label_string) where label_string
is the cluster-id sequence — actual character semantics is TBD (cluster_id 5
might be 'A' or '7'; we don't know yet).

Usage:
    python -m src.compose \\
        --scheme-dir data/courier_market \\
        --bg-dir data/courier_market/bg_lama \\   # optional
        --out data/courier_market/synth_samples \\
        --n 20 --chars-per 4
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.part_b.glyph_bank import GlyphBank, cluster_id_to_label


def _pick_length(lengths: list[tuple[int, float]] | None, default: int, rng: random.Random) -> int:
    if not lengths:
        return default
    cum = 0.0
    r = rng.random()
    for L, w in lengths:
        cum += w
        if r <= cum:
            return L
    return lengths[-1][0]


def sample_bg(bg_dir: Path | None, size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Sample a background — either from extracted-bg dir, or fall back to solid white."""
    if bg_dir and bg_dir.exists():
        files = [p for p in bg_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if files:
            f = rng.choice(files)
            try:
                bg = Image.open(f).convert("RGB").resize(size, Image.LANCZOS)
                return bg
            except Exception:
                pass
    # Fallback: light grey
    return Image.new("RGB", size, (245, 245, 245))


def compose_one(bank: GlyphBank, n_chars: int, canvas_size: tuple[int, int],
                bg_dir: Path | None, seed: int) -> tuple[Image.Image, list[str]]:
    rng = random.Random(seed)
    W, H = canvas_size
    bg = sample_bg(bg_dir, (W, H), rng).convert("RGBA")

    # Pick chars from observed_chars (each char is sampled uniformly — no
    # bias from clusters-per-char count, that's the leakage fix).
    chars = rng.choices(bank.observed_chars, k=n_chars)
    rendered = []
    target_h = int(H * 0.7)
    for ch in chars:
        g = bank.render_char(ch, seed=rng.randint(0, 2**31))
        # Resize so glyph height ~ target_h
        scale = target_h / max(1, g.height)
        new_w = max(4, int(g.width * scale))
        new_h = target_h
        g = g.resize((new_w, new_h), Image.LANCZOS)
        rendered.append(bank.to_alpha(g))

    # Layout horizontally with jittered y, equal x slots
    total_w = sum(r.width for r in rendered)
    if total_w >= W:
        # Squeeze: scale all glyphs down uniformly so they fit with margins
        s = (W - 8) / total_w
        rendered = [r.resize((max(4, int(r.width * s)), max(4, int(r.height * s))), Image.LANCZOS) for r in rendered]
        total_w = sum(r.width for r in rendered)

    margin_left = max(2, (W - total_w) // (n_chars + 1))
    x = margin_left
    for r in rendered:
        y = max(0, (H - r.height) // 2 + rng.randint(-3, 3))
        bg.paste(r, (x, y), r)
        x += r.width + margin_left

    return bg.convert("RGB"), chars


def main(scheme_dir: Path, out_dir: Path, n: int, chars_per: int,
         canvas_w: int, canvas_h: int, bg_dir: Path | None, seed: int,
         length_modes: list[tuple[int, float]] | None = None):
    bank = GlyphBank(scheme_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compose] observed chars: {len(bank.observed_chars)} ({''.join(bank.observed_chars)})")
    if length_modes:
        print(f"[compose] length modes: {length_modes}")
    else:
        print(f"[compose] fixed length: {chars_per}")

    label_lines = []
    rng = random.Random(seed)
    for i in range(n):
        L = _pick_length(length_modes, chars_per, rng)
        img, chars = compose_one(bank, L, (canvas_w, canvas_h), bg_dir, seed + i)
        fname = f"synth_{i:04d}.png"
        img.save(out_dir / fname)
        label = "".join(chars)
        label_lines.append(f"{fname},{label}")

    (out_dir / "labels.csv").write_text(
        "filename,label\n" + "\n".join(label_lines) + "\n"
    )
    print(f"[compose] saved {n} samples + labels to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--chars-per", type=int, default=5)
    ap.add_argument("--canvas-w", type=int, default=200)
    ap.add_argument("--canvas-h", type=int, default=60)
    ap.add_argument("--bg-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(
        scheme_dir=args.scheme_dir,
        out_dir=args.out,
        n=args.n,
        chars_per=args.chars_per,
        canvas_w=args.canvas_w,
        canvas_h=args.canvas_h,
        bg_dir=args.bg_dir,
        seed=args.seed,
    )
