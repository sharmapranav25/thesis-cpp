"""Extract alphabet + length distribution from Qwen2.5-VL predictions.

Reads vlm_predictions.jsonl produced by part_c/extract_bgs_vlm.py and
aggregates:
  - char_freqs: how often each char was identified across the corpus
  - length_freqs: per-image label-length histogram
  - per_char_crops: optional, save crops indexed by char (uses VLM bboxes
    directly — same as Part A but without CC + DINO + KMeans)

Then writes an alphabet_manifest.json compatible with GlyphBank, so compose
can immediately use the VLM-discovered alphabet.
"""
from __future__ import annotations

import argparse
import json
import math
import string
from collections import Counter
from pathlib import Path

from PIL import Image


DIGITS = set(string.digits)
UPPER = set(string.ascii_uppercase)
LOWER = set(string.ascii_lowercase)


def _expand_bank(observed):
    bank = set(observed)
    if observed & DIGITS:
        bank |= DIGITS
    if observed & UPPER:
        bank |= UPPER
    if observed & LOWER:
        bank |= LOWER
    return bank


def main(scheme_dir: Path, source_images_dir: Path, max_per_char: int = 50):
    scheme_dir = Path(scheme_dir)
    vlm_log = scheme_dir / "bg_lama_vlm" / "vlm_predictions.jsonl"
    if not vlm_log.exists():
        raise FileNotFoundError(f"no vlm_predictions.jsonl at {vlm_log}")

    out_dir = Path(str(scheme_dir) + "_vlm")
    alpha_dir = out_dir / "alphabet"
    alpha_dir.mkdir(parents=True, exist_ok=True)

    char_freqs: Counter[str] = Counter()
    length_freqs: Counter[int] = Counter()
    char_to_crops: dict[str, list[Image.Image]] = {}

    with vlm_log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            preds = row.get("preds", [])
            if not preds:
                continue
            length_freqs[len(preds)] += 1
            src_name = row["src"]
            try:
                img = Image.open(source_images_dir / src_name).convert("RGB")
            except Exception:
                continue
            for pr in preds:
                ch = pr.get("label") or ""
                if not ch or len(ch) != 1:
                    continue
                char_freqs[ch] += 1
                if len(char_to_crops.get(ch, [])) < max_per_char:
                    x1, y1, x2, y2 = pr["bbox"]
                    x1 = max(0, x1 - 2); y1 = max(0, y1 - 2)
                    x2 = min(img.width, x2 + 2); y2 = min(img.height, y2 + 2)
                    if x2 - x1 >= 4 and y2 - y1 >= 6:
                        crop = img.crop((x1, y1, x2, y2))
                        char_to_crops.setdefault(ch, []).append(crop)

    if not char_freqs:
        raise RuntimeError("No characters found in VLM predictions")

    sorted_chars = sorted(char_freqs.keys())
    char_to_cid = {ch: i for i, ch in enumerate(sorted_chars)}

    centroid_imgs = []
    for ch in sorted_chars:
        cid = char_to_cid[ch]
        cdir = alpha_dir / f"cluster_{cid:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        crops = char_to_crops.get(ch, [])
        for j, c in enumerate(crops):
            c.save(cdir / f"{j:04d}.png")
        if crops:
            crops[0].save(cdir / "_centroid.png")
            centroid_imgs.append(crops[0])
        else:
            centroid_imgs.append(None)

    # Centroid grid
    cols = 10
    cell = 64
    rows = math.ceil(len(centroid_imgs) / cols)
    grid = Image.new("RGB", (cols * cell, max(1, rows) * cell), color=(255, 255, 255))
    for i, im in enumerate(centroid_imgs):
        if im is None:
            continue
        r, c = i // cols, i % cols
        thumb = im.copy().convert("RGB")
        thumb.thumbnail((cell, cell), Image.LANCZOS)
        x = c * cell + (cell - thumb.width) // 2
        y = r * cell + (cell - thumb.height) // 2
        grid.paste(thumb, (x, y))
    grid.save(out_dir / "centroid_grid.png")

    # Length modes (with plausibility filter)
    total = sum(length_freqs.values())
    plausible = {L: n for L, n in length_freqs.items() if L >= 3}
    pl_total = sum(plausible.values()) or 1
    modes_raw = [(L, n / pl_total) for L, n in plausible.items() if n / pl_total >= 0.05]
    if not modes_raw and length_freqs:
        L, n = max(length_freqs.items(), key=lambda kv: kv[1])
        modes_raw = [(L, 1.0)]
    s = sum(w for _, w in modes_raw) or 1
    length_modes = sorted([(L, w / s) for L, w in modes_raw])

    observed = set(sorted_chars)
    full_bank = _expand_bank(observed)
    missing = full_bank - observed

    cluster_counts = [char_freqs[ch] for ch in sorted_chars]
    cluster_to_char = {
        str(i): {"char": ch, "votes": char_freqs[ch], "total": char_freqs[ch]}
        for i, ch in enumerate(sorted_chars)
    }

    manifest = {
        "scheme": scheme_dir.name + "_vlm",
        "method": "vlm",
        "n_predictions": sum(char_freqs.values()),
        "alphabet_cluster_ids": list(range(len(sorted_chars))),
        "cluster_counts": cluster_counts,
        "cluster_to_char": cluster_to_char,
        "inferred_length_modes": [[L, w] for L, w in length_modes],
        "length_freq_raw": dict(length_freqs),
        "inferred_bank": {
            "observed_chars": sorted(observed),
            "inferred_full_bank": sorted(full_bank),
            "missing_glyphs": sorted(missing),
            "n_observed": len(observed),
            "n_inferred": len(full_bank),
            "n_missing": len(missing),
        },
    }
    (out_dir / "alphabet_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[vlm_alphabet] {scheme_dir.name}:")
    print(f"  observed ({len(observed)}): {''.join(sorted(observed))}")
    print(f"  inferred bank ({len(full_bank)}): {''.join(sorted(full_bank))}")
    print(f"  length modes: {length_modes}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme-dir", required=True, type=Path,
                    help="e.g. data/courier_market — vlm_predictions.jsonl is read from <scheme-dir>/bg_lama_vlm/")
    ap.add_argument("--source-images", required=True, type=Path)
    args = ap.parse_args()
    main(args.scheme_dir, args.source_images)
