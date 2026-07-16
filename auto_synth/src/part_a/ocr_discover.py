"""Discover alphabet + length + glyph crops by running pretrained OCR on the
unlabeled images. Replaces the CC+clustering+OCR-on-centroids pipeline for
schemes where the captchas are weak enough that easyOCR can read them.

Outputs a directory layout compatible with GlyphBank, so downstream compose
works without changes:

    <out_dir>/
        alphabet/
            cluster_000/   _centroid.png + N example crops    (cluster_id 0 = first char)
            cluster_001/   ...
            ...
        alphabet_manifest.json
        centroid_grid.png

Manifest fields populated:
    method = "ocr"
    cluster_to_char       — { "0": {"char": "A", "votes": N, "total": N}, ... }
    alphabet_cluster_ids  — list of all cluster ids that have >= min_freq crops
    cluster_counts        — per-cluster crop count
    inferred_length_modes — [(length, weight), ...] sampled from compose
    inferred_bank         — observed + expanded full bank (digits→0-9 etc.)
    n_images_used, n_predictions, mean_ocr_conf
"""
from __future__ import annotations

import argparse
import json
import math
import string
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
DIGITS = set(string.digits)
UPPER = set(string.ascii_uppercase)
LOWER = set(string.ascii_lowercase)


def _interp_char_bbox(bbox, i: int, n: int) -> tuple[int, int, int, int]:
    """Given easyOCR's quad bbox [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] (TL,TR,BR,BL)
    and char index i of n, return an axis-aligned (x1,y1,x2,y2) crop bbox for
    that character — linear interpolation along the top and bottom edges."""
    n = max(1, n)
    (xtl, ytl), (xtr, ytr), (xbr, ybr), (xbl, ybl) = bbox
    # Linear interp factors
    f0 = i / n
    f1 = (i + 1) / n
    # Top edge interp
    xt0 = xtl + (xtr - xtl) * f0
    xt1 = xtl + (xtr - xtl) * f1
    yt0 = ytl + (ytr - ytl) * f0
    yt1 = ytl + (ytr - ytl) * f1
    # Bottom edge interp
    xb0 = xbl + (xbr - xbl) * f0
    xb1 = xbl + (xbr - xbl) * f1
    yb0 = ybl + (ybr - ybl) * f0
    yb1 = ybl + (ybr - ybl) * f1
    # Axis-aligned envelope
    x1 = int(min(xt0, xb0))
    x2 = int(max(xt1, xb1))
    y1 = int(min(yt0, yt1))
    y2 = int(max(yb0, yb1))
    return x1, y1, x2, y2


def _expand_bank(observed: set[str]) -> set[str]:
    bank = set(observed)
    if observed & DIGITS:
        bank |= DIGITS
    if observed & UPPER:
        bank |= UPPER
    if observed & LOWER:
        bank |= LOWER
    return bank


def discover(images_dir: Path, out_dir: Path, max_images: int = 0,
             conf_threshold: float = 0.20, max_per_char: int = 50,
             allowlist: str | None = None,
             pad: int = 2) -> dict:
    import easyocr  # heavy import; defer

    out_dir = Path(out_dir)
    alpha_dir = out_dir / "alphabet"
    alpha_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in Path(images_dir).rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if max_images:
        paths = paths[:max_images]
    print(f"[ocr_discover] {len(paths)} images, conf>={conf_threshold}, "
          f"allowlist={allowlist!r}")

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    char_freqs: Counter[str] = Counter()
    length_freqs: Counter[int] = Counter()
    char_to_crops: dict[str, list[dict]] = {}   # char -> list of {crop, conf, src}
    confs_all = []
    n_predictions = 0

    for k, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
            arr = np.array(img)
        except Exception:
            continue

        kwargs = {
            "detail": 1,
            "paragraph": False,
            "min_size": 4,
            "text_threshold": 0.3,
            "low_text": 0.2,
            "link_threshold": 0.2,
        }
        if allowlist:
            kwargs["allowlist"] = allowlist
        try:
            results = reader.readtext(arr, **kwargs)
        except Exception:
            continue

        # Sort detections left-to-right by their bbox left edge
        try:
            results.sort(key=lambda r: r[0][0][0])
        except Exception:
            pass

        per_image_text = ""
        per_image_conf = []
        for bbox, text, conf in results:
            if conf < conf_threshold or not text:
                continue
            n_predictions += 1
            confs_all.append(float(conf))
            per_image_text += text
            per_image_conf.append(conf)

            for i, ch in enumerate(text):
                x1, y1, x2, y2 = _interp_char_bbox(bbox, i, len(text))
                # Expand by `pad`
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img.width, x2 + pad)
                y2 = min(img.height, y2 + pad)
                if x2 - x1 < 4 or y2 - y1 < 6:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                char_freqs[ch] += 1
                if len(char_to_crops.get(ch, [])) < max_per_char:
                    char_to_crops.setdefault(ch, []).append({
                        "crop": crop,
                        "conf": float(conf),
                        "src": p.name,
                    })

        if per_image_text:
            length_freqs[len(per_image_text)] += 1

        if (k + 1) % 200 == 0:
            print(f"[ocr_discover] {k+1}/{len(paths)} images")

    if not char_freqs:
        raise RuntimeError("OCR discovered no characters at all — try lower conf_threshold or different scheme")

    # Build cluster mapping: each unique char becomes one cluster
    sorted_chars = sorted(char_freqs.keys())
    char_to_cid = {ch: i for i, ch in enumerate(sorted_chars)}

    # Save per-cluster crops
    centroid_imgs = []
    for ch in sorted_chars:
        cid = char_to_cid[ch]
        cdir = alpha_dir / f"cluster_{cid:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        crops = char_to_crops[ch]
        # Pick the highest-conf crop as the centroid
        crops_sorted = sorted(crops, key=lambda c: -c["conf"])
        for j, c in enumerate(crops_sorted):
            c["crop"].save(cdir / f"{j:04d}.png")
        # Centroid = highest-conf crop
        centroid = crops_sorted[0]["crop"]
        centroid.save(cdir / "_centroid.png")
        centroid_imgs.append(centroid)

    # Centroid grid
    cols = 10
    cell = 64
    rows = math.ceil(len(centroid_imgs) / cols)
    grid = Image.new("RGB", (cols * cell, rows * cell), color=(255, 255, 255))
    for i, im in enumerate(centroid_imgs):
        r, c = i // cols, i % cols
        thumb = im.copy().convert("RGB")
        thumb.thumbnail((cell, cell), Image.LANCZOS)
        x = c * cell + (cell - thumb.width) // 2
        y = r * cell + (cell - thumb.height) // 2
        grid.paste(thumb, (x, y))
    grid.save(out_dir / "centroid_grid.png")

    # Length modes
    total_imgs = sum(length_freqs.values())
    length_modes = []
    if total_imgs > 0:
        for L, n in length_freqs.items():
            w = n / total_imgs
            if w >= 0.05 and L >= 1:
                length_modes.append((L, w))
        if not length_modes:
            L, n = max(length_freqs.items(), key=lambda kv: kv[1])
            length_modes = [(L, 1.0)]
        s = sum(w for _, w in length_modes)
        length_modes = [(L, w / s) for L, w in length_modes]
        length_modes.sort()

    # Inferred bank (digits → 0-9, etc.)
    observed = set(sorted_chars)
    full_bank = _expand_bank(observed)
    missing = full_bank - observed

    # Manifest in the same format GlyphBank expects
    cluster_counts = [char_freqs[sorted_chars[i]] for i in range(len(sorted_chars))]
    cluster_to_char = {
        str(i): {"char": ch, "votes": char_freqs[ch], "total": char_freqs[ch]}
        for i, ch in enumerate(sorted_chars)
    }
    alphabet_cluster_ids = list(range(len(sorted_chars)))

    manifest = {
        "scheme": images_dir.name,
        "method": "ocr",
        "n_images_used": len(paths),
        "n_predictions": n_predictions,
        "mean_ocr_conf": float(np.mean(confs_all)) if confs_all else 0.0,
        "alphabet_cluster_ids": alphabet_cluster_ids,
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

    print(f"[ocr_discover] discovered {len(sorted_chars)} unique chars: {''.join(sorted_chars)}")
    print(f"[ocr_discover] inferred bank: {''.join(sorted(full_bank))}")
    print(f"[ocr_discover] missing glyphs: {''.join(sorted(missing))}")
    print(f"[ocr_discover] length modes: {length_modes}")
    print(f"[ocr_discover] mean OCR conf: {manifest['mean_ocr_conf']:.3f}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--conf-threshold", type=float, default=0.20)
    ap.add_argument("--max-per-char", type=int, default=50)
    ap.add_argument("--allowlist", type=str, default=None)
    args = ap.parse_args()
    discover(
        images_dir=args.images,
        out_dir=args.out,
        max_images=args.max_images,
        conf_threshold=args.conf_threshold,
        max_per_char=args.max_per_char,
        allowlist=args.allowlist,
    )
