"""Map cluster IDs -> human-readable characters via a generic OCR.

For each cluster, run easyOCR on a few of its crops and take the modal
prediction. Writes results into the scheme's alphabet_manifest.json under
a new 'cluster_to_char' field.

This is intentionally a separate optional step. Without it, compose.py
labels synth samples with C012-style cluster IDs which is fine for visual
inspection but not for training a recogniser. With it, labels become
real characters and the synth pool becomes usable for downstream training.

Usage:
    python -m src.part_a.map_clusters --scheme-dir data/courier_market
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def _load_easyocr(allowlist: str | None = None):
    import easyocr
    # GPU off if no CUDA; allowlist constrains output to a charset
    return easyocr.Reader(["en"], gpu=True, verbose=False)


def _ocr_one(reader, img: Image.Image, allowlist: str | None) -> str | None:
    """Return single best character or None."""
    arr = np.array(img.convert("RGB"))
    kwargs = {"detail": 0, "paragraph": False, "min_size": 4, "text_threshold": 0.3,
              "low_text": 0.2, "link_threshold": 0.2}
    if allowlist:
        kwargs["allowlist"] = allowlist
    try:
        out = reader.readtext(arr, **kwargs)
    except Exception:
        return None
    # Concatenate, strip, take first non-empty char
    s = "".join(out).strip()
    return s[0] if s else None


def map_clusters(scheme_dir: Path, n_per_cluster: int = 8, allowlist: str | None = None):
    scheme_dir = Path(scheme_dir)
    manifest_path = scheme_dir / "alphabet_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    alpha_dir = scheme_dir / "alphabet"
    cluster_ids = manifest["alphabet_cluster_ids"]

    print(f"[map_clusters] {scheme_dir.name}: {len(cluster_ids)} clusters, OCR per-cluster votes")
    reader = _load_easyocr()

    mapping = {}
    for cid in cluster_ids:
        cdir = alpha_dir / f"cluster_{cid:03d}"
        if not cdir.exists():
            continue
        # Sample n_per_cluster crops (prefer the centroid first)
        files = []
        cent = cdir / "_centroid.png"
        if cent.exists():
            files.append(cent)
        all_crops = sorted([p for p in cdir.iterdir() if p.suffix == ".png" and not p.name.startswith("_")])
        files.extend(all_crops[:n_per_cluster])

        votes = []
        for f in files:
            try:
                im = Image.open(f)
                # Upscale small crops for OCR
                if max(im.size) < 64:
                    s = 64 / max(im.size)
                    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
                ch = _ocr_one(reader, im, allowlist)
                if ch:
                    votes.append(ch)
            except Exception:
                pass

        if votes:
            counts = Counter(votes)
            best, n = counts.most_common(1)[0]
            mapping[str(cid)] = {"char": best, "votes": n, "total": len(votes)}
        else:
            mapping[str(cid)] = {"char": "?", "votes": 0, "total": 0}

    manifest["cluster_to_char"] = mapping
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Friendly summary
    inferred_alphabet = "".join(sorted({v["char"] for v in mapping.values() if v["char"] != "?"}))
    print(f"[map_clusters] inferred alphabet ({len(inferred_alphabet)} unique): '{inferred_alphabet}'")
    return mapping


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme-dir", required=True, type=Path)
    ap.add_argument("--allowlist", type=str, default=None,
                    help="optional restrict charset, e.g. '0123456789' for digits-only schemes")
    ap.add_argument("--n-per-cluster", type=int, default=8)
    args = ap.parse_args()
    map_clusters(args.scheme_dir, n_per_cluster=args.n_per_cluster, allowlist=args.allowlist)
