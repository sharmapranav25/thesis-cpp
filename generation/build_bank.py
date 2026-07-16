"""build_bank.py — build a labeled glyph bank for ANY scheme from unlabeled captchas.

This is our self-contained "Part A": segment glyph crops -> DINOv2 embed -> KMeans
cluster -> easyOCR-vote a character per cluster -> write a GlyphBank-compatible
bank (alphabet/cluster_XXX/*.png + alphabet_manifest.json with cluster_to_char).

The real-crop compose path (the chosen glyph strategy) needs one of these per scheme.
auto_synth already shipped banks for cryptbb/city/courier; this builds the missing
ones (e.g. DarkBay). Label-free: easyOCR is a generic pretrained tool, not human labels;
the char we assign a cluster is a model pseudo-label, and OUR rendered text is the
correct-by-construction training label downstream.

  python -m build_bank --dir <pool> --out banks/<scheme> --allow "0-9A-Z" --k 45
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import extract
from font_atlas import _Embedder


def _expand_allow(spec: str) -> str:
    """"0-9A-Z" -> full charset string."""
    out = []
    for m in re.finditer(r"(.)-(.)|(.)", spec):
        if m.group(1):
            for c in range(ord(m.group(1)), ord(m.group(2)) + 1):
                out.append(chr(c))
        else:
            out.append(m.group(3))
    return "".join(out)


def build(pool_dir: str, out_dir: Path, allow: str, k: int = 45, expected=(4, 8),
          max_images: int = 500, min_cluster: int = 8, min_conf: float = 0.30,
          max_per_cluster: int = 40):
    import easyocr
    from sklearn.cluster import KMeans

    allow_chars = _expand_allow(allow)
    paths = sorted(glob.glob(str(Path(pool_dir) / "**" / "*.png"), recursive=True)
                   + glob.glob(str(Path(pool_dir) / "**" / "*.jpg"), recursive=True))[:max_images]
    print(f"[bank] {len(paths)} images; segmenting…")

    crops: list[Image.Image] = []
    for p in paths:
        try:
            crops.extend(extract.char_crops(Image.open(p), expected))
        except Exception:
            pass
    print(f"[bank] {len(crops)} glyph crops; embedding…")
    if len(crops) < k:
        raise SystemExit(f"too few crops ({len(crops)}) for k={k}")

    emb = _Embedder()
    # embed the NORMALISED (black-on-white) crops so clustering keys on glyph shape
    from font_atlas import to_black_on_white
    vecs = emb.embed([to_black_on_white(c) for c in crops])

    print(f"[bank] KMeans k={k}…")
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(vecs)
    labels = km.labels_

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    def ocr_char(im: Image.Image) -> tuple[str, float]:
        arr = np.asarray(to_black_on_white(im, 96).convert("L"))
        res = reader.readtext(arr, allowlist=allow_chars, detail=1, paragraph=False)
        best = ("", 0.0)
        for _, txt, conf in res:
            t = "".join(ch for ch in txt if ch in allow_chars)
            if len(t) == 1 and conf > best[1]:
                best = (t, conf)
        return best

    # per-cluster: keep crops, vote a char over up to 10 members
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    alpha_dir = out_dir / "alphabet"; alpha_dir.mkdir(parents=True)
    cluster_to_char, alphabet_ids = {}, []
    for cid in range(k):
        members = [i for i in range(len(crops)) if labels[i] == cid]
        if len(members) < min_cluster:
            continue
        votes: Counter = Counter()
        for i in members[:12]:
            ch, conf = ocr_char(crops[i])
            if ch and conf >= min_conf:
                votes[ch] += 1
        cdir = alpha_dir / f"cluster_{cid:03d}"; cdir.mkdir()
        for j, i in enumerate(members[:max_per_cluster]):
            crops[i].save(cdir / f"{j:05d}.png")
        alphabet_ids.append(cid)
        if votes:
            ch, n = votes.most_common(1)[0]
            cluster_to_char[str(cid)] = {"char": ch, "votes": n, "total": min(12, len(members))}
    manifest = {
        "alphabet_cluster_ids": alphabet_ids,
        "cluster_to_char": cluster_to_char,
        "source_pool": pool_dir, "k": k, "allow": allow_chars,
    }
    (out_dir / "alphabet_manifest.json").write_text(json.dumps(manifest, indent=2))
    mapped = sorted({v["char"] for v in cluster_to_char.values()})
    print(f"[bank] {len(alphabet_ids)} clusters kept, {len(cluster_to_char)} labeled; "
          f"charset = {''.join(mapped)}  -> {out_dir}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow", default="0-9A-Za-z")
    ap.add_argument("--k", type=int, default=45)
    ap.add_argument("--expected", default="4,8")
    ap.add_argument("--max-images", type=int, default=500)
    args = ap.parse_args()
    build(args.dir, Path(args.out), args.allow, k=args.k,
          expected=tuple(int(x) for x in args.expected.split(",")),
          max_images=args.max_images)
