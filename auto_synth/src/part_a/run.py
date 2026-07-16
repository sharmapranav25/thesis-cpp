"""End-to-end Part A pipeline: scheme dir of unlabeled images -> alphabet.

Steps:
  1. Iterate over images in --images-dir
  2. Segment characters via connected components (segment.crop_characters)
  3. Embed crops with DINOv2 (embed.DinoEmbedder)
  4. KMeans cluster + frequency filter -> alphabet
  5. Save: per-cluster crops, centroid grid PNG, freq JSON, source-image map

Usage:
    python -m src.part_a.run \\
        --scheme courier_market \\
        --images "E:/pranav/Courier Market (live)-...path.../Courier Market (live)" \\
        --out "E:/pranav/auto_synth/data/courier_market"
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.part_a.segment import crop_characters
from src.part_a.embed import DinoEmbedder
from src.part_a import cluster as clu


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(d: Path) -> list[Path]:
    out = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def make_centroid_grid(centroid_imgs: list[Image.Image], counts: list[int], cols: int = 10, cell: int = 64) -> Image.Image:
    rows = math.ceil(len(centroid_imgs) / cols)
    grid = Image.new("RGB", (cols * cell, rows * cell), color=(255, 255, 255))
    for i, im in enumerate(centroid_imgs):
        if im is None:
            continue
        r, c = i // cols, i % cols
        thumb = im.copy().convert("RGB")
        thumb.thumbnail((cell, cell), Image.LANCZOS)
        x = c * cell + (cell - thumb.width) // 2
        y = r * cell + (cell - thumb.height) // 2
        grid.paste(thumb, (x, y))
    return grid


def main(scheme: str, images_dir: Path, out_dir: Path, max_images: int,
         k_min: int, k_max: int, min_freq_frac: float, seed: int,
         force_k: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    alpha_dir = out_dir / "alphabet"
    crops_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    paths = list_images(images_dir)
    rng.shuffle(paths)
    if max_images and len(paths) > max_images:
        paths = paths[:max_images]
    print(f"[part_a] scheme={scheme} images={len(paths)}")

    # Step 1+2: segment all
    all_crops_meta = []  # list of {scheme, src, idx_in_src, bbox}
    all_crop_imgs = []

    for i, p in enumerate(paths):
        try:
            img = Image.open(p)
        except Exception as e:
            print(f"[part_a] skip {p.name}: {e}")
            continue
        crops = crop_characters(img)
        for j, c in enumerate(crops):
            all_crop_imgs.append(c["crop"])
            all_crops_meta.append({"src": p.name, "idx": j, "bbox": c["bbox"]})
        if (i + 1) % 200 == 0:
            print(f"[part_a] segmented {i+1}/{len(paths)} | crops so far: {len(all_crop_imgs)}")

    print(f"[part_a] total crops: {len(all_crop_imgs)}")
    if len(all_crop_imgs) < 100:
        print("[part_a] warning: very few crops; segmentation may have failed for this scheme")

    # Step 3: embed
    print(f"[part_a] embedding...")
    embedder = DinoEmbedder()
    feats = embedder.embed(all_crop_imgs)
    print(f"[part_a] features: {feats.shape}")

    # Step 4: cluster -> sweep k, pick by silhouette (or force k if specified)
    if force_k > 0:
        print(f"[part_a] forcing k={force_k}")
        k = force_k
        sweep = {"sweep": []}
    else:
        print(f"[part_a] sweeping k from {k_min} to {k_max}...")
        sweep = clu.sweep_k(feats, k_min=k_min, k_max=k_max, seed=seed)
        k = clu.pick_k_from_sweep(sweep["sweep"])
        print(f"[part_a] picked k={k} (best silhouette in sweep)")

    result = clu.cluster_with_k(feats, k=k, seed=seed)
    counts = result["counts"]
    alphabet_ids = clu.filter_alphabet(counts, min_freq_frac=min_freq_frac)
    print(f"[part_a] discovered alphabet size: {len(alphabet_ids)} of {k} clusters above {min_freq_frac:.1%} freq floor")

    # Step 5: save outputs
    # 5a: per-cluster crop dirs (limit to ~30 examples per cluster)
    labels = result["labels"]
    cluster_crops = {int(cid): [] for cid in range(k)}
    for idx, lbl in enumerate(labels):
        cluster_crops[int(lbl)].append(idx)

    centroid_imgs = []
    for cid in range(k):
        cdir = alpha_dir / f"cluster_{cid:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        idxs = cluster_crops[cid]
        sample = idxs[:30]
        for idx in sample:
            try:
                all_crop_imgs[idx].save(cdir / f"{idx:06d}.png")
            except Exception:
                pass
        # centroid: nearest crop to mean feature
        if len(idxs) > 0:
            mean_feat = feats[idxs].mean(axis=0, keepdims=True)
            dists = ((feats[idxs] - mean_feat) ** 2).sum(axis=1)
            best = idxs[int(np.argmin(dists))]
            centroid_imgs.append(all_crop_imgs[best])
            all_crop_imgs[best].save(cdir / "_centroid.png")
        else:
            centroid_imgs.append(None)

    # 5b: centroid grid (for visual sanity check)
    grid = make_centroid_grid(centroid_imgs, counts)
    grid.save(out_dir / "centroid_grid.png")

    # 5c: alphabet json (manifest)
    manifest = {
        "scheme": scheme,
        "n_images_used": len(paths),
        "n_crops": len(all_crop_imgs),
        "k_swept_min": k_min,
        "k_swept_max": k_max,
        "k_picked": k,
        "n_clusters_above_floor": len(alphabet_ids),
        "min_freq_frac": min_freq_frac,
        "alphabet_cluster_ids": alphabet_ids,
        "cluster_counts": counts,
        "sweep": sweep["sweep"],
    }
    (out_dir / "alphabet_manifest.json").write_text(json.dumps(manifest, indent=2))

    # 5d: save features for later (e.g., re-clustering with different k)
    np.savez_compressed(
        out_dir / "embeddings.npz",
        features=feats.astype(np.float32),
        labels=labels.astype(np.int32),
        counts=np.asarray(counts, dtype=np.int32),
    )

    print(f"[part_a] done. alphabet manifest at {out_dir/'alphabet_manifest.json'}")
    print(f"[part_a] centroid grid at {out_dir/'centroid_grid.png'}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--k-min", type=int, default=14)
    ap.add_argument("--k-max", type=int, default=70)
    ap.add_argument("--min-freq-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-k", type=int, default=0, help="if >0 skip sweep and use this k")
    args = ap.parse_args()
    main(
        scheme=args.scheme,
        images_dir=args.images,
        out_dir=args.out,
        max_images=args.max_images,
        k_min=args.k_min,
        k_max=args.k_max,
        min_freq_frac=args.min_freq_frac,
        seed=args.seed,
        force_k=args.force_k,
    )
