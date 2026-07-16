"""Cluster character embeddings into a discovered alphabet.

Strategy: KMeans with k swept by silhouette + frequency filter to cull noise
clusters. The discovered alphabet is the set of clusters that exceed a
frequency floor (by default 0.5% of crops), so single-image junk doesn't
inflate the alphabet count.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def sweep_k(features: np.ndarray, k_min: int = 12, k_max: int = 80, sample_n: int = 5000,
            seed: int = 0) -> dict:
    """Run KMeans across a range of k, return per-k inertia + silhouette."""
    rng = np.random.default_rng(seed)
    if len(features) > sample_n:
        idx = rng.choice(len(features), sample_n, replace=False)
        sample = features[idx]
    else:
        sample = features
    results = []
    for k in range(k_min, k_max + 1, 2):
        km = KMeans(n_clusters=k, random_state=seed, n_init=4).fit(sample)
        try:
            sil = float(silhouette_score(sample, km.labels_, sample_size=min(2000, len(sample))))
        except Exception:
            sil = float("nan")
        results.append({"k": k, "inertia": float(km.inertia_), "silhouette": sil})
    return {"sweep": results}


def cluster_with_k(features: np.ndarray, k: int, seed: int = 0) -> dict:
    """Final clustering at chosen k. Returns labels + centroids + per-cluster counts."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(features)
    labels = km.labels_
    centroids = km.cluster_centers_
    counts = np.bincount(labels, minlength=k)
    return {"labels": labels, "centroids": centroids, "counts": counts.tolist(), "k": k}


def filter_alphabet(counts: list[int], min_freq_frac: float = 0.005) -> list[int]:
    """Return cluster indices whose frequency exceeds min_freq_frac of total."""
    total = sum(counts)
    threshold = max(3, int(total * min_freq_frac))
    return [i for i, c in enumerate(counts) if c >= threshold]


def pick_k_from_sweep(sweep: list[dict]) -> int:
    """Pick k that maximizes silhouette in the sweep."""
    valid = [s for s in sweep if not np.isnan(s["silhouette"])]
    if not valid:
        return sweep[len(sweep) // 2]["k"]  # middle of sweep as fallback
    best = max(valid, key=lambda s: s["silhouette"])
    return best["k"]
