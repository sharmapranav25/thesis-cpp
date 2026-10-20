"""Interpretable distribution descriptors, optional frozen DINOv2, and coverage."""
from __future__ import annotations

import numpy as np
from PIL import Image


def describe(image):
    a = np.asarray(image.resize((128, 48), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    edge = np.sqrt(gx * gx + gy * gy)
    colors = np.concatenate([np.histogram(a[:, :, c], bins=8, range=(0, 1))[0] / gray.size
                             for c in range(3)])
    color_stats = np.concatenate([np.quantile(a, [0.1, 0.5, 0.9], axis=(0, 1)).ravel(),
                                  [float((a.max(2) - a.min(2)).mean())]])
    gradients = np.concatenate([np.histogram(e, bins=8, range=(0, 1))[0] / gray.size
                                for e in (gx, gy)])
    layout = np.concatenate([edge.reshape(6, 8, 8, 16).mean((1, 3)).ravel(),
                             [float(edge.mean()), float(edge.std())]])
    # Autocorrelation responds to repeating grid/noise without trusting character segmentation.
    centered = gray - gray.mean()
    var = float(np.mean(centered ** 2)) + 1e-5
    texture = np.array([np.mean(centered[:, :-lag] * centered[:, lag:]) / var
                        for lag in (1, 2, 4, 8, 12, 16)] +
                       [np.mean(centered[:-lag] * centered[lag:]) / var for lag in (1, 2, 4, 8)])
    vec = np.concatenate([colors, color_stats, gradients, layout, texture]).astype(np.float32)
    return vec


def _color_modes(pixels, count):
    bins = pixels.astype(np.int32) // 32
    keys = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    unique, counts = np.unique(keys, return_counts=True)
    order = np.argsort(counts)[::-1][:count]
    # Each color comes from one occupied RGB bin, not independent channel quantiles
    # across unrelated red/green/blue glyphs (which would invent gray colors).
    return [np.median(pixels[keys == unique[i]], axis=0).astype(int).tolist() for i in order]


def palette_profile(image, opaque_id):
    a = np.asarray(image, dtype=np.float32)
    # Thin frames must not dominate a smooth-pixel heuristic. Estimate color modes
    # inside a small generic margin, retaining native pixels to avoid color mixing.
    margin = max(1, round(min(image.size) * 0.04))
    pixels = a[margin:-margin, margin:-margin].reshape(-1, 3)
    base = np.array(_color_modes(pixels, 1)[0])
    distance = np.linalg.norm(pixels - base, axis=1)
    background_pixels = pixels[distance <= np.quantile(distance, 0.65)]
    background = _color_modes(background_pixels, 5)
    fg = pixels[distance >= max(20, float(np.quantile(distance, 0.80)))]
    if len(fg) < 4:
        fg = pixels
    foreground = _color_modes(fg, 6)
    return {"id": opaque_id, "width": image.width, "height": image.height,
            "background": background, "foreground": foreground}


class FeatureExtractor:
    def __init__(self, kind="pixels", cache_dir=None, allow_download=False):
        self.kind = kind
        self.model = self.processor = None
        self.groups = {"color": (0, 34), "edges": (34, 50), "layout": (50, 100), "texture": (100, 110)}
        if kind == "dinov2":
            import torch
            from transformers import AutoImageProcessor, AutoModel
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs = {"cache_dir": cache_dir, "local_files_only": not allow_download}
            name = "facebook/dinov2-small"
            self.processor = AutoImageProcessor.from_pretrained(name, **kwargs)
            self.model = AutoModel.from_pretrained(name, **kwargs).to(self.device).eval()
            self.groups["dinov2"] = (110, 110 + self.model.config.hidden_size)
        elif kind != "pixels":
            raise ValueError(f"Unknown feature extractor: {kind}")

    def batch(self, images):
        vectors = np.stack([describe(image) for image in images])
        if self.model is not None:
            import torch
            embeddings = []
            # Letterbox preserves the full word before the processor's square resize/crop.
            for start in range(0, len(images), 8):
                padded = []
                for image in images[start:start + 8]:
                    side = max(image.size)
                    square = Image.new("RGB", (side, side), (127, 127, 127))
                    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
                    padded.append(square)
                inputs = self.processor(images=padded, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    patches = self.model(**inputs).last_hidden_state[:, 1:]
                    embedding = patches.mean(1)
                    embedding = torch.nn.functional.normalize(embedding, dim=-1)
                embeddings.append(embedding.float().cpu().numpy())
            vectors = np.concatenate([vectors, np.concatenate(embeddings)], axis=1)
        return vectors

    def close(self):
        if self.model is not None:
            import gc
            import torch
            self.model = self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class DistributionMetric:
    def __init__(self, real, groups):
        self.center = np.median(real, axis=0)
        self.scale = np.maximum(np.std(real, axis=0), 0.03)
        self.groups = groups

    def normalize(self, a):
        return np.clip((a - self.center) / self.scale, -12, 12)

    def compare(self, real, synth):
        r, s = self.normalize(real), self.normalize(synth)
        scores = {}
        for name, (lo, hi) in self.groups.items():
            x, y = r[:, lo:hi], s[:, lo:hi]
            mean_gap = np.mean((x.mean(0) - y.mean(0)) ** 2)
            spread_gap = np.mean((x.std(0) - y.std(0)) ** 2)
            # Symmetric nearest coverage penalizes missing styles AND implausible extra styles.
            dist = np.maximum((x * x).mean(1)[:, None] + (y * y).mean(1)[None, :]
                              - 2 * x @ y.T / x.shape[1], 0)
            coverage = 0.5 * (dist.min(1).mean() + dist.min(0).mean())
            scores[name] = float(mean_gap + 0.5 * spread_gap + 0.25 * coverage)
        scores["total"] = float(np.mean(list(scores.values())))
        return scores


def cluster_styles(vectors, count, seed):
    count = max(1, min(count, len(vectors) // 12))
    a = (vectors - vectors.mean(0)) / np.maximum(vectors.std(0), 0.03)
    rng = np.random.default_rng(seed)
    centers = [a[int(rng.integers(len(a)))]]
    while len(centers) < count:
        d = np.min(np.stack([np.mean((a - c) ** 2, axis=1) for c in centers]), axis=0)
        centers.append(a[int(np.argmax(d))])
    centers = np.stack(centers)
    for _ in range(20):
        labels = np.argmin(((a[:, None] - centers[None]) ** 2).mean(2), axis=1)
        updated = np.stack([a[labels == j].mean(0) if np.any(labels == j) else centers[j]
                            for j in range(count)])
        if np.allclose(updated, centers):
            break
        centers = updated
    # Tiny clusters are merged into the nearest sufficiently populated style.
    sizes = np.bincount(labels, minlength=count)
    valid = np.where(sizes >= 8)[0]
    if not len(valid):
        return np.zeros(len(a), dtype=int)
    nearest = valid[np.argmin(((a[:, None] - centers[valid][None]) ** 2).mean(2), axis=1)]
    return np.array([list(valid).index(j) for j in nearest], dtype=int)
