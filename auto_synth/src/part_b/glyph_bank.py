"""Glyph bank: load per-cluster crops and serve them as a callable.

Builds on Part A's output. Loads the per-cluster crops produced by
src/part_a/run.py and exposes:
  - GlyphBank.cluster_ids: list of valid cluster ids (filtered alphabet)
  - GlyphBank.render(cluster_id, seed=None) -> PIL.Image — return a (lightly
    augmented) glyph from that cluster

Augmentations are kept gentle so the rendered glyph still belongs to the
target scheme: small color jitter, small affine, optional alpha threshold
to soften the white background.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class GlyphBank:
    def __init__(self, scheme_dir: Path, max_per_cluster: int = 30, only_mapped: bool = True):
        scheme_dir = Path(scheme_dir)
        self.scheme_dir = scheme_dir
        manifest_path = scheme_dir / "alphabet_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no alphabet_manifest.json at {scheme_dir}")
        self.manifest = json.loads(manifest_path.read_text())
        self.alpha_dir = scheme_dir / "alphabet"
        self._crops_cache: dict[int, list[Image.Image]] = {}
        self.max_per_cluster = max_per_cluster

        # Optional cluster->char mapping (populated by part_a/map_clusters.py)
        self.cluster_to_char: dict[int, str] = {}
        if "cluster_to_char" in self.manifest:
            for k, v in self.manifest["cluster_to_char"].items():
                if v.get("char") and v["char"] != "?":
                    self.cluster_to_char[int(k)] = v["char"]

        all_ids = list(self.manifest["alphabet_cluster_ids"])
        if only_mapped and self.cluster_to_char:
            self.cluster_ids = [c for c in all_ids if c in self.cluster_to_char]
        else:
            self.cluster_ids = all_ids

        # Char -> [cluster_ids]: groups stylization variants of the same char.
        # When sampling glyphs, we pick a CHAR uniformly (or per inferred bank
        # weights), then a random cluster among that char's variants. This
        # avoids over-weighting chars that happen to have more clusters
        # (the "character leakage" issue).
        self.char_to_clusters: dict[str, list[int]] = {}
        for cid, ch in self.cluster_to_char.items():
            if cid in set(self.cluster_ids):
                self.char_to_clusters.setdefault(ch, []).append(cid)
        self.observed_chars: list[str] = sorted(self.char_to_clusters.keys())

    def label(self, cid: int) -> str:
        """Return char if mapped, else cluster id."""
        if cid in self.cluster_to_char:
            return self.cluster_to_char[cid]
        return f"C{cid:03d}"

    def _load_cluster(self, cid: int) -> list[Image.Image]:
        if cid in self._crops_cache:
            return self._crops_cache[cid]
        cdir = self.alpha_dir / f"cluster_{cid:03d}"
        crops = []
        if cdir.exists():
            files = sorted([p for p in cdir.iterdir() if p.suffix == ".png" and not p.name.startswith("_")])
            for f in files[:self.max_per_cluster]:
                try:
                    crops.append(Image.open(f).convert("RGB"))
                except Exception:
                    pass
        self._crops_cache[cid] = crops
        return crops

    def render_char(self, ch: str, seed: int | None = None, **kwargs) -> Image.Image:
        """Pick a random cluster for char `ch`, render it. This is the
        leakage-aware sampler: chars with more stylization variants don't get
        over-represented because we pick CHAR first, cluster second."""
        if ch not in self.char_to_clusters:
            raise ValueError(f"char {ch!r} not in glyph bank")
        rng = random.Random(seed)
        cid = rng.choice(self.char_to_clusters[ch])
        return self.render(cid, seed=rng.randint(0, 2**31), **kwargs)

    def render(self, cid: int, seed: int | None = None,
               jitter_color: bool = True,
               jitter_size: bool = True,
               jitter_rotation: bool = True) -> Image.Image:
        crops = self._load_cluster(cid)
        if not crops:
            raise ValueError(f"cluster {cid} has no crops")
        rng = random.Random(seed)
        glyph = rng.choice(crops).copy()

        if jitter_color:
            try:
                glyph = ImageEnhance.Color(glyph).enhance(rng.uniform(0.7, 1.3))
                glyph = ImageEnhance.Brightness(glyph).enhance(rng.uniform(0.85, 1.15))
                glyph = ImageEnhance.Contrast(glyph).enhance(rng.uniform(0.85, 1.15))
            except Exception:
                pass
        if jitter_size:
            scale = rng.uniform(0.85, 1.15)
            new_w = max(4, int(glyph.width * scale))
            new_h = max(4, int(glyph.height * scale))
            glyph = glyph.resize((new_w, new_h), Image.LANCZOS)
        if jitter_rotation:
            angle = rng.uniform(-6, 6)
            glyph = glyph.rotate(angle, expand=True, fillcolor=(255, 255, 255))

        return glyph

    def to_alpha(self, glyph: Image.Image, white_thresh: int = 240) -> Image.Image:
        """Convert glyph to RGBA where near-white pixels become transparent."""
        rgb = np.array(glyph.convert("RGB"))
        alpha = ((rgb < white_thresh).any(axis=-1).astype(np.uint8)) * 255
        rgba = np.dstack([rgb, alpha])
        return Image.fromarray(rgba, mode="RGBA")


def cluster_id_to_label(cid: int) -> str:
    """Stub mapping cluster_id -> human-readable label.

    For now, just returns f'C{cid}'. Once we have a way to map clusters to
    actual chars (e.g. by running a generic OCR on cluster centroids), this
    becomes the place to plug in that mapping.
    """
    return f"C{cid:03d}"
