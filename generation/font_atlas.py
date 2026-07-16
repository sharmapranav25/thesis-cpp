"""Font atlas — recreate a scheme's font by RETRIEVING the closest generic library font.

FontCLIP/DeepFont aren't obtainable offline, so we fall back to a general vision
backbone (DINOv2-base, already on E:). That gives coarse *family* matching
(serif / sans / mono / blackletter / script / stencil), which is exactly the
"coarse font style" bar in DESIGN.md.

Method (all local, no training):
  * enumerate ~175 generic .ttf (model 0 font_pool + venv DejaVu),
  * render each font's glyph set as clean black-on-white, embed each glyph with
    DINOv2, average+normalise -> one style vector per font,
  * a real captcha's extracted glyph crops are normalised the SAME way, embedded,
    averaged -> a query vector; cosine-NN (brute force, ~175 rows) picks the font.

Run once to build the cache:   python -m font_atlas --build
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "E:/hf_cache/hub")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent
FONT_POOL_JSON = HERE / "configs" / "font_pool.json"
ATLAS_NPZ = HERE / "configs" / "font_atlas.npz"
DEJAVU_DIR = Path("E:/pranav/model 1/.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf")

EMBED_MODEL = "facebook/dinov2-base"   # on E:; 768-dim
CANVAS = 112
# representative glyphs — mix of upper/lower/digits with distinctive shapes
ATLAS_CHARS = "ABEGMQRSaebgmnrs2345689"


# --------------------------------------------------------------------------- #
# Glyph rendering + normalisation (shared by atlas build and query)
# --------------------------------------------------------------------------- #
def to_black_on_white(img: Image.Image, canvas: int = CANVAS) -> Image.Image:
    """Binarise -> autocrop to ink -> square-pad -> resize. Domain-aligns atlas
    glyphs (clean) and real crops (mask-applied glyph-on-white from segment)."""
    g = np.asarray(img.convert("L"))
    # ink is dark; Otsu split, keep the darker side as foreground
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (th == 0).sum() > (th == 255).sum():   # more black than white -> invert polarity
        th = 255 - th
    ink = th == 0
    ys, xs = np.where(ink)
    if len(xs) < 3:
        return Image.new("RGB", (canvas, canvas), "white")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = (g[y0:y1 + 1, x0:x1 + 1])
    h, w = crop.shape
    side = max(h, w)
    pad = np.full((side, side), 255, np.uint8)
    pad[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    out = cv2.resize(pad, (canvas, canvas), interpolation=cv2.INTER_AREA)
    return Image.fromarray(out).convert("RGB")


def render_glyph(font_path: str, ch: str, canvas: int = CANVAS) -> Optional[Image.Image]:
    """Clean black glyph on white, centred. None if the font can't be loaded."""
    try:
        font = ImageFont.truetype(font_path, int(canvas * 0.7))
    except Exception:
        return None
    im = Image.new("L", (canvas, canvas), 255)
    d = ImageDraw.Draw(im)
    try:
        bb = d.textbbox((0, 0), ch, font=font)
    except Exception:
        return None
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    if w <= 0 or h <= 0:
        return None
    d.text(((canvas - w) / 2 - bb[0], (canvas - h) / 2 - bb[1]), ch, font=font, fill=0)
    return im.convert("RGB")


# --------------------------------------------------------------------------- #
# DINOv2 embedder (minimal; mirrors auto_synth/src/part_a/embed.py)
# --------------------------------------------------------------------------- #
class _Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = 64):
        import torch
        from torchvision import transforms as T
        from transformers import AutoModel
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.tf = T.Compose([
            T.Lambda(lambda im: im.convert("RGB")),
            T.Resize((224, 224), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.batch = batch_size
        self.dim = self.model.config.hidden_size

    def embed(self, crops: Sequence[Image.Image]) -> np.ndarray:
        torch = self.torch
        out = np.empty((len(crops), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(crops), self.batch):
                b = crops[i:i + self.batch]
                x = torch.stack([self.tf(c) for c in b]).to(self.device)
                f = self.model(x).last_hidden_state[:, 0]
                f = torch.nn.functional.normalize(f, dim=-1)
                out[i:i + len(b)] = f.cpu().numpy()
        return out


def _font_list() -> list[str]:
    fonts = []
    if FONT_POOL_JSON.exists():
        fonts += [p.replace("\\", "/") for p in json.loads(FONT_POOL_JSON.read_text())]
    if DEJAVU_DIR.exists():
        fonts += [str(p).replace("\\", "/") for p in sorted(DEJAVU_DIR.glob("*.ttf"))]
    # de-dup preserving order
    seen, uniq = set(), []
    for f in fonts:
        k = f.lower()
        if k not in seen and Path(f).is_file():
            seen.add(k); uniq.append(f)
    return uniq


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


# --------------------------------------------------------------------------- #
# Build + query
# --------------------------------------------------------------------------- #
def build_atlas(out: Path = ATLAS_NPZ, chars: str = ATLAS_CHARS, canvas: int = CANVAS):
    emb = _Embedder()
    fonts = _font_list()
    vectors, kept, blanks = [], [], 0
    for fp in fonts:
        glyphs = []
        for ch in chars:
            g = render_glyph(fp, ch, canvas)
            if g is not None:
                glyphs.append(to_black_on_white(g, canvas))
        if len(glyphs) < max(3, len(chars) // 3):   # font failed to render most glyphs
            blanks += 1
            continue
        vecs = emb.embed(glyphs)
        style = _norm(vecs.mean(axis=0, keepdims=True))[0]
        vectors.append(style)
        kept.append(fp)
    V = np.stack(vectors).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, vectors=V, fonts=np.array(kept), chars=chars, canvas=canvas,
             embed_model=EMBED_MODEL)
    print(f"[font_atlas] built {V.shape[0]} font vectors (dim {V.shape[1]}), "
          f"{blanks} fonts skipped as blank -> {out}")
    return out


class FontAtlas:
    def __init__(self, npz: Path = ATLAS_NPZ):
        d = np.load(npz, allow_pickle=True)
        self.vectors = d["vectors"].astype(np.float32)   # [N, D], normalised
        self.fonts = [str(f) for f in d["fonts"]]
        self.canvas = int(d["canvas"])
        self._emb: Optional[_Embedder] = None

    def _embedder(self) -> _Embedder:
        if self._emb is None:
            self._emb = _Embedder()
        return self._emb

    def query_vector(self, crops: Sequence[Image.Image]) -> np.ndarray:
        norm = [to_black_on_white(c, self.canvas) for c in crops]
        vecs = self._embedder().embed(norm)
        return _norm(vecs.mean(axis=0, keepdims=True))[0]

    def query(self, crops: Sequence[Image.Image], top_k: int = 5) -> list[tuple[str, float]]:
        """Return [(font_path, cosine)] best-first for the real glyph crops."""
        q = self.query_vector(crops)
        sims = self.vectors @ q
        idx = np.argsort(-sims)[:top_k]
        return [(self.fonts[i], float(sims[i])) for i in idx]


# --------------------------------------------------------------------------- #
# CLI:  python -m font_atlas --build   |   --selftest
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="render glyphs in a known font and confirm it retrieves itself")
    args = ap.parse_args()
    if args.build:
        build_atlas()
    if args.selftest:
        atlas = FontAtlas()
        for name in ["arial.ttf", "times.ttf", "STENCIL.TTF", "impact.ttf"]:
            fp = next((f for f in atlas.fonts if f.lower().endswith(name.lower())), None)
            if not fp:
                print(f"  (no {name} in atlas)"); continue
            crops = [render_glyph(fp, c) for c in "ARSbe37"]
            crops = [c for c in crops if c is not None]
            top = atlas.query(crops, top_k=3)
            print(f"{name:16s} -> " + ", ".join(
                f"{Path(p).name}:{s:.3f}" for p, s in top))
