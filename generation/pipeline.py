"""Pipeline orchestrator — unlabeled captcha folder -> labeled synth pool.

Path 1 (recreate): director reads the reals -> structured spec; the font atlas
retrieves the closest generic library font; php_gen_ext renders OUR text (label
correct by construction) in that font with the spec's bg/colour/distortion knobs,
in ONE pass (no compositing seam). This is the primary path for schemes whose font
is close to a generic family.

Path 2 (real-crop fallback, for exotic fonts) is added separately and selected by
the difficulty router when the best font-match cosine is below threshold.

Run:  python -m pipeline --scheme cryptbb --n 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image

import compose
import director
import extract
from font_atlas import FontAtlas

HERE = Path(__file__).resolve().parent
PHP = "C:/xampp/php/php.exe"
PHP_GEN = str(HERE / "php_gen_ext.php")

# charset by inferred case (avoid visually ambiguous glyphs a touch)
_CHARSETS = {
    "digits": "0123456789",
    "upper": "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789",
    "lower": "abcdefghijkmnpqrstuvwxyz0123456789",
    "mixed": "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz0123456789",
}

SCHEMES = {
    "cryptbb": dict(dir="E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001",
                    expected=(4, 7), bank="E:/pranav/auto_synth/data/cryptbb_unsolved"),
    "darkbay": dict(dir="E:/pranav/cryptbb_sourcefree/data/darkbay/pool", expected=(4, 8),
                    bank="E:/pranav/generation/banks/darkbay"),
    "city":    dict(dir="E:/pranav/City (live)-20260503T045450Z-3-001", expected=(4, 8),
                    bank="E:/pranav/auto_synth/data/city"),
    "courier": dict(dir="E:/pranav/Courier Market (live)-20260503T045531Z-3-001", expected=(4, 8),
                    bank="E:/pranav/auto_synth/data/courier_market"),
}


def _list_images(d: str) -> list[str]:
    ps = glob.glob(str(Path(d) / "**" / "*.png"), recursive=True)
    ps += glob.glob(str(Path(d) / "**" / "*.jpg"), recursive=True)
    return sorted(ps)


def _median_canvas(paths, k=30):
    ws, hs = [], []
    for p in paths[:k]:
        try:
            w, h = Image.open(p).size
            ws.append(w); hs.append(h)
        except Exception:
            pass
    ws.sort(); hs.sort()
    return (ws[len(ws) // 2], hs[len(hs) // 2]) if ws else (200, 60)


def _sample_length(modes, rng):
    lens, wts = zip(*modes)
    return rng.choices(lens, weights=wts, k=1)[0]


def run(scheme: str, out_dir: Path, n: int = 200, seed: int = 0,
        use_vlm: bool = True, top_fonts: int = 1):
    cfg = SCHEMES[scheme]
    paths = _list_images(cfg["dir"])
    if not paths:
        raise SystemExit(f"no images under {cfg['dir']}")
    rng = random.Random(seed)
    W, H = _median_canvas(paths)

    # step 2 — director spec (VLM + pixels)
    sample = [Image.open(p) for p in paths[:8]]
    spec = director.direct(sample, use_vlm=use_vlm)

    # step 3(light) — glyph crops for font match + length distribution
    crops = extract.sample_crops(paths, per_image=8, max_images=60, expected=cfg["expected"])
    atlas = FontAtlas()
    ranked = atlas.query(crops, top_k=max(3, top_fonts)) if crops else []
    fonts = [p for p, _ in ranked[:top_fonts]] or [
        "E:/pranav/model 1/.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"]
    spec.length_modes = extract.length_modes(paths, expected=cfg["expected"])

    charset = _CHARSETS.get(spec.case, _CHARSETS["mixed"])

    # build php cfg + jobs
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"; img_dir.mkdir(exist_ok=True)
    php_cfg = spec.to_php_cfg(W, H, fonts)
    (out_dir / "_php_cfg.json").write_text(json.dumps(php_cfg, indent=2))

    labels = []
    jobs_lines = []
    for i in range(n):
        L = _sample_length(spec.length_modes, rng)
        text = "".join(rng.choice(charset) for _ in range(L))
        out_png = img_dir / f"synth_{i:05d}.png"
        labels.append((out_png.name, text))
        jobs_lines.append(f"{text}\t{out_png.as_posix()}")
    jobs_path = out_dir / "_jobs.tsv"
    jobs_path.write_text("\n".join(jobs_lines))

    # step 4/5 — render (php one-pass = bg+glyph+distort, no seam)
    r = subprocess.run([PHP, PHP_GEN, str(jobs_path), str(out_dir / "_php_cfg.json")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"php render failed ({r.returncode})")

    (out_dir / "labels.csv").write_text(
        "filename,label\n" + "\n".join(f"{fn},{lb}" for fn, lb in labels))
    manifest = dict(scheme=scheme, n=n, canvas=[W, H], path="recreate",
                    fonts=fonts, spec=asdict(spec))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[{scheme}] {n} synth -> {img_dir}  | font={Path(fonts[0]).name}  "
          f"| len_modes={spec.length_modes}  | bg={spec.bg} colored={spec.colored}")
    return manifest


def _render_bg_pool(spec, W, H, out_dir: Path, m: int):
    cfg = spec.to_php_cfg(W, H, ["dummy"]); cfg["bg_only"] = True
    (out_dir / "_bg_cfg.json").write_text(json.dumps(cfg))
    bgdir = out_dir / "_bg"; bgdir.mkdir(exist_ok=True)
    jobs = "\n".join(f"x\t{(bgdir / f'bg_{i:04d}.png').as_posix()}" for i in range(m))
    (out_dir / "_bg_jobs.tsv").write_text(jobs)
    r = subprocess.run([PHP, PHP_GEN, str(out_dir / "_bg_jobs.tsv"), str(out_dir / "_bg_cfg.json")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr); raise SystemExit("php bg render failed")
    return sorted(bgdir.glob("bg_*.png"))


def run_realcrop(scheme: str, out_dir: Path, n: int = 200, seed: int = 0, use_vlm: bool = True):
    """Chosen path: real glyph crops (exact font) composited onto a generated bg."""
    cfg = SCHEMES[scheme]
    bank_dir = Path(cfg["bank"])
    if not (bank_dir / "alphabet_manifest.json").exists():
        raise SystemExit(f"no glyph bank at {bank_dir} — build it first (see banks/README).")
    paths = _list_images(cfg["dir"])
    rng = random.Random(seed)
    W, H = _median_canvas(paths)

    sample = [Image.open(p) for p in paths[:8]]
    spec = director.direct(sample, use_vlm=use_vlm)
    spec.length_modes = extract.length_modes(paths, expected=cfg["expected"])

    bank = compose.bank_for(bank_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bgs = [Image.open(p).convert("RGB") for p in _render_bg_pool(spec, W, H, out_dir, max(30, n // 4))]

    img_dir = out_dir / "images"; img_dir.mkdir(exist_ok=True)
    labels = []
    for i in range(n):
        L = max(3, _sample_length(spec.length_modes, rng))
        text = compose.sample_text(bank, L, rng)
        img = compose.compose_one(bank, rng.choice(bgs), text, rng)
        fn = f"synth_{i:05d}.png"
        img.save(img_dir / fn)
        labels.append((fn, text))
    (out_dir / "labels.csv").write_text(
        "filename,label\n" + "\n".join(f"{fn},{lb}" for fn, lb in labels))
    manifest = dict(scheme=scheme, n=n, canvas=[W, H], path="realcrop", bank=str(bank_dir),
                    charset="".join(bank.observed_chars), spec=asdict(spec))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[{scheme}] {n} realcrop synth -> {img_dir}  | bank_chars={''.join(bank.observed_chars)}  "
          f"| len_modes={spec.length_modes}  | bg={spec.bg}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=list(SCHEMES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--path", choices=["realcrop", "recreate"], default="realcrop")
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else HERE / "out" / args.scheme
    fn = run_realcrop if args.path == "realcrop" else run
    fn(args.scheme, out, n=args.n, seed=args.seed, use_vlm=not args.no_vlm)
