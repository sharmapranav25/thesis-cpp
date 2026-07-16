"""Build per-character crop bank by pseudolabeling unlabeled real captchas.

Pipeline:
  1. Run detector (model 1) on the scheme's raw_dir → N boxes per image.
  2. Run PARSeq (model 1) on the same dir → predicted string + per-char conf.
  3. For each image: if len(pred) == n_chars, walk the (box, char, conf)
     triples; keep ones with conf > --min-conf and char in scheme.charset.
  4. Crop the real image at each kept box (with pad), and use the
     real-vs-LaMa pixel diff as an alpha channel so the saved RGBA crop
     contains only the character (not the bg around it).
  5. Save to <scheme.bank_dir>/<char>/<source>_<idx>.png

Outputs:
  <bank_dir>/<char>/*.png        — RGBA char crops, alpha-masked
  <bank_dir>/_meta.json          — per-char counts + thresholds used
  <bank_dir>/_qa.png             — 8 random crops per char (grid)
  <bank_dir>/_skipped.csv        — filename, reason, parseq_pred for skipped images

Run:
  cd "E:/pranav/synthetic image generation"
  "E:/pranav/model 1/.venv/Scripts/python.exe" -m src.synth_v1.build_bank \\
      --scheme cryptbb [--min-conf 0.9] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

# Anaconda DLL prep matches model 1's pattern.
_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parents[2]
MODEL1 = Path(r"E:\pranav\model 1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL1))
sys.path.insert(0, str(MODEL1 / "third_party" / "parseq"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from src.synth_v1.scheme import SchemeSpec, get  # noqa: E402

# model 1 inference imports — defer-style so we get a clear error if env breaks
from src.detector.infer import ImageDirSimple, load_detector  # type: ignore  # noqa: E402
from src.detector.model import sort_by_cx  # type: ignore  # noqa: E402
from src.parseq.train import _to_rgb, build_parseq  # type: ignore  # noqa: E402

DETECTOR_CKPT = MODEL1 / "checkpoints" / "detector_v2" / "best.pt"
PARSEQ_CKPT = MODEL1 / "checkpoints" / "parseq_v2" / "parseq-epoch=11-val_accuracy=57.260.ckpt"
PARSEQ_CFG = MODEL1 / "configs" / "parseq_v2.yaml"

# Continuous-alpha keying: pixels with diff < ALPHA_LOW are bg (transparent),
# diff > ALPHA_HIGH are solid char (opaque), and intermediate diffs fade
# proportionally. The bg leakage at character edges fades naturally because
# those pixels have small diffs from LaMa.
ALPHA_LOW = 25
ALPHA_HIGH = 90
PIX_DIFF_THR = ALPHA_LOW  # for connected-components filtering only
CROP_PAD = 2              # px pad around each detector box when cropping
MIN_CC_AREA = 12          # drop connected components smaller than this (px)


def char_to_folder(ch: str) -> str:
    """Windows filesystem is case-insensitive — 'a' and 'A' would collide.
    Encode case explicitly so digits and both letter cases stay separate."""
    if ch.isdigit():
        return ch
    if ch.islower():
        return f"L_{ch}"
    if ch.isupper():
        return f"U_{ch}"
    return f"X_{ord(ch):02x}"  # symbols/other (not used for cryptbb)


def text_mask_from_diff(real_rgb: np.ndarray, lama_rgb: np.ndarray) -> np.ndarray:
    """Return continuous float [0,1] HxW alpha map.
    Uses real-vs-LaMa pixel diff; small diffs (bg residue) → ~0,
    large diffs (char strokes) → 1. This fades out the bg halo around
    char edges that a hard threshold would leave opaque."""
    diff = np.abs(real_rgb.astype(np.int16) - lama_rgb.astype(np.int16)).max(axis=-1).astype(np.float32)
    a = (diff - ALPHA_LOW) / max(1.0, (ALPHA_HIGH - ALPHA_LOW))
    return np.clip(a, 0.0, 1.0)


@torch.inference_mode()
def detect_all(model, raw_dir: Path, n_chars: int, w: int, h: int,
               batch: int = 64, device: str = "cuda") -> dict[str, list[list[float]]]:
    """name -> [n_chars, 4] in pixel coords."""
    ds = ImageDirSimple(raw_dir)
    loader = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=True)
    results: dict[str, list[list[float]]] = {}
    for x, names in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        out = sort_by_cx(out).cpu()
        for i, name in enumerate(names):
            boxes = []
            for b in out[i]:
                cx, cy, bw, bh = (v.item() for v in b)
                x0 = max(0.0, (cx - bw / 2) * w)
                y0 = max(0.0, (cy - bh / 2) * h)
                x1 = min(w - 1, (cx + bw / 2) * w)
                y1 = min(h - 1, (cy + bh / 2) * h)
                boxes.append([x0, y0, x1, y1])
            results[name] = boxes
    return results


class _ParseqDir(torch.utils.data.Dataset):
    def __init__(self, root: Path, img_size: tuple[int, int]):
        self.paths = sorted(p for p in root.glob("*.png") if not p.name.startswith("_"))
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i])), self.paths[i].name


@torch.inference_mode()
def recognize_all(parseq, raw_dir: Path, img_size: tuple[int, int],
                  batch: int = 64, device: str = "cuda") -> dict[str, tuple[str, list[float]]]:
    """name -> (predicted_string, [per-char conf])."""
    ds = _ParseqDir(raw_dir, img_size)
    loader = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=True)
    out: dict[str, tuple[str, list[float]]] = {}
    for x, names in loader:
        x = x.to(device, non_blocking=True)
        logits = parseq(x)
        preds, probs = parseq.tokenizer.decode(logits.softmax(-1))
        for name, pred, prob in zip(names, preds, probs):
            cps = [float(prob[i].item()) for i in range(len(pred))]
            out[name] = (pred, cps)
    return out


def _largest_cc_mask(binary_u8: np.ndarray) -> np.ndarray:
    """Keep components that touch the central horizontal band (where the char
    sits) and meet a min-area threshold. Stray bg specks are removed."""
    if binary_u8.max() == 0:
        return binary_u8
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if n <= 1:
        return binary_u8
    h = binary_u8.shape[0]
    band_y0, band_y1 = int(h * 0.20), int(h * 0.80)
    out = np.zeros_like(binary_u8)
    for cid in range(1, n):
        if stats[cid, cv2.CC_STAT_AREA] < MIN_CC_AREA:
            continue
        comp = (labels == cid)
        if not comp[band_y0:band_y1].any():
            continue
        out[comp] = 255
    return out


def crop_rgba(real: np.ndarray, alpha01: np.ndarray, box: list[float],
              pad: int = CROP_PAD) -> Image.Image:
    H, W = real.shape[:2]
    x0 = max(0, int(round(box[0])) - pad)
    y0 = max(0, int(round(box[1])) - pad)
    x1 = min(W, int(round(box[2])) + pad)
    y1 = min(H, int(round(box[3])) + pad)
    if x1 <= x0 or y1 <= y0:
        return None  # type: ignore[return-value]
    rgb = real[y0:y1, x0:x1]
    a_cont = alpha01[y0:y1, x0:x1]
    # Build a binary mask from the strong-alpha pixels and keep only the
    # central connected component. Then multiply the continuous alpha by
    # this mask to suppress disconnected speckles while preserving the
    # gradient at char edges.
    binary = ((a_cont > 0.4).astype(np.uint8) * 255)
    binary = _largest_cc_mask(binary)
    if binary.max() == 0:
        return None  # all-bg crop, skip
    keep = (binary > 0).astype(np.float32)
    a = (a_cont * keep * 255.0).clip(0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    return Image.fromarray(rgba)


def make_qa_sheet(bank_dir: Path, charset: str, per_char: int = 8,
                  cell: int = 48) -> Path:
    """One row per char (only chars with samples), `per_char` random crops each.
    Renders RGBA crops on a checkerboard so transparency is visible."""
    chars = [c for c in charset if (bank_dir / char_to_folder(c)).is_dir()
             and any((bank_dir / char_to_folder(c)).iterdir())]
    if not chars:
        return None  # type: ignore[return-value]
    bg = Image.new("RGB", (per_char * cell + 60, len(chars) * cell + 4),
                   (40, 40, 40))
    rng = random.Random(0)
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(bg)
    for ri, ch in enumerate(chars):
        draw.text((4, ri * cell + cell // 2 - 9), ch, font=font, fill=(220, 220, 220))
        crops = list((bank_dir / char_to_folder(ch)).glob("*.png"))
        rng.shuffle(crops)
        for ci, p in enumerate(crops[:per_char]):
            im = Image.open(p).convert("RGBA")
            im.thumbnail((cell - 4, cell - 4), Image.BICUBIC)
            tile = Image.new("RGB", (cell, cell), (90, 90, 90))
            # checker
            for yy in range(0, cell, 8):
                for xx in range(0, cell, 8):
                    if ((xx // 8) + (yy // 8)) % 2:
                        for ddy in range(min(8, cell - yy)):
                            for ddx in range(min(8, cell - xx)):
                                tile.putpixel((xx + ddx, yy + ddy), (60, 60, 60))
            ox = (cell - im.width) // 2
            oy = (cell - im.height) // 2
            tile.paste(im, (ox, oy), im)
            bg.paste(tile, (60 + ci * cell, ri * cell + 2))
    out = bank_dir / "_qa.png"
    bg.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, help="scheme name from src/synth_v1/scheme.py")
    ap.add_argument("--min-conf", type=float, default=0.9,
                    help="minimum per-char PARSeq confidence to keep a crop")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap raw images processed (debugging)")
    ap.add_argument("--detector-ckpt", type=Path, default=DETECTOR_CKPT)
    ap.add_argument("--parseq-ckpt", type=Path, default=PARSEQ_CKPT)
    ap.add_argument("--parseq-cfg", type=Path, default=PARSEQ_CFG)
    args = ap.parse_args()

    scheme: SchemeSpec = get(args.scheme)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"scheme    {scheme.name}  W={scheme.W} H={scheme.H} n_chars={scheme.n_chars}")
    print(f"raw_dir   {scheme.raw_dir}")
    print(f"bg_dir    {scheme.bg_dir}")
    print(f"bank_dir  {scheme.bank_dir}")
    print(f"device    {device}")
    print(f"min_conf  {args.min_conf}")

    if not scheme.raw_dir.exists():
        raise FileNotFoundError(scheme.raw_dir)
    if not scheme.bg_dir.exists():
        raise FileNotFoundError(f"bg_dir missing — run extract_bgs first: {scheme.bg_dir}")

    scheme.bank_dir.mkdir(parents=True, exist_ok=True)
    for ch in scheme.charset:
        (scheme.bank_dir / char_to_folder(ch)).mkdir(exist_ok=True)

    print("\n[1/4] loading detector ...")
    det = load_detector(args.detector_ckpt, device=device)

    print("[2/4] loading parseq ...")
    cfg = yaml.safe_load(args.parseq_cfg.read_text())
    parseq = build_parseq(cfg)
    state = torch.load(args.parseq_ckpt, map_location=device, weights_only=False)
    parseq.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    parseq = parseq.to(device).eval()

    print("[3/4] running detector + parseq on raw ...")
    t0 = time.time()
    boxes_by_name = detect_all(det, scheme.raw_dir, scheme.n_chars,
                               scheme.W, scheme.H, device=device)
    rec_by_name = recognize_all(parseq, scheme.raw_dir,
                                tuple(cfg["img_size"]), device=device)
    print(f"  detector: {len(boxes_by_name)} imgs   parseq: {len(rec_by_name)} imgs   {time.time()-t0:.1f}s")

    # iteration order = sorted PNG list (matches _ParseqDir / ImageDirSimple)
    names = sorted(p.name for p in scheme.raw_dir.glob("*.png") if not p.name.startswith("_"))
    if args.limit:
        names = names[: args.limit]

    print("\n[4/4] cropping + alpha-keying ...")
    skipped: list[tuple[str, str, str]] = []
    saved = 0
    per_char_count: dict[str, int] = {ch: 0 for ch in scheme.charset}
    t0 = time.time()
    for idx, name in enumerate(names):
        if name not in rec_by_name or name not in boxes_by_name:
            skipped.append((name, "missing_inference", ""))
            continue
        pred, confs = rec_by_name[name]
        boxes = boxes_by_name[name]
        if len(pred) != scheme.n_chars:
            skipped.append((name, f"len(pred)={len(pred)}!= {scheme.n_chars}", pred))
            continue
        if len(boxes) != scheme.n_chars:
            skipped.append((name, f"len(boxes)={len(boxes)}", pred))
            continue

        real_pil = Image.open(scheme.raw_dir / name).convert("RGB")
        if real_pil.size != (scheme.W, scheme.H):
            real_pil = real_pil.resize((scheme.W, scheme.H))
        real = np.array(real_pil)

        lama_path = scheme.bg_dir / name
        if not lama_path.exists():
            skipped.append((name, "missing_lama", pred))
            continue
        lama_pil = Image.open(lama_path).convert("RGB")
        if lama_pil.size != (scheme.W, scheme.H):
            lama_pil = lama_pil.resize((scheme.W, scheme.H))
        mask01 = text_mask_from_diff(real, np.array(lama_pil))

        for ci, (ch, conf, box) in enumerate(zip(pred, confs, boxes)):
            if conf < args.min_conf:
                continue
            if ch not in scheme.charset:
                continue
            crop = crop_rgba(real, mask01, box)
            if crop is None or crop.size[0] < 4 or crop.size[1] < 4:
                continue
            out_path = scheme.bank_dir / char_to_folder(ch) / f"{Path(name).stem}_{ci}.png"
            crop.save(out_path)
            per_char_count[ch] += 1
            saved += 1

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(names)}  saved={saved}  skipped={len(skipped)}  "
                  f"{time.time()-t0:.0f}s")

    # summary
    meta = {
        "scheme": scheme.name,
        "n_chars": scheme.n_chars,
        "min_conf": args.min_conf,
        "alpha_low": ALPHA_LOW,
        "alpha_high": ALPHA_HIGH,
        "min_cc_area": MIN_CC_AREA,
        "crop_pad": CROP_PAD,
        "n_raw": len(names),
        "n_saved_crops": saved,
        "n_skipped_imgs": len(skipped),
        "per_char_count": per_char_count,
        "detector_ckpt": str(args.detector_ckpt),
        "parseq_ckpt": str(args.parseq_ckpt),
    }
    (scheme.bank_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    with (scheme.bank_dir / "_skipped.csv").open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["filename", "reason", "parseq_pred"])
        wcsv.writerows(skipped)

    nonzero = [(ch, n) for ch, n in per_char_count.items() if n > 0]
    nonzero.sort(key=lambda kv: -kv[1])
    print(f"\n=== summary ===")
    print(f"saved {saved} crops across {len(nonzero)} chars; "
          f"skipped {len(skipped)} images")
    print(f"top chars: {nonzero[:10]}")
    print(f"empty chars: {[ch for ch, n in per_char_count.items() if n == 0]}")

    print("\nbuilding QA sheet ...")
    qa = make_qa_sheet(scheme.bank_dir, scheme.charset)
    if qa:
        print(f"qa sheet -> {qa}")


if __name__ == "__main__":
    main()
