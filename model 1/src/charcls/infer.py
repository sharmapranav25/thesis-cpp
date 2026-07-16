"""Inference helpers for char classifier.

Two functions:
  - load_charcls(ckpt) → model
  - charcls_softmax_for_dir(model, src_dir, boxes_jsonl) → dict[name, [5,62]]

The boxes JSONL is expected to come from the detector (or any source of
5 axis-aligned boxes per image). For each image the function crops with
fixed pad_frac, runs the model, and returns the per-position softmax
distribution over the 62 charset chars.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image
from torchvision import transforms as T

from src.charcls.data import _crop_with_pad, _to_rgb
from src.charcls.model import CHAR_TO_IDX, NUM_CLASSES, NUM_CASES, build_charcls


def load_charcls(ckpt_path: Path, device: str = "cuda"):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("cfg", {})
    crop_size = cfg.get("crop_size", 64)
    pad_frac = cfg.get("pad_frac", 0.20)
    model = build_charcls().to(device).eval()
    model.load_state_dict(state["model"])
    return model, crop_size, pad_frac


def _make_tf(crop_size: int):
    return T.Compose([
        T.Lambda(_to_rgb),
        T.Resize((crop_size, crop_size), antialias=True),
        T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


@torch.inference_mode()
def charcls_softmax_for_dir(
    model, src_dir: Path, boxes_jsonl: Path,
    crop_size: int, pad_frac: float, batch_size: int = 256,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Returns (letter_softmax, case_softmax):
       letter_softmax: {filename: [5, 62]}
       case_softmax:   {filename: [5, 3]}  — UPPER, LOWER, DIGIT
    """
    tf = _make_tf(crop_size)
    out_letter: dict[str, torch.Tensor] = {}
    out_case: dict[str, torch.Tensor] = {}

    # Load box records
    box_records = {}
    with boxes_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            box_records[r["filename"]] = r["bboxes"]

    items = sorted(
        p for p in src_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and not p.name.startswith("_")
    )

    buf_imgs = []
    buf_keys = []
    def flush():
        if not buf_imgs:
            return
        x = torch.stack(buf_imgs, dim=0).cuda(non_blocking=True)
        logit_l, logit_c = model(x)
        prob_l = logit_l.softmax(-1).cpu()
        prob_c = logit_c.softmax(-1).cpu()
        for k, pl, pc in zip(buf_keys, prob_l, prob_c):
            fn, ci = k
            if fn not in out_letter:
                out_letter[fn] = torch.zeros(5, NUM_CLASSES)
                out_case[fn] = torch.zeros(5, NUM_CASES)
            out_letter[fn][ci] = pl
            out_case[fn][ci] = pc
        buf_imgs.clear()
        buf_keys.clear()

    for path in items:
        if path.name not in box_records:
            continue
        img = Image.open(path).convert("RGB")
        for ci, box in enumerate(box_records[path.name][:5]):
            crop = _crop_with_pad(img, box, pad_frac=pad_frac, jitter_frac=0.0)
            buf_imgs.append(tf(crop))
            buf_keys.append((path.name, ci))
            if len(buf_imgs) >= batch_size:
                flush()
    flush()

    return out_letter, out_case


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/eval/_charcls_softmax.pt"))
    args = ap.parse_args()

    model, crop_size, pad_frac = load_charcls(args.ckpt)
    letter_sm, case_sm = charcls_softmax_for_dir(model, args.src, args.boxes, crop_size, pad_frac)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"letter": letter_sm, "case": case_sm}, args.out)
    print(f"wrote {args.out}  ({len(letter_sm)} entries)")


if __name__ == "__main__":
    main()
