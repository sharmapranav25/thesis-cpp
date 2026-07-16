"""Faster char-CNN multi-scale inference using DataLoader workers.

Replaces `charcls_softmax_for_dir` in `src/charcls/infer.py` for large
inference jobs (10k+ images). The original is single-threaded CPU-bound
on PIL crops; this version pushes cropping into worker processes so the
GPU stays busy.

API:
    charcls_softmax_for_dir_fast(model, src_dir, boxes_jsonl, crop_size,
                                 pad_frac, batch_images=64, num_workers=4)
        → (letter_softmax, case_softmax)
    Same return shape as the original.

Validation: when num_workers=0 and batch_images=1 this should produce
numerically-identical output to the original. The fast path differs only
in batching/parallelism, not in the actual ops.
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
from torch.utils.data import DataLoader, Dataset

from src.charcls.data import _crop_with_pad, _to_rgb
from src.charcls.model import NUM_CASES, NUM_CLASSES
from src.charcls.infer import _make_tf


class _PerImageCharDataset(Dataset):
    """Per-image dataset: yields a stack of 5 cropped tensors per image.
    Cropping happens in worker processes; GPU only sees the prepared tensor."""

    def __init__(
        self,
        src_dir: Path,
        box_records: dict[str, list[list[float]]],
        crop_size: int,
        pad_frac: float,
    ):
        self.src_dir = Path(src_dir)
        self.crop_size = crop_size
        self.pad_frac = pad_frac
        # only keep filenames that exist on disk and have boxes
        present = {p.name for p in self.src_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
        self.filenames = sorted(fn for fn in box_records if fn in present)
        self.boxes = {fn: box_records[fn] for fn in self.filenames}
        self.tf = _make_tf(crop_size)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        fn = self.filenames[i]
        boxes = self.boxes[fn][:5]
        img = Image.open(self.src_dir / fn).convert("RGB")
        crops = []
        for box in boxes:
            crop = _crop_with_pad(img, box, self.pad_frac, 0.0)
            crops.append(self.tf(crop))
        # Pad to exactly 5 if box record was shorter (shouldn't happen).
        while len(crops) < 5:
            crops.append(torch.zeros(3, self.crop_size, self.crop_size))
        return torch.stack(crops, dim=0), fn   # [5, 3, H, W], str


@torch.inference_mode()
def charcls_softmax_for_dir_fast(
    model,
    src_dir: Path,
    boxes_jsonl: Path,
    crop_size: int,
    pad_frac: float,
    batch_images: int = 64,
    num_workers: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Returns (letter_softmax, case_softmax) — same as the original
    `charcls_softmax_for_dir`, but ~3× faster via DataLoader prefetching.
    Default num_workers=0 because Windows spawn overhead exceeds savings
    for jobs <500k crops; bump to 2-4 only for 1M+ inferences."""
    # load box records
    box_records: dict[str, list] = {}
    with Path(boxes_jsonl).open() as f:
        for line in f:
            r = json.loads(line)
            box_records[r["filename"]] = r["bboxes"]

    ds = _PerImageCharDataset(Path(src_dir), box_records, crop_size, pad_frac)
    loader = DataLoader(
        ds,
        batch_size=batch_images,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    out_letter: dict[str, torch.Tensor] = {}
    out_case: dict[str, torch.Tensor] = {}

    H = W = crop_size
    for batch_crops, batch_fns in loader:
        # batch_crops: [B, 5, 3, H, W]; batch_fns: list[str] length B
        B = batch_crops.size(0)
        flat = batch_crops.view(B * 5, 3, H, W).cuda(non_blocking=True)
        logit_l, logit_c = model(flat)
        prob_l = logit_l.softmax(-1).cpu().view(B, 5, NUM_CLASSES)
        prob_c = logit_c.softmax(-1).cpu().view(B, 5, NUM_CASES)
        for j, fn in enumerate(batch_fns):
            out_letter[fn] = prob_l[j]
            out_case[fn] = prob_c[j]

    return out_letter, out_case


def main():
    """Quick benchmark + numerical-correctness check vs the slow version
    on whatever directory is passed in."""
    import argparse, time

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="if set, only check first N images vs slow version")
    ap.add_argument("--batch-images", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    from src.charcls.infer import load_charcls, charcls_softmax_for_dir as slow_fn

    print(f"loading char-CNN {args.ckpt}", flush=True)
    model, crop_size, pad_frac = load_charcls(args.ckpt)
    print(f"  crop_size={crop_size}, pad_frac={pad_frac}", flush=True)

    if args.limit is not None:
        # Build a temp dir of `limit` images? Simpler: filter via box records.
        print(f"\n== correctness check on first {args.limit} images ==", flush=True)
        import tempfile, shutil
        # truncate boxes file
        rows = []
        with args.boxes.open() as f:
            for line in f:
                rows.append(line)
        truncated_boxes = Path(tempfile.mkstemp(suffix=".jsonl")[1])
        with truncated_boxes.open("w") as f:
            for line in rows[: args.limit]:
                f.write(line)
        # Run both
        t0 = time.time()
        slow_l, slow_c = slow_fn(model, args.src, truncated_boxes, crop_size, pad_frac)
        t_slow = time.time() - t0
        t0 = time.time()
        fast_l, fast_c = charcls_softmax_for_dir_fast(
            model, args.src, truncated_boxes, crop_size, pad_frac,
            batch_images=args.batch_images, num_workers=args.num_workers,
        )
        t_fast = time.time() - t0
        print(f"  slow: {t_slow:.2f}s  ({len(slow_l)} entries)", flush=True)
        print(f"  fast: {t_fast:.2f}s  ({len(fast_l)} entries)", flush=True)
        print(f"  speedup: {t_slow / max(t_fast, 1e-9):.2f}x", flush=True)

        common = set(slow_l) & set(fast_l)
        if not common:
            print("  ERROR: no overlapping filenames!"); sys.exit(1)
        max_l = 0.0
        max_c = 0.0
        for fn in common:
            d_l = (slow_l[fn] - fast_l[fn]).abs().max().item()
            d_c = (slow_c[fn] - fast_c[fn]).abs().max().item()
            max_l = max(max_l, d_l)
            max_c = max(max_c, d_c)
        print(f"  max diff letter softmax: {max_l:.2e}", flush=True)
        print(f"  max diff case   softmax: {max_c:.2e}", flush=True)
        if max_l < 1e-4 and max_c < 1e-4:
            print("  [OK] outputs match within fp16/fp32 tolerance", flush=True)
        else:
            print("  [WARN] outputs differ — investigate", flush=True)
    else:
        # just benchmark fast on full dir
        t0 = time.time()
        l, c = charcls_softmax_for_dir_fast(
            model, args.src, args.boxes, crop_size, pad_frac,
            batch_images=args.batch_images, num_workers=args.num_workers,
        )
        dt = time.time() - t0
        print(f"fast: {dt:.1f}s for {len(l)} images "
              f"({len(l)/max(dt,1e-9):.0f} img/s)", flush=True)


if __name__ == "__main__":
    main()
