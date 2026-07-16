"""TinySTR + char-CNN ensemble evaluation on the 651 real CryptBB captchas.

Same per-position log-prob mix as Model 1.5:  alpha * log(tinystr) + (1-alpha) * log(charcls)
For each alpha, prints word/char accuracy and top confusions.

Reuses Model 1's pre-trained detector + char-CNN (no retraining).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import Model 1 charcls infrastructure. Anaconda DLL hack mirrors what model 1's
# scripts do, since charcls -> timm -> nltk -> sqlite3 needs the conda libs.
_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")
MODEL1_ROOT = Path(r"E:\pranav\model 1")
sys.path.insert(0, str(MODEL1_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data import _to_rgb
from model import CHARSET as TINYSTR_CHARSET, NUM_POSITIONS, TinySTR

# Model 1 imports (paths set above)
from src.charcls.infer import load_charcls, charcls_softmax_for_dir  # noqa: E402
from src.charcls.model import CHARSET as CLS_CHARSET  # noqa: E402

PARSEQ_BASELINE = {"word_acc": 0.3118, "char_acc": 0.7437}
MODEL_15_BASELINE = {"word_acc": 0.3641, "char_acc": 0.8189}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tinystr-ckpt", type=Path,
                   default=HERE.parent / "checkpoints" / "best.pt")
    p.add_argument("--charcls-ckpt", type=Path,
                   default=MODEL1_ROOT / "checkpoints" / "charcls" / "best.pt")
    p.add_argument("--boxes-jsonl", type=Path,
                   default=MODEL1_ROOT / "data" / "_detector_boxes_eval.jsonl",
                   help="Precomputed detector bboxes (Model 1.5 already produced these).")
    p.add_argument("--eval-dir", type=Path,
                   default=MODEL1_ROOT / "data" / "eval")
    p.add_argument("--labels-csv", type=Path,
                   default=MODEL1_ROOT / "data" / "eval" / "labels.csv")
    p.add_argument("--alphas", default="0.30,0.50,0.70,0.90",
                   help="comma-separated alpha (TinySTR weight) values to sweep")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--tta", type=int, default=0,
                   help="If >0, run TinySTR under N spatially-aligned augmentations and average softmaxes")
    return p.parse_args()


def assert_charsets_match():
    if TINYSTR_CHARSET != CLS_CHARSET:
        # Both should be 62-char "0-9A-Za-z"; warn if not.
        print(f"WARN: charset mismatch")
        print(f"  TinySTR: {TINYSTR_CHARSET}")
        print(f"  CharCLS: {CLS_CHARSET}")
        # Build alignment: charcls_idx -> tinystr_idx
        return [TINYSTR_CHARSET.index(c) for c in CLS_CHARSET]
    return None  # identical, no remap needed


class EvalDir(Dataset):
    """Read images named in labels.csv. Returns (filename, image_tensor, label)."""
    def __init__(self, root: Path, labels_csv: Path, img_size: tuple[int, int] = (60, 200)):
        self.root = Path(root)
        with labels_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f); next(reader)
            self.rows = [(fn, lbl) for fn, lbl in reader if fn]
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fn, lbl = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        return fn, self.tf(img), lbl


def collate(batch):
    fns, imgs, lbls = zip(*batch)
    return list(fns), torch.stack(imgs, 0), list(lbls)


def _gaussian_blur_3(x: torch.Tensor) -> torch.Tensor:
    k1d = torch.tensor([0.27406862, 0.45186275, 0.27406862], device=x.device, dtype=x.dtype)
    k2d = (k1d[:, None] * k1d[None, :]).view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    return F.conv2d(x, k2d, padding=1, groups=x.shape[1])


def _tta_transforms(n_augs: int) -> list[tuple[str, callable]]:
    """Spatially-aligned tensor-space augmentations. Order matches Model 1.5."""
    augs = [
        ("identity",  lambda x: x),
        ("bright+",   lambda x: (x + 0.10).clamp(-1, 1)),
        ("bright-",   lambda x: (x - 0.10).clamp(-1, 1)),
        ("contrast+", lambda x: (x * 1.06).clamp(-1, 1)),
        ("contrast-", lambda x: (x * 0.94).clamp(-1, 1)),
        ("blur",      _gaussian_blur_3),
    ]
    return augs[:n_augs]


@torch.no_grad()
def tinystr_softmax(model: TinySTR, loader: DataLoader, device,
                    tta: int = 0) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    model.eval()
    sm: dict[str, torch.Tensor] = {}
    labels: dict[str, str] = {}
    augs = _tta_transforms(tta) if tta and tta > 1 else [("identity", lambda x: x)]
    if tta and tta > 1:
        print(f"  TTA: averaging over {len(augs)} variants {[a[0] for a in augs]}")
    for ai, (aug_name, aug_fn) in enumerate(augs):
        for fns, imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            imgs_aug = aug_fn(imgs)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(imgs_aug)
            probs = logits.float().softmax(dim=-1).cpu()  # [B, 5, 62]
            for fn, row, lbl in zip(fns, probs, lbls):
                if fn not in sm:
                    sm[fn] = row.clone()
                    labels[fn] = lbl
                else:
                    sm[fn] = sm[fn] + row
        if tta and tta > 1:
            print(f"    [{ai+1}/{len(augs)}] {aug_name} done")
    if len(augs) > 1:
        for fn in sm:
            sm[fn] = sm[fn] / len(augs)
    return sm, labels


def decode_ensemble(t_probs: torch.Tensor, c_probs: torch.Tensor, alpha: float,
                    cls_to_tinystr: list[int] | None) -> str:
    """Per-position log-prob mix. t_probs [5,62] in TinySTR charset order;
    c_probs [5,62] in CharCLS charset order. Returns 5-char string in TinySTR charset."""
    eps = 1e-9
    if cls_to_tinystr is not None:
        # reorder c_probs into TinySTR charset order
        c_probs = c_probs[:, cls_to_tinystr]
    log_mix = alpha * torch.log(t_probs + eps) + (1.0 - alpha) * torch.log(c_probs + eps)
    idx = log_mix.argmax(dim=-1)
    return "".join(TINYSTR_CHARSET[int(i)] for i in idx)


def score(preds: dict[str, str], labels: dict[str, str]) -> dict:
    common = sorted(set(preds) & set(labels))
    n_exact = sum(1 for k in common if preds[k] == labels[k])
    char_total = char_correct = 0
    confusions: Counter = Counter()
    for k in common:
        p, l = preds[k], labels[k]
        for i in range(max(len(p), len(l))):
            char_total += 1
            pc = p[i] if i < len(p) else "<gap>"
            lc = l[i] if i < len(l) else "<gap>"
            if pc == lc:
                char_correct += 1
            else:
                confusions[(lc, pc)] += 1
    return {
        "n": len(common),
        "word": n_exact / max(1, len(common)),
        "char": char_correct / max(1, char_total),
        "confusions": confusions,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    cls_to_tinystr = assert_charsets_match()
    if cls_to_tinystr is None:
        print(f"charsets identical: {TINYSTR_CHARSET}")

    # --- TinySTR pass ---
    print(f"\n[TinySTR] loading {args.tinystr_ckpt}")
    tckpt = torch.load(args.tinystr_ckpt, map_location=device, weights_only=False)
    targs = tckpt.get("args", {})
    model_kwargs = {
        "embed_dim": targs.get("embed_dim", 256),
        "encoder_depth": targs.get("encoder_depth", 6),
        "query_self_attn_layers": targs.get("query_self_attn_layers", 0),
    }
    tinystr = TinySTR(**model_kwargs).to(device)
    tinystr.load_state_dict(tckpt["model"])
    n = sum(p.numel() for p in tinystr.parameters())
    print(f"  TinySTR {n/1e6:.2f}M from epoch {tckpt.get('epoch')}  {model_kwargs}")

    ds = EvalDir(args.eval_dir, args.labels_csv)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate)
    print(f"  running on {len(ds)} captchas (tta={args.tta})")
    t0 = time.time()
    t_sm, labels = tinystr_softmax(tinystr, loader, device, tta=args.tta)
    print(f"  tinystr softmax done in {time.time()-t0:.2f}s")
    del tinystr; torch.cuda.empty_cache()

    # --- char-CNN pass (uses precomputed boxes from Model 1.5) ---
    print(f"\n[char-CNN] loading {args.charcls_ckpt}")
    cls_model, crop_size, pad_frac = load_charcls(args.charcls_ckpt)
    print(f"  crop={crop_size} pad_frac={pad_frac}")
    print(f"  using boxes from {args.boxes_jsonl}")
    t0 = time.time()
    letter_sm, case_sm = charcls_softmax_for_dir(
        cls_model, args.eval_dir, args.boxes_jsonl, crop_size, pad_frac
    )
    print(f"  charcls softmax for {len(letter_sm)} images in {time.time()-t0:.2f}s")

    # --- TinySTR alone (sanity check) ---
    preds_tinystr = {fn: "".join(TINYSTR_CHARSET[int(i)] for i in t_sm[fn].argmax(dim=-1))
                     for fn in t_sm}
    rt = score(preds_tinystr, labels)
    print(f"\n[TinySTR alone (greedy)]  word={rt['word']*100:6.2f}%  char={rt['char']*100:6.2f}%")

    # --- Ensemble sweep ---
    common = sorted(set(t_sm) & set(letter_sm) & set(labels))
    print(f"\nensemble eval over {len(common)} captchas (intersection of TinySTR / charcls / labels)")

    rows = []
    for a_str in args.alphas.split(","):
        alpha = float(a_str.strip())
        preds = {fn: decode_ensemble(t_sm[fn], letter_sm[fn], alpha, cls_to_tinystr)
                 for fn in common}
        r = score(preds, labels)
        rows.append((alpha, r))
        print(f"  alpha={alpha:.2f}   word={r['word']*100:6.2f}%   char={r['char']*100:6.2f}%")

    # --- Summary table ---
    print("\n=== SUMMARY ===")
    print(f"{'config':<32} {'word %':>8} {'char %':>8}")
    print(f"{'TinySTR alone':<32} {rt['word']*100:>8.2f} {rt['char']*100:>8.2f}")
    for alpha, r in rows:
        print(f"{'TinySTR + charcls a=%.2f' % alpha:<32} {r['word']*100:>8.2f} {r['char']*100:>8.2f}")
    print(f"{'PARSeq alone (baseline)':<32} {PARSEQ_BASELINE['word_acc']*100:>8.2f} {PARSEQ_BASELINE['char_acc']*100:>8.2f}")
    print(f"{'Model 1.5 (PARSeq + charcls)':<32} {MODEL_15_BASELINE['word_acc']*100:>8.2f} {MODEL_15_BASELINE['char_acc']*100:>8.2f}")

    # Best alpha confusions
    best = max(rows, key=lambda x: x[1]['word'])
    alpha_best, r_best = best
    print(f"\ntop confusions  (best alpha={alpha_best:.2f},  word={r_best['word']*100:.2f}%)")
    for (gt, pr), c in r_best['confusions'].most_common(15):
        print(f"  {gt!r:>6} -> {pr!r:<6} : {c}")


if __name__ == "__main__":
    main()
