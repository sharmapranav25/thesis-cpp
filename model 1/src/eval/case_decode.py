"""Case-aware decoding experiments on a trained PARSeq checkpoint.

Loads the checkpoint, runs the eval set once to collect per-position softmax,
then applies a battery of decoding strategies offline. Reports word/char
accuracy per strategy so we can compare without retraining.

Strategies:
  A. greedy            — current baseline. Argmax per position.
  B. force5            — force exactly 5 chars (no EOS truncation).
  C. case-marginal     — for case-pair letters, sum P(c)+P(C) then pick the
                         case with higher individual P. Targets case errors.
  D. force5+marginal   — combine B + C.
  E. case-flip-bias    — apply per-character empirical case bias from synth
                         val (train-time) confusion. NOT IMPLEMENTED HERE
                         (placeholder for second pass if A-D show signal).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

PARSEQ_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "parseq"
sys.path.insert(0, str(PARSEQ_ROOT))

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.parseq.train import build_parseq, _to_rgb, CHARSET
from src.parseq.infer import ImageDir


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Letters that look (almost) the same upper vs lower in our font set.
# These are the case errors observed in the confusion table.
CASE_PAIRS = list("cCkKmMoOpPsSuUvVwWxXyYzZ")
# Map each pairable letter -> its case-flipped counterpart.
PAIR_OF: dict[str, str] = {}
for i in range(0, len(CASE_PAIRS), 2):
    lo = CASE_PAIRS[i].lower()
    up = CASE_PAIRS[i].upper()
    PAIR_OF[lo] = up
    PAIR_OF[up] = lo


def char_score(pred: str, gold: str) -> tuple[int, int]:
    n = max(len(pred), len(gold))
    correct = sum(1 for i in range(n) if i < len(pred) and i < len(gold) and pred[i] == gold[i])
    return correct, n


# ---------------------------------------------------------------------------
# decoders. each takes a softmax tensor over [L, V] and a tokenizer, returns str.
# ---------------------------------------------------------------------------

def _itos_list(tok) -> list[str]:
    return list(tok._itos)


def decode_greedy(probs: torch.Tensor, tok) -> str:
    itos = _itos_list(tok)
    eos = tok.eos_id
    out = []
    for t in range(probs.shape[0]):
        i = int(probs[t].argmax().item())
        if i == eos:
            break
        out.append(itos[i])
    return "".join(out)


def detect_fixed_length(labels_csv: Path, threshold: float = 0.90) -> int | None:
    """Read training-label lengths. If a single length covers >= threshold of
    rows, return it; else return None (signal: variable length, use EOS)."""
    lengths: Counter = Counter()
    with labels_csv.open("r", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if not row:
                continue
            lengths[len(row[1])] += 1
    total = sum(lengths.values())
    if total == 0:
        return None
    most_common, count = lengths.most_common(1)[0]
    if count / total >= threshold:
        return most_common
    return None


def decode_force_length(probs: torch.Tensor, tok, n: int) -> str:
    """Emit exactly n chars by zeroing EOS probability."""
    itos = _itos_list(tok)
    V = probs.shape[1]
    out = []
    for t in range(min(probs.shape[0], n)):
        masked = probs[t].clone()
        masked[tok.eos_id] = -1.0
        i = int(masked.argmax().item())
        if i >= V:
            continue
        out.append(itos[i])
    return "".join(out)


def decode_smart(probs: torch.Tensor, tok, fixed_n: int | None) -> str:
    """If a fixed length was detected from training labels, force that length.
    Otherwise fall back to EOS-aware greedy decoding."""
    if fixed_n is None:
        return decode_greedy(probs, tok)
    return decode_force_length(probs, tok, fixed_n)


# ---------------------------------------------------------------------------
# ensemble decoding (PARSeq logits + char-CNN logits, log-prob mix)
# ---------------------------------------------------------------------------

def _charset_index_maps(tok, charcls_charset: str) -> tuple[list[int], list[str]]:
    """Returns (parseq_idx_for_charcls_pos, charset_chars).
    parseq_idx_for_charcls_pos[i] is the PARSeq vocab index for char
    charcls_charset[i]. Used to align the two distributions."""
    parseq_idx = [tok._stoi[c] for c in charcls_charset]
    return parseq_idx, list(charcls_charset)


def decode_ensemble(
    parseq_probs: torch.Tensor,        # [L_p, V_parseq]
    charcls_probs: torch.Tensor,       # [5, 62] aligned to charcls_charset
    parseq_idx_for_cls: list[int],     # mapping
    chars: list[str],                  # charcls_charset as a list
    fixed_n: int,
    alpha: float,                      # weight on PARSeq; (1-α) on CharCLS
) -> str:
    """Per-position log-prob mix; greedy argmax over the 62 charset chars."""
    # Pull per-char parseq probs aligned to charcls order: [fixed_n, 62]
    L = min(fixed_n, parseq_probs.shape[0])
    p_p = parseq_probs[:L][:, parseq_idx_for_cls]                 # [L, 62]
    p_c = charcls_probs[:L]                                       # [L, 62]
    eps = 1e-9
    log_combined = alpha * torch.log(p_p + eps) + (1 - alpha) * torch.log(p_c + eps)
    idx = log_combined.argmax(dim=-1)
    out = "".join(chars[int(i.item())] for i in idx)
    # if fixed_n > L (shouldn't happen with PARSeq max_label_length+1) pad with greedy on charcls alone
    if len(out) < fixed_n:
        for t in range(L, fixed_n):
            i = int(p_c[t].argmax().item())
            out += chars[i]
    return out


def decode_disagreement_override(
    parseq_probs: torch.Tensor,
    charcls_probs: torch.Tensor,
    parseq_idx_for_cls: list[int],
    chars: list[str],
    fixed_n: int,
    cls_threshold: float = 0.6,
) -> str:
    """Use PARSeq's pick unless CharCLS is confident AND disagrees on the
    full letter identity. Broad override — can change letter, not just case."""
    L = min(fixed_n, parseq_probs.shape[0])
    out = []
    for t in range(L):
        p_p = parseq_probs[t][parseq_idx_for_cls]
        p_c = charcls_probs[t]
        p_idx = int(p_p.argmax().item())
        c_idx = int(p_c.argmax().item())
        if p_idx == c_idx or float(p_c[c_idx].item()) < cls_threshold:
            out.append(chars[p_idx])
        else:
            out.append(chars[c_idx])
    while len(out) < fixed_n:
        out.append(chars[int(charcls_probs[len(out)].argmax().item())])
    return "".join(out)


def decode_caseflip_override(
    parseq_probs: torch.Tensor,
    case_probs: torch.Tensor,            # [5, 3]: UPPER, LOWER, DIGIT
    parseq_idx_for_cls: list[int],
    chars: list[str],
    fixed_n: int,
    case_threshold: float = 0.65,
) -> str:
    """Targeted override: keep PARSeq's letter identity; allow the case-CNN
    to *only* flip case when it is confident and disagrees with PARSeq's
    case. Cannot change letter identity (so won't introduce new error modes
    beyond case)."""
    from src.charcls.model import CASE_UPPER, CASE_LOWER, CASE_DIGIT
    L = min(fixed_n, parseq_probs.shape[0])
    out = []
    for t in range(L):
        p_p = parseq_probs[t][parseq_idx_for_cls]
        p_idx = int(p_p.argmax().item())
        ch = chars[p_idx]
        # PARSeq's implied case
        if ch.isdigit():
            parseq_case = CASE_DIGIT
        elif ch.isupper():
            parseq_case = CASE_UPPER
        else:
            parseq_case = CASE_LOWER
        case_idx = int(case_probs[t].argmax().item())
        case_conf = float(case_probs[t][case_idx].item())
        if case_idx != parseq_case and case_conf >= case_threshold and not ch.isdigit():
            # flip case (only sensible for letters; leave digits alone)
            if case_idx == CASE_UPPER and ch.islower():
                ch = ch.upper()
            elif case_idx == CASE_LOWER and ch.isupper():
                ch = ch.lower()
        out.append(ch)
    while len(out) < fixed_n:
        out.append("?")
    return "".join(out)


def decode_case_marginal(probs: torch.Tensor, tok, fixed_n: int | None = None) -> str:
    """For each position:
      1. Find the highest-probability *letter or digit* by case-marginalizing.
      2. Once a letter is chosen, pick the case (lower/upper) with higher P
         in that position's softmax.
    If fixed_n is None: stop at EOS. Else: emit exactly fixed_n chars.
    """
    itos = _itos_list(tok)
    stoi = tok._stoi
    out = []
    L = fixed_n if fixed_n is not None else probs.shape[0]
    for t in range(min(probs.shape[0], L)):
        p = probs[t]
        if fixed_n is None:
            i_argmax = int(p.argmax().item())
            if i_argmax == tok.eos_id:
                break
            if i_argmax in (tok.bos_id, tok.pad_id):
                continue
        # marginal score per "casefolded letter": for paired chars, sum the
        # two cases; for unpaired chars (digits, single-case letters), keep
        # the standalone prob.
        best_score = -1.0
        best_char = ""
        for ch in CHARSET:
            idx = stoi[ch]
            score = float(p[idx].item())
            if ch in PAIR_OF:
                pair_idx = stoi[PAIR_OF[ch]]
                pair_score = float(p[pair_idx].item())
                # only credit the pair to the higher of the two so we don't
                # double-count when picking the winner letter.
                if score < pair_score:
                    continue
                score = score + pair_score
            if score > best_score:
                best_score = score
                best_char = ch
        # best_char is the higher-prob case already (we filtered above).
        out.append(best_char)
    return "".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def collect_softmax(model, loader, max_len: int) -> tuple[list[str], list[torch.Tensor]]:
    """Returns parallel lists: image filenames, per-image softmax tensors [L, V]."""
    names_all: list[str] = []
    probs_all: list[torch.Tensor] = []
    with torch.inference_mode():
        for imgs, names in loader:
            imgs = imgs.cuda(non_blocking=True)
            logits = model(imgs)  # [B, L, V]
            probs = logits.softmax(-1).cpu()
            for i, n in enumerate(names):
                names_all.append(n)
                probs_all.append(probs[i, :max_len])
    return names_all, probs_all


# Test-time augmentations applied AFTER normalization (data is mean=0.5,
# std=0.5 → output range ~[-1, 1]). All transforms must preserve spatial
# alignment so per-position softmaxes from each augmentation can be averaged.
def _tta_transforms(n_augs: int) -> list[tuple[str, callable]]:
    augs: list[tuple[str, callable]] = [
        ("identity",  lambda x: x),
        ("bright+",   lambda x: (x + 0.10).clamp(-1, 1)),
        ("bright-",   lambda x: (x - 0.10).clamp(-1, 1)),
        ("contrast+", lambda x: (x * 1.06).clamp(-1, 1)),
        ("contrast-", lambda x: (x * 0.94).clamp(-1, 1)),
        ("blur",      _gaussian_blur_3),
    ]
    return augs[:n_augs]


def _gaussian_blur_3(x: torch.Tensor) -> torch.Tensor:
    # 3x3 separable gauss with sigma~0.6, applied per-channel.
    import torch.nn.functional as F
    k1d = torch.tensor([0.27406862, 0.45186275, 0.27406862], device=x.device, dtype=x.dtype)
    k2d = (k1d[:, None] * k1d[None, :]).view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    return F.conv2d(x, k2d, padding=1, groups=x.shape[1])


def collect_softmax_tta(
    model, loader, max_len: int, n_augs: int = 5,
) -> tuple[list[str], list[torch.Tensor]]:
    """Same return as `collect_softmax` but averages softmax over `n_augs`
    spatially-aligned augmentations of each image. ~n_augs× slower."""
    augs = _tta_transforms(n_augs)
    names_all: list[str] = []
    sum_by_name: dict[str, torch.Tensor] = {}
    print(f"  TTA: averaging over {len(augs)} variants {[a[0] for a in augs]}", flush=True)
    for ai, (aug_name, aug_fn) in enumerate(augs):
        with torch.inference_mode():
            for imgs, names in loader:
                imgs = imgs.cuda(non_blocking=True)
                imgs_aug = aug_fn(imgs)
                logits = model(imgs_aug)
                probs = logits.softmax(-1).cpu()
                for i, n in enumerate(names):
                    p = probs[i, :max_len]
                    if n not in sum_by_name:
                        sum_by_name[n] = p.clone()
                        if ai == 0:
                            names_all.append(n)
                    else:
                        sum_by_name[n] = sum_by_name[n] + p
        print(f"    [{ai+1}/{len(augs)}] {aug_name} done", flush=True)
    avg_probs = [sum_by_name[n] / float(len(augs)) for n in names_all]
    return names_all, avg_probs


def load_labels(label_csv: Path) -> dict[str, str]:
    out = {}
    with label_csv.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row:
                continue
            out[row[0]] = row[1]
    return out


def evaluate(name: str, preds: dict[str, str], labels: dict[str, str]) -> dict:
    common = sorted(set(preds) & set(labels))
    n_exact = 0
    char_correct = 0
    char_total = 0
    confusions: Counter = Counter()
    for k in common:
        p, l = preds[k], labels[k]
        if p == l:
            n_exact += 1
        cc, ct = char_score(p, l)
        char_correct += cc
        char_total += ct
        for i in range(max(len(p), len(l))):
            pc = p[i] if i < len(p) else "<gap>"
            lc = l[i] if i < len(l) else "<gap>"
            if pc != lc:
                confusions[(lc, pc)] += 1
    return {
        "name": name,
        "n": len(common),
        "exact": n_exact,
        "exact_pct": 100 * n_exact / max(1, len(common)),
        "char_acc": 100 * char_correct / max(1, char_total),
        "top_confusions": confusions.most_common(15),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True, help="dir of eval images")
    ap.add_argument("--labels", type=Path, required=True, help="ground-truth CSV")
    ap.add_argument("--config", type=Path, default=Path("configs/parseq.yaml"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--save-probs", type=Path, default=None,
                    help="optional npz path to save per-image softmaxes")
    ap.add_argument("--train-labels", type=Path,
                    default=Path("data/synth_mybb/labels.csv"),
                    help="CSV used to detect fixed-length captcha format")
    ap.add_argument("--length-threshold", type=float, default=0.90)
    ap.add_argument("--detector-ckpt", type=Path, default=None,
                    help="optional detector checkpoint for char crops")
    ap.add_argument("--charcls-ckpt", type=str, default=None,
                    help="optional char-CNN checkpoint(s); comma-separated for multi-scale ensemble")
    ap.add_argument("--alphas", default="0.3,0.5,0.7,0.9",
                    help="comma-separated alpha values for PARSeq weight in ensemble")
    ap.add_argument("--cls-threshold", type=float, default=0.6,
                    help="confidence threshold for disagreement-override")
    ap.add_argument("--tta", type=int, default=0,
                    help="if >0, run PARSeq under N spatially-aligned augmentations and average softmaxes")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    model = build_parseq(cfg)
    state = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model = model.cuda().eval()

    ds = ImageDir(args.src, tuple(cfg["img_size"]))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=4, pin_memory=True)

    print(f"running model on {len(ds)} images...")
    if args.tta and args.tta > 1:
        names, probs = collect_softmax_tta(
            model, loader, max_len=cfg["max_label_length"] + 1, n_augs=args.tta
        )
    else:
        names, probs = collect_softmax(model, loader, max_len=cfg["max_label_length"] + 1)
    print(f"  collected softmax for {len(names)} images, vocab={probs[0].shape[1]}")

    if args.save_probs:
        args.save_probs.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_probs,
            names=np.array(names),
            probs=np.stack([p.numpy() for p in probs]),
        )
        print(f"  saved softmax to {args.save_probs}")

    labels = load_labels(args.labels)
    tok = model.tokenizer

    fixed_n = detect_fixed_length(args.train_labels, args.length_threshold)
    if fixed_n is not None:
        print(f"  detected fixed length {fixed_n} from {args.train_labels} "
              f"(>= {args.length_threshold:.0%} of labels)")
    else:
        print(f"  no fixed length detected from {args.train_labels} "
              f"(threshold {args.length_threshold:.0%})  — falling back to EOS-aware decoding")

    strategies: dict[str, callable] = {
        "A. greedy": lambda n, p: decode_greedy(p, tok),
        "B. force-detected": lambda n, p: decode_smart(p, tok, fixed_n),
        "C. case-marginal": lambda n, p: decode_case_marginal(p, tok, fixed_n=None),
        "D. detected+marginal": lambda n, p: decode_case_marginal(p, tok, fixed_n=fixed_n),
    }

    # ---- optional ensemble with detector + char-CNN ---------------------
    if args.detector_ckpt is not None and args.charcls_ckpt is not None:
        if fixed_n is None:
            print("  WARN: ensemble requires fixed_n; skipping ensemble strategies")
        else:
            from src.detector.infer import load_detector, predict_dir
            from src.charcls.infer import load_charcls
            from src.charcls.infer_fast import charcls_softmax_for_dir_fast as charcls_softmax_for_dir

            print(f"  loading detector {args.detector_ckpt}", flush=True)
            det_model = load_detector(args.detector_ckpt)
            box_path = args.src.parent / f"_detector_boxes_{args.src.name}.jsonl"
            with box_path.open("w") as f:
                count = 0
                for nm, boxes in predict_dir(det_model, args.src):
                    f.write(json.dumps({"filename": nm, "bboxes": boxes}) + "\n")
                    count += 1
                print(f"  detector: wrote {count} box records to {box_path}")
            del det_model
            torch.cuda.empty_cache()

            ckpt_paths = [Path(p.strip()) for p in str(args.charcls_ckpt).split(",") if p.strip()]
            print(f"  loading {len(ckpt_paths)} char-CNN checkpoint(s): {[p.name for p in ckpt_paths]}", flush=True)
            letter_sm: dict[str, torch.Tensor] = {}
            case_sm: dict[str, torch.Tensor] = {}
            for ckpt_path in ckpt_paths:
                cls_model, crop_size, pad_frac = load_charcls(ckpt_path)
                print(f"    {ckpt_path.parent.name}: crop_size={crop_size}", flush=True)
                ls_one, cs_one = charcls_softmax_for_dir(
                    cls_model, args.src, box_path, crop_size, pad_frac
                )
                # accumulate (sum, divide later)
                for nm, p in ls_one.items():
                    letter_sm[nm] = letter_sm.get(nm, torch.zeros_like(p)) + p
                for nm, p in cs_one.items():
                    case_sm[nm] = case_sm.get(nm, torch.zeros_like(p)) + p
                del cls_model
                torch.cuda.empty_cache()
            n_models = len(ckpt_paths)
            for nm in letter_sm:
                letter_sm[nm] = letter_sm[nm] / n_models
            for nm in case_sm:
                case_sm[nm] = case_sm[nm] / n_models
            print(f"  char-CNN ensemble ({n_models} scale(s)): softmax for {len(letter_sm)} images")

            from src.charcls.model import CHARSET as CLS_CHARSET
            parseq_idx, chars = _charset_index_maps(tok, CLS_CHARSET)

            for a_str in args.alphas.split(","):
                alpha = float(a_str.strip())
                strategies[f"E. ensemble a={alpha:.2f}"] = (
                    (lambda alpha: lambda nm, p: decode_ensemble(
                        p, letter_sm[nm], parseq_idx, chars, fixed_n, alpha
                    ))(alpha)
                )
            strategies["F. letter override"] = (
                lambda nm, p: decode_disagreement_override(
                    p, letter_sm[nm], parseq_idx, chars, fixed_n,
                    cls_threshold=args.cls_threshold,
                )
            )
            strategies["G. case-flip override"] = (
                lambda nm, p: decode_caseflip_override(
                    p, case_sm[nm], parseq_idx, chars, fixed_n,
                    case_threshold=args.cls_threshold,
                )
            )

    results = []
    for sname, fn in strategies.items():
        preds = {n: fn(n, p) for n, p in zip(names, probs)}
        r = evaluate(sname, preds, labels)
        results.append(r)
        print(f"\n[{sname}]")
        print(f"  exact-match: {r['exact_pct']:6.2f}%   ({r['exact']}/{r['n']})")
        print(f"  char acc:    {r['char_acc']:6.2f}%")
        print(f"  top confusions:")
        for (gt, pr), c in r["top_confusions"][:8]:
            print(f"    {gt!r:>6} -> {pr!r:<6} : {c}")

    print("\n=== summary ===")
    print(f"  {'strategy':<28} {'word%':>8} {'char%':>8}")
    for r in results:
        print(f"  {r['name']:<28} {r['exact_pct']:>8.2f} {r['char_acc']:>8.2f}")


if __name__ == "__main__":
    main()
