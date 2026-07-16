"""Shared eval helpers: load (image_path, label) pairs from a directory of
captchas where the *filename stem* is the label, run a callable predictor,
score case-sensitive exact-match + char-level + simple confusion."""
from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
from tqdm import tqdm


EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_labelled"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_eval_pairs(eval_dir: Path = EVAL_DIR) -> list[tuple[Path, str]]:
    """Each PNG's stem is the ground-truth captcha text (case-sensitive)."""
    pairs = []
    for p in sorted(eval_dir.iterdir()):
        if p.suffix.lower() != ".png":
            continue
        label = p.stem
        if len(label) != 5:
            continue
        pairs.append((p, label))
    return pairs


def score(preds: dict[str, str], labels: dict[str, str]) -> dict:
    common = sorted(set(preds) & set(labels))
    n = len(common)
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
        "n": n,
        "exact": n_exact / n * 100 if n else 0.0,
        "char": char_correct / char_total * 100 if char_total else 0.0,
        "n_exact": n_exact,
        "confusions_top": confusions.most_common(15),
    }


def write_results(model_name: str, preds: dict[str, str], labels: dict[str, str], elapsed_s: float) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{model_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "prediction", "match"])
        for k in sorted(preds):
            l = labels.get(k, "")
            w.writerow([k, l, preds[k], int(preds[k] == l)])
    s = score(preds, labels)
    s["model"] = model_name
    s["elapsed_s"] = elapsed_s
    s["per_image_ms"] = elapsed_s * 1000 / max(1, s["n"])
    print(f"\n=== {model_name} ===")
    print(f"  n={s['n']}  exact={s['exact']:.2f}%  char={s['char']:.2f}%  ({s['n_exact']}/{s['n']})")
    print(f"  time: {elapsed_s:.1f}s  ({s['per_image_ms']:.1f} ms/img)")
    print(f"  top confusions: {s['confusions_top'][:8]}")
    summary_path = RESULTS_DIR / "summary.csv"
    header_needed = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["model", "n", "exact_pct", "char_pct", "elapsed_s", "ms_per_img"])
        w.writerow([model_name, s["n"], f"{s['exact']:.2f}", f"{s['char']:.2f}",
                    f"{elapsed_s:.1f}", f"{s['per_image_ms']:.1f}"])
    return s


def run_predictor(model_name: str, predict_fn: Callable[[Image.Image], str]) -> dict:
    """predict_fn takes a PIL.Image -> predicted string (we strip whitespace)."""
    pairs = load_eval_pairs()
    print(f"loaded {len(pairs)} eval pairs")
    preds: dict[str, str] = {}
    labels: dict[str, str] = {p.name: lbl for p, lbl in pairs}
    t0 = time.time()
    for path, _ in tqdm(pairs, desc=model_name):
        img = Image.open(path).convert("RGB")
        try:
            pred = predict_fn(img)
        except Exception as e:
            pred = f"<ERR:{type(e).__name__}>"
        preds[path.name] = (pred or "").strip()
    elapsed = time.time() - t0
    return write_results(model_name, preds, labels, elapsed)


def run_predictor_batch(
    model_name: str,
    batch_predict_fn: Callable[[list[Image.Image]], list[str]],
    batch_size: int = 16,
) -> dict:
    pairs = load_eval_pairs()
    print(f"loaded {len(pairs)} eval pairs")
    preds: dict[str, str] = {}
    labels: dict[str, str] = {p.name: lbl for p, lbl in pairs}
    t0 = time.time()
    for i in tqdm(range(0, len(pairs), batch_size), desc=model_name):
        chunk = pairs[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p, _ in chunk]
        try:
            outs = batch_predict_fn(imgs)
        except Exception as e:
            outs = [f"<ERR:{type(e).__name__}>"] * len(chunk)
        for (p, _), out in zip(chunk, outs):
            preds[p.name] = (out or "").strip()
    elapsed = time.time() - t0
    return write_results(model_name, preds, labels, elapsed)
