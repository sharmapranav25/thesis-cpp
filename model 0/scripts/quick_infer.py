"""Quick inference probe: load best PARSeq checkpoint, predict on a tiny
sample from each library, and print (truth, prediction) pairs plus accuracy
stratified by length and library.

Run from model 0 root:
    python scripts/quick_infer.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Same DLL hack as train.py for nltk on Anaconda
_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

PARSEQ_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "parseq"
sys.path.insert(0, str(PARSEQ_ROOT))

import torch
from PIL import Image
from torchvision import transforms as T
from strhub.models.parseq.system import PARSeq  # noqa: E402


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def load_ckpt(path: Path) -> PARSeq:
    model = PARSeq.load_from_checkpoint(path, charset_train=CHARSET, charset_test=CHARSET)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def parse_lib_from_params(tsv_row_params: str) -> str:
    try:
        d = json.loads(tsv_row_params)
        return d.get("lib", "?")
    except Exception:
        return "?"


def build_samples(corpus_root: Path, per_lib: int = 50) -> list[tuple[str, str, str]]:
    """Pull `per_lib` random samples per library by walking shard TSVs."""
    import random
    rng = random.Random(13)
    rows = []
    for shard in sorted(corpus_root.iterdir()):
        if not shard.is_dir() or not shard.name.startswith("shard_"):
            continue
        tsv = shard / "labels.tsv"
        if not tsv.exists():
            continue
        with tsv.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                fn, label, _, params = parts[0], parts[1], parts[2], parts[3]
                lib = parse_lib_from_params(params)
                rows.append((shard.name + "/" + fn, label, lib))
    rng.shuffle(rows)
    by_lib = defaultdict(list)
    for r in rows:
        if len(by_lib[r[2]]) < per_lib:
            by_lib[r[2]].append(r)
    out = []
    for lib in sorted(by_lib):
        out.extend(by_lib[lib])
    return out


def main():
    corpus = Path("train_corpus")
    ckpt_dir = Path("checkpoints/parseq")
    best = sorted(ckpt_dir.glob("parseq-epoch=*-val_accuracy=*.ckpt"))
    if not best:
        raise SystemExit("no best ckpt found")
    # last in lex order = highest val_accuracy (since same epoch field padding)
    best_ckpt = sorted(best, key=lambda p: float(p.stem.split("=")[-1]))[-1]
    print(f"using ckpt: {best_ckpt}")

    model = load_ckpt(best_ckpt)
    img_size = (32, 128)
    tf = T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize(img_size, antialias=True),
        T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    samples = build_samples(corpus, per_lib=40)
    print(f"samples: {len(samples)} ({len({s[2] for s in samples})} libs)")

    by_lib_correct = defaultdict(int)
    by_lib_total   = defaultdict(int)
    by_len_correct = defaultdict(int)
    by_len_total   = defaultdict(int)

    show_n = 15
    shown = 0
    with torch.no_grad():
        # batch by 32
        batch = []
        for s in samples:
            batch.append(s)
            if len(batch) == 32:
                imgs = torch.stack([tf(Image.open(corpus / fn)) for fn, _, _ in batch])
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                logits = model(imgs)
                probs = logits.softmax(-1)
                preds, _ = model.tokenizer.decode(probs)
                for (fn, gt, lib), pred in zip(batch, preds):
                    pred = model.charset_adapter(pred)
                    ok = pred == gt
                    by_lib_total[lib] += 1
                    by_len_total[len(gt)] += 1
                    if ok:
                        by_lib_correct[lib] += 1
                        by_len_correct[len(gt)] += 1
                    if shown < show_n:
                        flag = "OK" if ok else "  "
                        print(f"  {flag} lib={lib:14s} gt={gt!r:14s} pred={pred!r}")
                        shown += 1
                batch.clear()

    print("\n--- per-library exact-match accuracy ---")
    for lib in sorted(by_lib_total):
        c = by_lib_correct[lib]; n = by_lib_total[lib]
        print(f"  {lib:14s} {c}/{n} = {100.0*c/n:.1f}%")
    print("\n--- per-length exact-match accuracy ---")
    for L in sorted(by_len_total):
        c = by_len_correct[L]; n = by_len_total[L]
        print(f"  len={L} {c}/{n} = {100.0*c/n:.1f}%")
    total_c = sum(by_lib_correct.values()); total_n = sum(by_lib_total.values())
    print(f"\n  total: {total_c}/{total_n} = {100.0*total_c/total_n:.1f}%")


if __name__ == "__main__":
    main()
