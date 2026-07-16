"""Generate a labeled synth pool via the real PHP-GD generator (src/php_gen.php).

We choose the text (labels known), write a jobs list, and let PHP-GD render every
image with the actual captcha primitives (imagettftext + wave warp + noise). This
is the faithful version of the domain-knowledge prior — pixel-accurate to how real
PHP captchas render, source-free (generic fonts, open-source rendering).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
from pathlib import Path

import yaml

PHP = r"C:/xampp/php/php.exe"
FONTS = [
    r"E:/pranav/model 1/.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
    r"E:/pranav/model 1/.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
    r"C:/Windows/Fonts/verdanab.ttf",
    r"C:/Windows/Fonts/arialbd.ttf",
]

# per-scheme PHP render config (style knobs); wave = the captcha sine warp
SCHEME_CFG = {
    "cryptbb": dict(charset_style="vivid", bg="pale", min_size_frac=0.66, max_size_frac=0.86,
                    max_rot=14, wave=dict(on=True, ampx=[0.0, 1.2], ampy=[0.6, 2.4], per=[0.8, 1.8]),
                    lines=[0, 3], arcs=[0, 2], dots=[0.0, 0.012], grid=True),
    "darkbay": dict(charset_style="bw", bg="dark", min_size_frac=0.58, max_size_frac=0.88,
                    max_rot=13, wave=dict(on=True, ampx=[0.0, 1.2], ampy=[0.5, 2.0], per=[0.8, 1.6]),
                    lines=[0, 2], arcs=[0, 1], dots=[0.0, 0.010]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="configs/corpus.yaml")
    ap.add_argument("--scheme", default="cryptbb")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--fixed-len", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="parallel PHP processes (the wave-warp is CPU-heavy)")
    args = ap.parse_args()

    s = yaml.safe_load(open(args.corpus))["schemes"][args.scheme]
    W, H = s["canvas"]; charset = s["charset"]
    lo, hi = s["len_range"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fonts = [f for f in FONTS if Path(f).is_file()]

    cfg = dict(W=W, H=H, fonts=fonts, **SCHEME_CFG.get(args.scheme, SCHEME_CFG["cryptbb"]))
    (out / "_php_cfg.json").write_text(json.dumps(cfg))

    rng = random.Random(args.seed)
    rows, jobs = [], []
    for i in range(args.n):
        n = args.fixed_len or rng.randint(lo, hi)
        text = "".join(rng.choice(charset) for _ in range(n))
        fn = f"{i:06d}_{text}.png"
        rows.append((fn, text))
        jobs.append(f"{text}\t{(out / fn).as_posix()}")
    # split jobs across parallel PHP workers (wave-warp is CPU-heavy)
    nw = max(1, args.workers)
    chunks = [jobs[i::nw] for i in range(nw)]
    cfg_path = out / "_php_cfg.json"
    procs = []
    for wi, ch in enumerate(chunks):
        jf = out / f"_jobs_{wi}.tsv"
        jf.write_text("\n".join(ch), encoding="utf-8")
        procs.append(subprocess.Popen([PHP, "src/php_gen.php", str(jf), str(cfg_path)],
                                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL))
    print(f"rendering {args.n} via PHP-GD ({W}x{H}, {args.scheme}) across {nw} workers ...", flush=True)
    for p in procs:
        p.wait()

    made = sorted(out.glob("*.png"))
    rows = [(fn, lbl) for fn, lbl in rows if (out / fn).exists()]
    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filename", "label"]); w.writerows(rows)
    print(f"wrote {len(rows)} PHP-rendered labeled samples to {out}", flush=True)


if __name__ == "__main__":
    main()
