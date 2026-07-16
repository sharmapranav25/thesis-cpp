"""M3 — turn the trained restyler G into a labeled synthetic pool.

For each of N random strings: render styled glyphs (label = ours), composite on
a recreated background, run G -> ŷ, resize to native canvas, save png + label.
Every pair is labeled by construction.

Optional --verify: keep only samples the cold-start reader reads correctly
(guards label integrity — a mislabeled pool is worse than none, GOAL.md). Report
the keep rate.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.render_source import SourceRenderer
from src.data import SourceDataset, source_collate
from src.model import Generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/restyler.yaml"))
    ap.add_argument("--corpus", type=Path, default=Path("configs/corpus.yaml"))
    ap.add_argument("--scheme", default="cryptbb")
    ap.add_argument("--ckpt", default=None, help="restyler ckpt (…_latest.pt); omit with --no-gan")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--out", type=Path, default=Path("data/synth_pool/cryptbb"))
    ap.add_argument("--verify", action="store_true", help="keep only reader-confirmed labels")
    ap.add_argument("--reader-ckpt", default="data/ckpts/reader_r0.ckpt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fixed-len", type=int, default=None,
                    help="render all labels this length (match eval; enables force-len decode)")
    ap.add_argument("--no-gan", action="store_true",
                    help="output the raw composite (skip G) — for the ±GAN ablation")
    ap.add_argument("--bg-mode", default=None, help="override bg_mode (procedural|lama|mix)")
    ap.add_argument("--distort", default=None, help="PHP-GD distortion profile (php_generic|php_wavy|php_mono|none)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    corpus = yaml.safe_load(args.corpus.read_text())
    s = corpus["schemes"][args.scheme]
    device = "cuda"
    Wc, Hc = s["canvas"]
    workH, workW = cfg["work_hw"]

    renderer = SourceRenderer(canvas=(Wc, Hc), charset=s["charset"],
                              len_range=tuple(s["len_range"]), seed=123,
                              fixed_len=args.fixed_len, glyph_style=s.get("glyph_style", "vivid"))
    # per-scheme background: schemes with a lama_dir use it; others procedural.
    bg_mode = args.bg_mode or (cfg.get("bg_mode", "mix") if s.get("bg_style", "pale") != "dark" else "procedural")
    ds = SourceDataset(renderer, (workH, workW), length=args.n,
                       bg_mode=bg_mode, lama_dir=cfg.get("lama_dir"),
                       p_procedural=cfg.get("p_procedural", 0.6), seed=123,
                       bg_style=s.get("bg_style", "pale"),
                       distort_profile=args.distort or s.get("distort_profile", "none"))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=6,
                        collate_fn=source_collate, pin_memory=True)

    G = None
    if not args.no_gan:
        G = Generator(base=cfg.get("base", 64), residual=cfg.get("residual", False),
                      residual_scale=cfg.get("residual_scale", 0.5)).to(device).eval()
        state = torch.load(args.ckpt, map_location=device)
        # train_restyler saves state["G"] = Generator.state_dict() (keys prefixed "net.")
        G.load_state_dict(state["G"])

    reader = None
    if args.verify:
        from src.parseq_reader import ParseqReader
        reader = ParseqReader(ckpt=args.reader_ckpt, train=False)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    kept = seen = 0
    with torch.inference_mode():
        for x_rgb, mask, labels in loader:
            if G is None:
                fake = x_rgb.to(device)                  # raw composite (±GAN ablation)
            else:
                x = torch.cat([x_rgb.to(device), mask.to(device)], dim=1)
                fake = G(x)                              # (B,3,workH,workW) in [-1,1]
            if reader is not None:
                preds = reader.read(fake, lengths=[len(l) for l in labels])
                keep_mask = [p == l for p, l in zip(preds, labels)]
            else:
                keep_mask = [True] * len(labels)
            arr = ((fake.detach().cpu().clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8)
            for i, (lbl, keep) in enumerate(zip(labels, keep_mask)):
                seen += 1
                if not keep:
                    continue
                im = Image.fromarray(np.transpose(arr[i], (1, 2, 0))).resize((Wc, Hc), Image.BILINEAR)
                fn = f"{kept:06d}_{lbl}.png"
                im.save(args.out / fn)
                rows.append((fn, lbl))
                kept += 1
            if seen % 5120 < args.batch:
                print(f"  {seen}/{args.n} seen, {kept} kept", flush=True)

    with (args.out / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filename", "label"]); w.writerows(rows)
    rate = 100.0 * kept / max(1, seen)
    print(f"wrote {kept} labeled samples to {args.out}  (keep rate {rate:.1f}%)")


if __name__ == "__main__":
    main()
