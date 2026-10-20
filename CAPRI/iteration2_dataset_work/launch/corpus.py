"""Generate a frozen corpus, audit it, then launch the simple 15-epoch baseline.

No images are generated or changed during full training. Every stage is logged.
Generation retries may repair ticket combinations before the corpus is frozen.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from feature_bank.iter2 import balanced as B


def status(root, stage, **fields):
    record = dict(stage=stage, pid=os.getpid(), updated=time.time(), **fields)
    tmp = root / "pipeline_status.tmp"
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    tmp.replace(root / "pipeline_status.json")
    print(json.dumps(record), flush=True)


def repair_pair(left, right, round_no, variant):
    """Exchange quota assignments, never weaken guards or relabel pixels."""
    left, right = copy.deepcopy(left), copy.deepcopy(right)
    keys = ["effects"]
    if round_no >= 2 or left["weight"] < 3:
        keys.append("weight")
    if round_no >= 4:
        keys.append("width")
    for key in keys:
        left[key], right[key] = right[key], left[key]
    for ticket in [left, right]:
        ticket["revision"] = 10000 * round_no + variant
        ticket["max_attempts"] = 120
    return left, right


def repair_rows(pool, root, tickets, rows, name):
    # Commit a pair only when BOTH render successfully. A failed trial must not
    # replace an accepted donor. Weight/width are independent quota axes; the
    # category/x-height/italic/mixture coupling and two-row eligibility stay fixed.
    for round_no in range(1, 7):
        bad = [i for i, row in enumerate(rows) if row["failed"]]
        if not bad:
            break
        status(root, "repairing_generation", part=name, failed=len(bad), round=round_no)
        used = set()
        for i in bad:
            candidates = [j for j, row in enumerate(rows) if j != i and j not in used and not row["failed"]
                          and tickets[j]["k"] == tickets[i]["k"]]
            candidates.sort(key=lambda j: (tickets[j]["weight"] < 3,
                                           tickets[j]["appearance"] in B.GEOMETRIC,
                                           tickets[j]["category"] == "Monospace",
                                           abs(tickets[j]["weight"] - 5), j))
            if not candidates:
                raise RuntimeError("No compatible pairing-repair partner")
            # Different donors and seeds avoid repeating an infeasible pairing.
            start = ((round_no - 1) * 8) % len(candidates)
            candidates = (candidates[start:] + candidates[:start])[:8]
            pairs = [repair_pair(tickets[i], tickets[j], round_no, j) for j in candidates]
            committed = False
            for offset in range(0, len(pairs), 2):
                batch = pairs[offset:offset+2]
                rendered = list(pool.map(B.render_ticket, [t for pair in batch for t in pair]))
                for local_pos, pair in enumerate(batch):
                    pos = offset + local_pos
                    j = candidates[pos]
                    left_row, right_row = rendered[2*local_pos:2*local_pos+2]
                    if left_row["failed"] or right_row["failed"]:
                        continue
                    exchanged = [key for key in ["effects", "weight", "width"]
                                 if tickets[i][key] != pair[0][key]]
                    tickets[i], tickets[j] = pair
                    rows[i], rows[j] = left_row, right_row
                    used.update([i, j])
                    status(root, "repaired_ticket", part=name, index=tickets[i]["index"],
                           partner=tickets[j]["index"], round=round_no, exchanged=exchanged)
                    committed = True
                    break
                if committed:
                    break


def make_part(pool, root, tickets, name):
    start = time.monotonic()
    rows = []
    for row in pool.map(B.render_ticket, tickets, chunksize=4):
        rows.append(row)
        if len(rows) % 200 == 0:
            status(root, "generating", part=name, attempted=len(rows), requested=len(tickets),
                   accepted=sum(not r["failed"] for r in rows), seconds=round(time.monotonic() - start))
    repair_rows(pool, root, tickets, rows, name)
    if any(row["failed"] for row in rows):
        (root / (name + "_failed_tickets.json")).write_text(
            json.dumps([r for r in rows if r["failed"]], indent=2), encoding="utf-8")
        raise RuntimeError(f"{name}: unreadable tickets remain; training will not start")
    planned = Counter(B.REGISTRY[n]["head"] for ticket in tickets for n in ticket["effects"])
    report = B.write_cache(root, rows, dict(planned), tickets, time.monotonic() - start, cache_name=name)
    status(root, "generated_part", part=name, images=len(rows), seconds=round(time.monotonic() - start))
    return report


def audit(root, expected):
    caches = sorted((root / "iter2_h96").iterdir())
    seen, counts, effect_counts, text_counts, font_counts = {}, Counter(), Counter(), Counter(), [Counter() for _ in range(7)]
    font_table = B.bank().by_key
    for cache in caches:
        meta = json.loads((cache / "meta.json").read_text())
        split = "train" if cache.name.startswith("train_") else cache.name
        idx = np.load(cache / "index.npz")
        n = len(idx["widths"])
        assert n == meta["n"] and all(len(idx[key]) == n for key in idx.files)
        assert np.all((idx["length"] >= 2) & (idx["length"] <= 8))
        assert np.all((idx["eff_present"].sum(1) >= 3) & (idx["eff_present"].sum(1) <= 5))
        assert (cache / "images.u8").stat().st_size == int(idx["widths"].sum()) * 96 * 3
        counts[split] += n
        for h, value in zip(meta["effect_heads"], idx["eff_present"].sum(0)): effect_counts[h] += int(value) if split == "train" else 0
        if split == "train":
            for h, value in zip(meta["text_heads"], idx["txt_present"].sum(0)): text_counts[h] += int(value)
            for j in range(7): font_counts[j].update(idx["font"][:, j].tolist())
        with gzip.open(cache / "labels.jsonl.gz", "rt", encoding="utf-8") as f:
            labels = 0
            for line in f:
                lab = json.loads(line); digest = lab["sha256"]
                if digest in seen: raise RuntimeError(f"Duplicate pixels between {seen[digest]} and {cache.name}")
                seen[digest] = cache.name
                required = "heldout" if split == "generalisation" else "train"
                assert all(font_table[key]["split"] == required for key in lab["text_spec"]["fonts"])
                assert len(lab["text"]["text"]) == int(lab["text"]["length"])
                labels += 1
            assert labels == n
    assert dict(counts) == expected, (dict(counts), expected)
    assert max(effect_counts.values()) - min(effect_counts.values()) <= 1
    manifest = dict(images=dict(counts), unique_images=len(seen), training_effect_positives=dict(effect_counts),
                    training_text_positives=dict(text_counts), training_font_bins=[dict(c) for c in font_counts],
                    structural_audit="pass", full_distribution_balance="not claimed",
                    limits=["No observed font-weight level 0 in the pilot; measured weight bins may be uneven.",
                            "Condensed and expanded labels are linked to width fitting, so their prevalence differs.",
                            "Held-out-font x-height support is narrower; report its distribution separately.",
                            "Effect levels retain renderer measurement semantics; perceptual visibility under degradation is a research limitation."],
                    immutable_for_training=True)
    (root / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_command(root, args, name):
    status(root, name, command=args)
    with (root / (name + ".log")).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        status(root, name, child_pid=process.pid, log=str(root / (name + ".log")))
        result = process.wait()
    if result: raise RuntimeError(f"{name} failed with exit code {result}; see {name}.log")


def final_report(root, config, training):
    import torch
    from feature_bank.iter2 import data as D, metrics as M
    from feature_bank.iter2.model import Iter2Model
    candidates = []
    for path in sorted(training.glob("epoch_*/scores.json")):
        record = json.loads(path.read_text())
        sc = record["scores"]["selection"]
        effect = list(sc["effects"].values()); text = list(sc["text_heads"].values())
        groups = [np.mean([x["f1"] for x in effect]), np.mean([x["lvl_pm1"] for x in effect if "lvl_pm1" in x]),
                  np.mean([x["f1"] for x in text]), np.mean([x["lvl_pm1"] for x in text if "lvl_pm1" in x]),
                  sc["reading"]["exact"], sc["reading"]["length_head"],
                  np.mean([v for v in sc["font"].values() if v is not None])]
        candidates.append((float(np.mean(groups)), path.parent))
    score, chosen = max(candidates, key=lambda x: (x[0], -int(x[1].name.split("_")[1])))
    model = Iter2Model(**config["model"]).cuda()
    model.load_state_dict(torch.load(chosen / "ema.pt", map_location="cuda", weights_only=True))
    sets = D.open_split(root, "seen_test")
    meta = sets[0].meta
    report = dict(chosen=str(chosen), selection_score=score,
                  rule="Equal mean of seven fixed metric groups; earliest epoch breaks ties", tests={})
    for split in ["seen_test", "generalisation"]:
        P, L = M.predict(model, D.open_split(root, split), torch.device("cuda"))
        report["tests"][split] = M.scores(P, L, meta["effect_heads"], meta["text_heads"])
    (training / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=100000)
    parser.add_argument("--n-fixed", type=int, default=2000)
    parser.add_argument("--n-heldout", type=int, default=2500)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.out).resolve(); root.mkdir(parents=True, exist_ok=False)
    try:
        status(root, "generating", workers=args.workers)
        sizes = {s: args.n_fixed for s in ["steering", "selection", "calibration", "seen_test"]}
        sizes["generalisation"] = args.n_heldout; sizes["train"] = args.n_train
        with ProcessPoolExecutor(args.workers) as pool:
            for j, split in enumerate(["steering", "selection", "calibration", "seen_test", "generalisation", "train"]):
                font_split = "heldout" if split == "generalisation" else "train"
                tickets, _ = B.schedule(sizes[split], 2026100302 + j, font_split=font_split, split=split)
                for start in range(0, len(tickets), 10000):
                    name = f"train_{start:06d}" if split == "train" else split
                    status(root, "generating", split=split, part=name, requested=min(10000, len(tickets) - start))
                    make_part(pool, root, tickets[start:start+10000], name)
        status(root, "auditing")
        audit(root, sizes)
        if args.generate_only:
            status(root, "generation_complete"); return
        config = yaml.safe_load(Path("configs/iter2_baseline.yaml").read_text())
        config["data"] = str(root)
        config["model"]["ctc_frames"] = 1
        config["train"].update(epochs=15, loss_weighting="fixed")
        config["score"]["every_epoch"] = ["steering", "selection"]
        saved = root / "training_config.yaml"; saved.write_text(yaml.safe_dump(config), encoding="utf-8")
        common = [sys.executable, "-B", "-m", "feature_bank.iter2.train", "--config", str(saved)]
        smoke = root / "smoke_training"
        run_command(root, common + ["--out", str(smoke), "--epochs", "1", "--train-limit", "10000", "--score-limit", "500"], "smoke_epoch1")
        run_command(root, common + ["--out", str(smoke), "--epochs", "2", "--train-limit", "10000", "--score-limit", "500", "--resume"], "smoke_resume")
        assert json.loads((smoke / "status.json").read_text())["epoch"] == 2
        training = root / "baseline_training"
        run_command(root, common + ["--out", str(training)], "training_15_epochs")
        status(root, "final_evaluation")
        final_report(root, config, training)
        status(root, "complete", training=str(training))
    except Exception as exc:
        status(root, "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
