"""Inference-only evaluation of a frozen Phase 3 reader checkpoint."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    reader_dir = args.package / "phase3" / "reader"
    sys.path.insert(0, str(reader_dir))
    import common  # type: ignore
    import models  # type: ignore
    import train as reader_train  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = models.Reader(args.model, False, args.seed).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.requires_grad_(False)

    cache = np.load(args.data / f"{args.split}.npy", mmap_mode="r")
    locators = read_json(args.data / "real_locators.json")[args.split]
    result = reader_train.infer(model, cache, np.arange(len(cache)), device)
    rows = []
    for locator, prediction, confidence in zip(locators, result["predictions"], result["confidence"]):
        rows.append(
            {
                "id": locator["id"],
                "pixel_sha256": locator["pixel_sha256"],
                "prediction": prediction,
                "decoded_length": len(prediction),
                "decoded_groups": common.groups(prediction),
                "confidence": float(confidence),
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "samples": len(rows),
        "predicted_length_histogram": dict(collections.Counter(row["decoded_length"] for row in rows)),
        "predicted_group_combinations": dict(
            collections.Counter("".join(map(str, row["decoded_groups"])) for row in rows)
        ),
        "predicted_group_prevalence": {
            name: float(sum(row["decoded_groups"][index] for row in rows) / len(rows))
            for index, name in enumerate(("digits", "lowercase", "uppercase"))
        },
        "mean_confidence": float(np.mean([row["confidence"] for row in rows])) if rows else 0.0,
        "ground_truth_status": "No real transcription labels used; filenames are locators only",
    }
    write_json(args.out / "summary.json", summary)
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=["crnn", "parseq"], default="crnn")
    parser.add_argument("--split", choices=["fit", "validation"], default="validation")
    parser.add_argument("--seed", type=int, default=731)
    main(parser.parse_args())
