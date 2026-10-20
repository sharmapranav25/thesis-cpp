"""Evaluate frozen City Phase 3 readers on the labelled CryptBB synthetic corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


READER_CODE = Path(r"E:\pranav\Final sem\sourcefree_synth\style_pipeline\phase3_reader_v2")
sys.path.insert(0, str(READER_CODE))
from common import group_key, metrics, sha, text_partition  # noqa: E402
from models import Reader  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class LabelledImages(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]]) -> None:
        self.root = root
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.root / row["filename"]
        with Image.open(path) as source:
            rgba = ImageOps.exif_transpose(source).convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
            if image.size != (192, 32):
                image = image.resize((192, 32), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        return tensor, row["label"], row["filename"]


def subset_metrics(labels: list[str], predictions: list[str], indices: list[int]) -> dict:
    return metrics([labels[i] for i in indices], [predictions[i] for i in indices])


@torch.inference_mode()
def evaluate_model(
    model_name: str,
    checkpoint: Path,
    loader: DataLoader,
    rows: list[dict[str, str]],
    out_dir: Path,
    device: str,
) -> dict:
    started = time.time()
    checkpoint_hash = sha(checkpoint)
    model = Reader(model_name, False, 731).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    model.eval().requires_grad_(False)

    predictions: list[str] = []
    confidences: list[float] = []
    filenames: list[str] = []
    status_path = out_dir / f"{model_name}_status.json"
    total = len(rows)
    for batch_index, (images, _, batch_filenames) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            batch_predictions, batch_confidences, _ = model.predict(images)
        predictions.extend(batch_predictions)
        confidences.extend(batch_confidences)
        filenames.extend(batch_filenames)
        if batch_index % 20 == 0 or len(predictions) == total:
            atomic_json(
                status_path,
                {
                    "state": "evaluating",
                    "model": model_name,
                    "processed": len(predictions),
                    "total": total,
                    "elapsed_seconds": time.time() - started,
                },
            )

    labels = [row["label"] for row in rows]
    result = metrics(labels, predictions)
    by_group: dict[str, dict] = {}
    for key in sorted(set(map(group_key, labels))):
        ids = [i for i, label in enumerate(labels) if group_key(label) == key]
        by_group[key] = subset_metrics(labels, predictions, ids)
    by_identity_partition: dict[str, dict] = {}
    for partition in ("train", "selection", "calibration", "test"):
        ids = [i for i, label in enumerate(labels) if text_partition(label) == partition]
        by_identity_partition[partition] = subset_metrics(labels, predictions, ids)

    result.update(
        model=model_name,
        auxiliary=False,
        checkpoint=str(checkpoint),
        checkpoint_epoch=saved.get("epoch"),
        checkpoint_sha256=checkpoint_hash,
        checkpoint_hash_unchanged=sha(checkpoint) == checkpoint_hash,
        corpus_images=total,
        elapsed_seconds=time.time() - started,
        by_true_group_combination=by_group,
        by_city_text_identity_partition=by_identity_partition,
        predicted_length_histogram=dict(sorted(Counter(map(len, predictions)).items())),
        note=(
            "Cross-family synthetic evaluation. Labels are CryptBB synthetic ground truth. "
            "The City text hash partition is reported because short text identities may recur across corpora."
        ),
    )
    atomic_json(out_dir / f"{model_name}_results.json", result)
    with (out_dir / f"{model_name}_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for filename, label, prediction, confidence in zip(filenames, labels, predictions, confidences):
            handle.write(
                json.dumps(
                    {
                        "filename": filename,
                        "label": label,
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                )
                + "\n"
            )
    atomic_json(
        status_path,
        {
            "state": "complete",
            "model": model_name,
            "processed": total,
            "total": total,
            "cer": result["cer"],
            "exact_string_accuracy": result["exact_string_accuracy"],
            "decoded_count_accuracy": result["decoded_count_accuracy"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    args.out.mkdir(parents=True)
    audit = json.loads((args.data / "audit.json").read_text(encoding="utf-8-sig"))
    if not audit.get("passed") or audit.get("images") != 100000:
        raise RuntimeError("CryptBB input audit has not passed for all 100,000 images")
    with (args.data / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100000:
        raise RuntimeError(f"Expected 100,000 labels, found {len(rows)}")
    if any(not (args.data / row["filename"]).is_file() for row in rows):
        raise RuntimeError("At least one labelled image is missing")

    protocol = {
        "task": "Frozen City Phase 3 reader cross-family synthetic evaluation",
        "corpus": str(args.data),
        "corpus_audit_sha256": sha(args.data / "audit.json"),
        "labels_sha256": sha(args.data / "labels.csv"),
        "images": len(rows),
        "models": ["crnn_plain", "parseq_plain"],
        "image_size": [32, 192],
        "filename_used_as_model_input": False,
        "labels_used_for_scoring_only": True,
    }
    atomic_json(args.out / "protocol.json", protocol)

    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(
        LabelledImages(args.data, rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        persistent_workers=args.workers > 0,
    )
    combined = {}
    for model_name in ("crnn", "parseq"):
        checkpoint = args.run / f"{model_name}_plain" / "best.pt"
        combined[model_name] = evaluate_model(model_name, checkpoint, loader, rows, args.out, device)
    atomic_json(args.out / "comparison.json", combined)
    atomic_json(args.out / "status.json", {"state": "complete", "models": ["crnn", "parseq"]})


if __name__ == "__main__":
    main()
