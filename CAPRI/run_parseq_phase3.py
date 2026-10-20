"""Train PARSeq from a prepared Phase 3 cache and evaluate the real split per epoch.

The Phase 0/1/2/3 generators are intentionally not called here.  This runner
consumes their audited Phase 3 output and only wires it to the existing reader
implementation.  Synthetic selection remains the only checkpoint-selection
signal; real-set results are diagnostic logs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
READER_DIR = ROOT / "capri_phase123_build" / "phase3" / "reader"
COPY_SUFFIX = re.compile(r"\s*\(\d+\)$")


def load_reader_modules(upstream: Path | None):
    """Load the package reader modules without changing the built phases."""
    sys.path.insert(0, str(READER_DIR))
    import common  # type: ignore

    if upstream is None:
        for candidate in (
            READER_DIR.parent / "upstream",
            Path(r"E:\pranav\Final sem\Models\level 1\upstream"),
        ):
            if candidate.is_dir():
                upstream = candidate
                break
    if upstream is not None:
        common.UPSTREAM = upstream.resolve()

    # models.py imports UPSTREAM from common at import time.  Import it only
    # after applying the optional override above.
    import models  # type: ignore
    import train as reader_train  # type: ignore

    return common, models, reader_train


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def load_real_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            if not rows.fieldnames or not {"id", "label"}.issubset(rows.fieldnames):
                raise ValueError("--real-labels CSV must contain id,label columns")
            return {row["id"]: row["label"] for row in rows}
    value = read_json(path)
    if isinstance(value, dict):
        return {str(key): str(label) for key, label in value.items()}
    if isinstance(value, list):
        return {str(row["id"]): str(row["label"]) for row in value}
    raise ValueError("--real-labels must be a CSV, id-to-label JSON object, or JSON rows")


def prepare_inputs(data: Path, real_part: str):
    required = [
        data / "audit.json",
        data / "status.json",
        data / "partitions.jsonl",
        data / "synthetic.npy",
        data / f"{real_part}.npy",
        data / "real_locators.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Prepared Phase 3 data is missing: " + ", ".join(missing))
    audit = read_json(data / "audit.json")
    status = read_json(data / "status.json")
    if audit.get("passed") is not True or status.get("state") != "complete":
        raise RuntimeError("Refusing to train from incomplete or unaudited Phase 3 data")
    if (data / "failure.json").exists():
        raise RuntimeError("Phase 3 data contains failure.json")

    rows = [json.loads(line) for line in (data / "partitions.jsonl").open(encoding="utf-8")]
    indices = {
        name: np.asarray([row["index"] for row in rows if row["partition"] == name], dtype=np.int64)
        for name in ("train", "selection", "calibration")
    }
    for name, values in indices.items():
        if not len(values):
            raise RuntimeError(f"Phase 3 partition is empty: {name}")
    real = read_json(data / "real_locators.json")
    if real_part not in real or len(real[real_part]) != len(np.load(data / f"{real_part}.npy", mmap_mode="r")):
        raise RuntimeError(f"Real locator/cache mismatch for split {real_part}")
    return audit, rows, indices, real


def load_external_real(common, path: Path | None):
    if path is None:
        return None, None, {}
    image_paths = sorted(
        [
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        ],
        key=lambda item: str(item).lower(),
    )
    if not image_paths:
        raise FileNotFoundError(f"No real images found under {path}")
    cache = np.empty((len(image_paths), 32, 192, 3), dtype=np.uint8)
    locators = []
    labels = {}
    seen = set()
    for index, image_path in enumerate(image_paths):
        label = COPY_SUFFIX.sub("", image_path.stem)
        if not (set(label) <= set(common.CHARSET) and 1 <= len(label) <= 32):
            raise ValueError(f"Cannot derive a label from real filename: {image_path}")
        image = common.load_image(image_path)
        pixel_hash = common.pixels(image)
        if pixel_hash in seen:
            raise RuntimeError(f"Duplicate real image pixels: {image_path}")
        seen.add(pixel_hash)
        cache[index] = np.asarray(image.resize(common.SIZE, Image.Resampling.BICUBIC))
        file_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        sample_id = image_path.name
        locators.append(
            {
                "id": sample_id,
                "path": str(image_path),
                "pixel_sha256": pixel_hash,
                "file_sha256": file_hash,
            }
        )
        labels[sample_id] = label
    return cache, locators, labels


def real_summary(result, locators, labels, score_fn, groups_fn):
    predictions = result["predictions"]
    summary = {
        "samples": len(predictions),
        "predicted_length_histogram": dict(collections.Counter(map(len, predictions))),
        "predicted_group_combinations": dict(
            collections.Counter("".join(map(str, groups_fn(text))) for text in predictions)
        ),
        "mean_confidence": float(np.mean(result["confidence"])) if predictions else 0.0,
        "real_accuracy": None,
        "ground_truth_status": "No real labels supplied; inference only",
    }
    if labels:
        ordered = [labels.get(row["id"]) for row in locators]
        if all(value is not None for value in ordered):
            summary["real_accuracy"] = score_fn(result, ordered, {"count_temperature": 1.0, "group_temperatures": [1.0] * 3})
            summary["ground_truth_status"] = "Labels supplied for diagnostics; never used for checkpoint selection"
    return summary


def write_real_epoch(output: Path, epoch: int, result, locators, summary, groups_fn) -> None:
    epoch_dir = output / "real_eval" / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    with (epoch_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for locator, prediction, confidence in zip(locators, result["predictions"], result["confidence"]):
            handle.write(
                json.dumps(
                    {
                        "id": locator["id"],
                        "pixel_sha256": locator["pixel_sha256"],
                        "prediction": prediction,
                        "decoded_length": len(prediction),
                        "decoded_groups": groups_fn(prediction),
                        "confidence": confidence,
                    }
                )
                + "\n"
            )
    write_json(epoch_dir / "summary.json", summary)


def train(args) -> None:
    common, models, reader_train = load_reader_modules(args.upstream)
    data = args.phase3_data.resolve()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new run directory: {output}")
    output.mkdir(parents=True)

    audit, rows, partitions, real_locators = prepare_inputs(data, args.real_part)
    external_cache, external_rows, external_labels = load_external_real(common, args.real_dir)
    if external_cache is None:
        real_cache = np.load(data / f"{args.real_part}.npy", mmap_mode="r")
        real_rows = real_locators[args.real_part]
        labels = load_real_labels(args.real_labels)
        if labels and not all(row["id"] in labels for row in real_rows):
            raise ValueError("--real-labels does not cover every image in the selected real split")
        real_source = f"prepared:{args.real_part}"
    else:
        real_cache = external_cache
        real_rows = external_rows
        labels = external_labels
        real_source = str(args.real_dir.resolve())

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cache = np.load(data / "synthetic.npy", mmap_mode="r")
    model = models.Reader("parseq", args.aux, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    steps_per_epoch = (len(partitions["train"]) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    step = 0
    best_key = (float("inf"), float("inf"))
    history = []
    started = time.time()
    protocol = {
        "model": "parseq",
        "auxiliary": args.aux,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "phase3_data": str(data),
        "phase3_partitions_sha256": common.sha(data / "partitions.jsonl"),
        "phase3_audit_sha256": common.sha(data / "audit.json"),
        "real_split": args.real_part,
        "real_source": real_source,
        "real_labels_used_for_selection": False,
        "checkpoint_selection": "synthetic selection CER, then synthetic exact accuracy",
        "real_evaluation": "after every epoch; diagnostic only",
        "device": device,
    }
    write_json(output / "protocol.json", protocol)
    (output / "checkpoints").mkdir()
    write_json(output / "status.json", {"state": "training", "epoch": 0, "epochs": args.epochs, "step": 0})

    try:
        for epoch in range(1, args.epochs + 1):
            order = np.random.default_rng(seed + epoch).permutation(partitions["train"])
            model.train()
            losses = []
            for start in range(0, len(order), args.batch_size):
                indices = order[start : start + args.batch_size]
                texts = [rows[int(index)]["label"] for index in indices]
                images = reader_train.batch(cache, indices, device)
                progress = step / max(1, total_steps - 1)
                factor = progress / args.warmup if progress < args.warmup else 0.5 * (
                    1.0 + np.cos(np.pi * (progress - args.warmup) / (1.0 - args.warmup))
                )
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * max(0.01, float(factor))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss, recognition = model.loss(images, texts)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                losses.append(float(recognition))

            synthetic = reader_train.infer(model, cache, partitions["selection"], device)
            synthetic_labels = [rows[int(index)]["label"] for index in partitions["selection"]]
            synthetic_metrics = reader_train.score(
                synthetic,
                synthetic_labels,
                {"count_temperature": 1.0, "group_temperatures": [1.0] * 3},
            )
            real = reader_train.infer(model, real_cache, np.arange(len(real_cache)), device)
            summary = real_summary(real, real_rows, labels, reader_train.score, common.groups)
            write_real_epoch(output, epoch, real, real_rows, summary, common.groups)
            append_jsonl(output / "real_eval.jsonl", {"epoch": epoch, **summary})

            entry = {
                "epoch": epoch,
                "step": step,
                "recognition_loss": float(np.mean(losses)),
                "synthetic_selection": synthetic_metrics,
                "real_evaluation": summary,
                "elapsed_seconds": time.time() - started,
            }
            history.append(entry)
            write_json(output / "history.json", history)
            checkpoint = {
                "state_dict": model.state_dict(),
                "protocol": protocol,
                "epoch": epoch,
                "steps": step,
                "synthetic_selection": synthetic_metrics,
            }
            torch.save(checkpoint, output / "checkpoints" / f"epoch_{epoch:03d}.pt")
            key = (synthetic_metrics["cer"], -synthetic_metrics["exact_string_accuracy"])
            if key < best_key:
                best_key = key
                torch.save(checkpoint, output / "best.pt")
            write_json(
                output / "status.json",
                {"state": "training", "epoch": epoch, "epochs": args.epochs, "step": step, "latest": entry},
            )
            print(json.dumps(entry), flush=True)

        write_json(
            output / "status.json",
            {
                "state": "complete",
                "epoch": args.epochs,
                "epochs": args.epochs,
                "step": step,
                "best_checkpoint": str((output / "best.pt").resolve()),
                "elapsed_seconds": time.time() - started,
            },
        )
    except BaseException as error:
        write_json(output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        write_json(output / "status.json", {"state": "failed", "error": repr(error), "step": step})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-data", type=Path, default=ROOT / "phase3_city_data_01")
    parser.add_argument("--out", type=Path, default=ROOT / "parseq_phase3_run")
    parser.add_argument("--real-part", choices=["fit", "validation"], default="validation")
    parser.add_argument("--real-dir", type=Path, help="labeled real image folder; labels are filename stems")
    parser.add_argument("--real-labels", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--aux", action="store_true", help="also train the existing auxiliary count/group heads")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.warmup <= 1:
        parser.error("epochs and batch-size must be positive; warmup must be in (0, 1]")
    train(args)


if __name__ == "__main__":
    main()
