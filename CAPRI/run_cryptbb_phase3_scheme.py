"""Run the City-style CryptBB Phase 3 reader-to-scheme-to-targeted workflow."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
GROUPS = ("digits", "lowercase", "uppercase")
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda p: str(p).lower(),
    )


def pixel_hash(path: Path) -> tuple[str, int, int]:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    digest = hashlib.sha256(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest(), image.width, image.height


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(30)
        if process.returncode:
            raise RuntimeError(f"Command failed with exit code {process.returncode}; see {log_path}")


def make_real_split(real_dir: Path, path: Path) -> dict:
    paths = image_paths(real_dir)
    if len(paths) < 2:
        raise RuntimeError(f"Need at least two real images under {real_dir}")
    seen = set()
    rows = []
    for image_path in paths:
        pixels, width, height = pixel_hash(image_path)
        if pixels in seen:
            raise RuntimeError(f"Duplicate real pixels: {image_path}")
        seen.add(pixels)
        file_digest = sha(image_path)
        rows.append(
            {
                "path": str(image_path),
                "file_sha256": file_digest,
                "pixel_sha256": pixels,
                "width": width,
                "height": height,
            }
        )
    fit_count = math.floor(len(rows) * 0.8)
    fit_count = max(1, min(len(rows) - 1, fit_count))
    split = {
        "labels_read": False,
        "fit": rows[:fit_count],
        "validation": rows[fit_count:],
        "source": str(real_dir),
        "created_for": "City-style CryptBB Phase 3: fit-only scheme inference; validation held out",
        "split_rule": "lexicographically sorted image paths; first floor(80%) fit, remainder validation",
    }
    write_json(path, split)
    return split


def groups(text: str) -> list[int]:
    return [
        int(any(c in "0123456789" for c in text)),
        int(any(c in "abcdefghijklmnopqrstuvwxyz" for c in text)),
        int(any(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in text)),
    ]


def model_summary(rows: list[dict], calibration: dict) -> dict:
    return {
        "samples": len(rows),
        "length_histogram": dict(collections.Counter(row["decoded_length"] for row in rows)),
        "group_combinations": dict(
            collections.Counter("".join(map(str, row["decoded_groups"])) for row in rows)
        ),
        "group_prevalence": [
            float(sum(row["decoded_groups"][index] for row in rows) / len(rows)) for index in range(3)
        ],
        "mean_decode_confidence": float(sum(row["confidence"] for row in rows) / len(rows)),
        "calibration": calibration,
    }


def aggregate_scheme(root: Path, data: Path, family: str, fit_count: int) -> dict:
    runs = [root / "crnn_plain", root / "parseq_plain"]
    predictions = []
    summaries = {}
    hashes = {}
    eligible = []
    for run in runs:
        status = read_json(run / "status.json")
        if status.get("state") != "complete":
            raise RuntimeError(f"Reader run is not complete: {run}")
        rows = [json.loads(line) for line in (run / "fit_predictions.jsonl").read_text().splitlines()]
        if len(rows) != fit_count:
            raise RuntimeError(f"Expected {fit_count} fit predictions in {run}, found {len(rows)}")
        if predictions and [(r["id"], r["pixel_sha256"]) for r in rows] != [
            (r["id"], r["pixel_sha256"]) for r in predictions[0]
        ]:
            raise RuntimeError("Reader fit locators do not match")
        calibration = read_json(run / "calibration_metrics.json")
        eligible.append(calibration["cer"] <= 0.35 and calibration["exact_string_accuracy"] >= 0.20)
        predictions.append(rows)
        hashes[run.name] = sha(run / "best.pt")
        summaries[run.name] = model_summary(rows, calibration)

    cells = [
        (length, "".join(map(str, bits)))
        for length in range(2, 9)
        for bits in itertools.product([0, 1], repeat=3)
        if 0 < sum(bits) <= length
    ]
    lookup = {cell: index for index, cell in enumerate(cells)}
    vectors = []
    invalid = 0
    for model_rows in predictions:
        matrix = []
        for row in model_rows:
            key = (row["decoded_length"], "".join(map(str, row["decoded_groups"])))
            vector = [0.0] * len(cells)
            if key in lookup:
                vector[lookup[key]] = 1.0
            else:
                invalid += 1
                vector = [1.0 / len(cells)] * len(cells)
            matrix.append(vector)
        vectors.append(matrix)

    length_disagreement = sum(
        len({predictions[model][index]["decoded_length"] for model in range(len(predictions))}) > 1
        for index in range(fit_count)
    ) / fit_count
    group_disagreement = sum(
        len({tuple(predictions[model][index]["decoded_groups"]) for model in range(len(predictions))}) > 1
        for index in range(fit_count)
    ) / fit_count
    text_disagreement = sum(
        len({predictions[model][index]["prediction"] for model in range(len(predictions))}) > 1
        for index in range(fit_count)
    ) / fit_count

    observed = [
        sum(vectors[model][sample][cell] for model in range(len(vectors)) for sample in range(fit_count))
        / (len(vectors) * fit_count)
        for cell in range(len(cells))
    ]
    prior = [1.0 / len(cells)] * len(cells)
    uncertainty = max(length_disagreement, group_disagreement, invalid / (fit_count * len(predictions)))
    prior_mass = min(0.8, 0.2 + 0.6 * uncertainty)
    sampling = [(1.0 - prior_mass) * value + prior_mass * base for value, base in zip(observed, prior)]

    scheme_rows = [
        {
            "length": length,
            "groups": group_key,
            "observed_weight": float(observed[index]),
            "sampling_weight": float(sampling[index]),
        }
        for index, (length, group_key) in enumerate(cells)
    ]
    audit = read_json(data / "audit.json")
    gate = all(eligible)
    scheme = {
        "schema": 1,
        "stage": "Official CAPRI Phase 3",
        "family": family,
        "source_partition": "fit",
        "images": fit_count,
        "model_selection": "CRNN and PARSeq plain readers; synthetic selection CER then exact accuracy",
        "models": hashes,
        "model_summaries": summaries,
        "disagreement": {
            "length": float(length_disagreement),
            "groups": float(group_disagreement),
            "text": float(text_disagreement),
        },
        "invalid_or_out_of_support_predictions": invalid,
        "joint_distribution": scheme_rows,
        "uncertainty_prior_mass": float(prior_mass),
        "uncertainty_rule": "0.2 + 0.6 * max(length disagreement, group disagreement, invalid fraction), capped .8",
        "group_order": list(GROUPS),
        "alphabet": CHARSET,
        "alphabet_policy": "All 62 characters retained; zero predictions do not establish exclusion",
        "length_policy": "Weighted 2..8 range inferred from real fit predictions",
        "validation_used_for_scheme": False,
        "synthetic_reading_gate": {
            "passed": gate,
            "criterion": "Each reader calibration CER <= .35 and exact >= .20",
            "selected_readers": [run.name for run in runs],
        },
        "real_accuracy": None,
        "real_ground_truth_status": "No real transcription labels used; real filenames are locators only",
        "shortcut_limit": read_json(data / "shortcut_audit.json").get("claim"),
        "phase3_data_audit_sha256": sha(data / "audit.json"),
        "status": "estimated_with_uncertainty" if gate else "blocked_inadequate_synthetic_reading",
    }
    write_json(root / "inferred_scheme.json", scheme)
    write_json(root / "scheme_review.json", {"status": "ready_for_targeted_generation" if gate else "blocked", "fit_only": True})
    with (root / "disagreement_examples.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(fit_count):
            if len({predictions[model][index]["prediction"] for model in range(len(predictions))}) > 1:
                handle.write(
                    json.dumps(
                        {
                            "id": predictions[0][index]["id"],
                            "pixel_sha256": predictions[0][index]["pixel_sha256"],
                            "models": {
                                run.name: predictions[model][index]["prediction"]
                                for model, run in enumerate(runs)
                            },
                        }
                    )
                    + "\n"
                )
    return scheme


def main(args: argparse.Namespace) -> None:
    run_root = args.run.resolve()
    if run_root.exists():
        raise FileExistsError(f"Run output already exists: {run_root}")
    split_path = run_root.with_name(run_root.name + "_real_split.json")
    split = make_real_split(args.real_dir.resolve(), split_path)
    fit_count = len(split["fit"])

    # The frozen CAPRI CLI creates the --run directory itself. Keeping the
    # split and prepare log beside it avoids pre-creating that directory.
    phase3_run = run_root
    prepare_cmd = [
        str(args.python),
        str(args.package / "capri.py"),
        "phase3-prepare",
        "--family",
        args.family,
        "--corpus",
        str(args.corpus.resolve()),
        "--real-split",
        str(split_path),
        "--run",
        str(phase3_run),
    ]
    run_command(prepare_cmd, run_root.with_name(run_root.name + "_phase3_prepare.log"))

    reader_train = args.package / "phase3" / "reader" / "train.py"
    for model in ("crnn", "parseq"):
        output = run_root / f"{model}_plain"
        command = [
            str(args.python),
            str(reader_train),
            "--model",
            model,
            "--data",
            str(phase3_run / "data"),
            "--out",
            str(output),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
        ]
        run_command(command, run_root / f"{model}_plain.log")

    scheme = aggregate_scheme(run_root, phase3_run / "data", args.family, fit_count)
    if not scheme["synthetic_reading_gate"]["passed"]:
        write_json(run_root / "status.json", {"state": "blocked_inadequate_synthetic_reading", "scheme": str(run_root / "inferred_scheme.json")})
        raise RuntimeError("Synthetic reader gate failed; targeted generation was not started")

    target_cmd = [
        str(args.python),
        str(args.package / "capri.py"),
        "generate-targeted",
        "--scheme",
        str(run_root / "inferred_scheme.json"),
        "--count",
        str(args.count),
        "--seed",
        str(args.target_seed),
        "--out",
        str(args.target.resolve()),
    ]
    run_command(target_cmd, run_root / "targeted_generation.log")
    write_json(
        run_root / "status.json",
        {
            "state": "complete",
            "scheme": str(run_root / "inferred_scheme.json"),
            "target": str(args.target.resolve()),
            "target_count": args.count,
            "fit_images": fit_count,
            "validation_images": len(split["validation"]),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--family", default="cryptbb")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--target-seed", type=int, default=732)
    parser.add_argument("--count", type=int, default=100000)
    main(parser.parse_args())
