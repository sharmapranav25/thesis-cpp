"""Run the corrected DarkBay Phase 1 -> Phase 3 targeted pool -> PARSeq cycle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"E:\pranav\Final sem\Models\level 1\.venv\Scripts\python.exe")
PACKAGE = Path(r"E:\pranav\Final sem\sourcefree_synth\releases\CAPRI_phase123_v1")
REAL_DIR = Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled")
COPY_SUFFIX = re.compile(r"\s*\(\d+\)$")
CHARSET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(root: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes],
        key=lambda p: str(p).lower(),
    )


def pixel_hash(path: Path) -> tuple[str, int, int]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        digest = hashlib.sha256(str(image.size).encode("ascii"))
        digest.update(image.tobytes())
        width, height = image.size
    return digest.hexdigest(), width, height


def make_real_inputs(real_dir: Path, run: Path, fit_count: int) -> None:
    paths = image_paths(real_dir)
    if len(paths) <= fit_count:
        raise RuntimeError(f"Need more than {fit_count} images under {real_dir}")
    rows = []
    seen = set()
    for path in paths:
        label = COPY_SUFFIX.sub("", path.stem)
        if not (2 <= len(label) <= 32 and set(label) <= CHARSET):
            raise RuntimeError(f"Filename stem is not a valid label: {path}")
        pixels, width, height = pixel_hash(path)
        if pixels in seen:
            raise RuntimeError(f"Duplicate real pixels: {path}")
        seen.add(pixels)
        rows.append(
            {
                "path": str(path),
                "file_sha256": sha(path),
                "pixel_sha256": pixels,
                "width": width,
                "height": height,
                "label": label,
            }
        )
    fit = rows[:fit_count]
    validation = rows[fit_count:]
    split = {
        "labels_read": False,
        "fit": [{k: v for k, v in row.items() if k != "label"} for row in fit],
        "validation": [{k: v for k, v in row.items() if k != "label"} for row in validation],
        "source": str(real_dir),
        "created_for": "DarkBay Phase 1 style fit and Phase 3 fit-only inference; held-out validation",
    }
    write_json(run / "real_split.json", split)
    write_json(run / "references.json", {"partition": "style_fit", "images": [r["path"] for r in fit]})
    with (run / "validation_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        for index, row in enumerate(validation):
            writer.writerow([f"validation_{index:04d}", row["label"]])
    write_json(
        run / "real_input_audit.json",
        {
            "source": str(real_dir),
            "images": len(rows),
            "fit": len(fit),
            "validation": len(validation),
            "unique_pixels": len(seen),
            "labels_read_by_phase1_or_phase3": False,
            "labels_used_only_for_final_parseq_diagnostics": True,
            "file_order": "case-insensitive path sort",
        },
    )


def run_step(command: list[str], log_path: Path, status_path: Path, state: str) -> None:
    write_json(status_path, {"state": state, "command": command, "updated_unix": time.time()})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            write_json(
                status_path,
                {
                    "state": state,
                    "pid": os.getpid(),
                    "child_pid": process.pid,
                    "updated_unix": time.time(),
                    "log": str(log_path),
                },
            )
            time.sleep(30)
    if process.returncode:
        raise RuntimeError(f"{state} failed with exit code {process.returncode}; see {log_path}")


def require_complete(path: Path, key: str = "state") -> None:
    if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get(key) != "complete":
        raise RuntimeError(f"Expected complete artifact: {path}")


def main(args: argparse.Namespace) -> None:
    run = args.run.resolve()
    if run.exists():
        raise FileExistsError(f"Output already exists: {run}")
    for path in (PYTHON, PACKAGE / "capri.py", REAL_DIR):
        if not path.exists():
            raise FileNotFoundError(path)
    run.mkdir(parents=True)
    status = run / "status.json"
    write_json(status, {"state": "starting", "run": str(run), "updated_unix": time.time()})
    try:
        make_real_inputs(REAL_DIR, run, args.fit_count)
        capri = str(PACKAGE / "capri.py")

        run_step(
            [
                str(PYTHON), capri, "profile",
                "--references", str(run / "references.json"),
                "--name", "darkbay_phase123",
                "--kind", "conformer", "--variant", "3", "--device", "cuda",
                "--out", str(run / "phase1_profile"),
            ],
            run / "phase1.log", status, "phase1_profile",
        )
        if not (run / "phase1_profile" / "profile.json").is_file():
            raise RuntimeError("Phase 1 did not produce profile.json")

        run_step(
            [
                str(PYTHON), capri, "generate-broad",
                "--profile", str(run / "phase1_profile" / "profile.json"),
                "--count", str(args.broad_count), "--seed", str(args.broad_seed),
                "--partition", "train", "--out", str(run / "phase2_broad"),
            ],
            run / "phase2.log", status, "phase2_broad_generation",
        )
        require_complete(run / "phase2_broad" / "audit.json", key="state")

        run_step(
            [
                str(PYTHON), capri, "phase3-prepare",
                "--family", "darkbay", "--corpus", str(run / "phase2_broad"),
                "--real-split", str(run / "real_split.json"),
                "--run", str(run / "phase3_broad"),
            ],
            run / "phase3_prepare_broad.log", status, "phase3_prepare_broad",
        )
        require_complete(run / "phase3_broad" / "data" / "status.json")

        run_step(
            [
                str(PYTHON), capri, "phase3-train",
                "--family", "darkbay", "--run", str(run / "phase3_broad"),
            ],
            run / "phase3_readers.log", status, "phase3_reader_comparison",
        )
        scheme_path = run / "phase3_broad" / "inferred_scheme.json"
        scheme = json.loads(scheme_path.read_text(encoding="utf-8"))
        if not scheme.get("synthetic_reading_gate", {}).get("passed"):
            raise RuntimeError("Phase 3 synthetic reading gate failed; refusing targeted generation")
        write_json(
            run / "phase3_broad" / "scheme_review.json",
            {
                "review": "automated protocol review",
                "accepted": True,
                "reason": "fit-only source, validation excluded, synthetic reading gate passed",
                "scheme_sha256": sha(scheme_path),
                "validation_used_for_scheme": scheme.get("validation_used_for_scheme"),
            },
        )

        run_step(
            [
                str(PYTHON), capri, "generate-targeted",
                "--scheme", str(scheme_path),
                "--profile", str(run / "phase1_profile" / "profile.json"),
                "--count", str(args.targeted_count), "--seed", str(args.targeted_seed),
                "--out", str(run / "phase3_targeted"),
            ],
            run / "phase3_targeted.log", status, "phase3_targeted_generation",
        )
        require_complete(run / "phase3_targeted" / "status.json")

        run_step(
            [
                str(PYTHON), capri, "phase3-prepare",
                "--family", "darkbay", "--corpus", str(run / "phase3_targeted"),
                "--real-split", str(run / "real_split.json"),
                "--run", str(run / "phase3_targeted_prepared"),
            ],
            run / "phase3_prepare_targeted.log", status, "phase3_prepare_targeted",
        )
        require_complete(run / "phase3_targeted_prepared" / "data" / "status.json")

        final_runner = ROOT / "run_parseq_phase3.py"
        run_step(
            [
                str(PYTHON), str(final_runner),
                "--phase3-data", str(run / "phase3_targeted_prepared" / "data"),
                "--out", str(run / "parseq"),
                "--real-part", "validation",
                "--real-labels", str(run / "validation_labels.csv"),
                "--upstream", str(PACKAGE / "phase3" / "upstream"),
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--device", "auto",
            ],
            run / "parseq.log", status, "final_parseq_training",
        )
        require_complete(run / "parseq" / "status.json")
        write_json(status, {"state": "complete", "run": str(run), "updated_unix": time.time()})
    except BaseException as error:
        write_json(run / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        write_json(status, {"state": "failed", "error": repr(error), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "tonight_runs" / time.strftime("darkbay_phase123_%Y%m%d_%H%M%S"))
    parser.add_argument("--fit-count", type=int, default=48)
    parser.add_argument("--broad-count", type=int, default=100000)
    parser.add_argument("--targeted-count", type=int, default=100000)
    parser.add_argument("--broad-seed", type=int, default=20260929)
    parser.add_argument("--targeted-seed", type=int, default=20260930)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    main(parser.parse_args())
