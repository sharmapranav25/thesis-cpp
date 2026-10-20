"""Run the requested CryptBB and DarkBay CAPRI/PARSeq jobs in sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path(r"E:\pranav\Final sem\Models\level 1\.venv\Scripts\python.exe")
DEFAULT_PACKAGE = Path(r"E:\pranav\Final sem\sourcefree_synth\releases\CAPRI_phase123_v1")
CRYPTBB_CORPUS = Path(r"E:\pranav\Final sem\synth_data\cryptbb_capri\cryptbb_capri_v3_variable_2to8_100k")
CRYPTBB_REAL = Path(r"E:\pranav\JoeyRetitling-20260502T012958Z-3-001\JoeyRetitling")
DARKBAY_REAL = Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(root: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    return sorted(
        [item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in suffixes],
        key=lambda item: str(item).lower(),
    )


def pixel_hash(path: Path) -> tuple[str, int, int]:
    image = Image.open(path).convert("RGB")
    digest = hashlib.sha256(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    width, height = image.size
    return digest.hexdigest(), width, height


def make_split(real_dir: Path, out: Path, fit_count: int = 48) -> None:
    paths = image_paths(real_dir)
    if len(paths) <= fit_count:
        raise RuntimeError(f"Need more than {fit_count} real images under {real_dir}")
    rows = []
    seen = set()
    for path in paths:
        pixels, width, height = pixel_hash(path)
        if pixels in seen:
            raise RuntimeError(f"Duplicate real pixels in {real_dir}: {path}")
        seen.add(pixels)
        rows.append(
            {
                "path": str(path),
                "file_sha256": sha(path),
                "pixel_sha256": pixels,
                "width": width,
                "height": height,
            }
        )
    write_json(
        out,
        {
            "labels_read": False,
            "fit": rows[:fit_count],
            "validation": rows[fit_count:],
            "source": str(real_dir),
            "created_for": "CAPRI Phase 3 cache boundary only; labels are not included",
        },
    )


def make_references(real_dir: Path, out: Path, count: int = 96) -> None:
    paths = image_paths(real_dir)
    if len(paths) < count:
        raise RuntimeError(f"Need at least {count} references under {real_dir}")
    write_json(out, {"partition": "style_fit", "images": [str(path) for path in paths[:count]]})


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


def ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def status_complete(path: Path) -> bool:
    return path.is_file() and read_json(path).get("state") == "complete"


def next_output(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 100):
        candidate = path.with_name(f"{path.name}_retry{index:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a retry output next to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--root", type=Path, default=ROOT / "tonight_runs" / time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true", help="reuse completed steps under --root")
    args = parser.parse_args()

    if args.resume:
        args.root.mkdir(parents=True, exist_ok=True)
    else:
        ensure_new(args.root)
        args.root.mkdir(parents=True)
    status = args.root / "status.json"
    write_json(status, {"state": "starting", "root": str(args.root), "updated_unix": time.time()})
    try:
        for required in (args.python, args.package / "capri.py", CRYPTBB_CORPUS, CRYPTBB_REAL, DARKBAY_REAL):
            if not required.exists():
                raise FileNotFoundError(required)

        cryptbb = args.root / "cryptbb"
        cryptbb.mkdir(exist_ok=args.resume)
        cryptbb_split = cryptbb / "real_split.json"
        make_split(CRYPTBB_REAL, cryptbb_split)
        if not status_complete(cryptbb / "phase3" / "data" / "status.json"):
            run_step(
                [
                    str(args.python),
                    str(args.package / "capri.py"),
                    "phase3-prepare",
                    "--family",
                    "cryptbb",
                    "--corpus",
                    str(CRYPTBB_CORPUS),
                    "--real-split",
                    str(cryptbb_split),
                    "--run",
                    str(cryptbb / "phase3"),
                ],
                cryptbb / "phase3_prepare.log",
                status,
                "cryptbb_phase3_prepare",
            )
        cryptbb_parseq = next_output(cryptbb / "parseq")
        run_step(
            [
                str(args.python),
                str(ROOT / "run_parseq_phase3.py"),
                "--phase3-data",
                str(cryptbb / "phase3" / "data"),
                "--out",
                str(cryptbb_parseq),
                "--real-dir",
                str(CRYPTBB_REAL),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
            ],
            cryptbb_parseq.with_suffix(".log"),
            status,
            "cryptbb_parseq_train",
        )

        darkbay = args.root / "darkbay"
        darkbay.mkdir(exist_ok=args.resume)
        references = darkbay / "references.json"
        split = darkbay / "real_split.json"
        make_references(DARKBAY_REAL, references)
        make_split(DARKBAY_REAL, split)
        if not (darkbay / "phase1_profile" / "profile.json").is_file():
            run_step(
                [
                    str(args.python),
                    str(args.package / "capri.py"),
                    "profile",
                    "--references",
                    str(references),
                    "--name",
                    "darkbay_tonight",
                    "--kind",
                    "conformer",
                    "--variant",
                    "3",
                    "--device",
                    "cuda",
                    "--out",
                    str(darkbay / "phase1_profile"),
                ],
                darkbay / "phase1.log",
                status,
                "darkbay_phase1_profile",
            )
        if not status_complete(darkbay / "phase2_broad_100k" / "audit.json"):
            run_step(
                [
                    str(args.python),
                    str(args.package / "capri.py"),
                    "generate-broad",
                    "--profile",
                    str(darkbay / "phase1_profile" / "profile.json"),
                    "--count",
                    "100000",
                    "--seed",
                    "20260929",
                    "--partition",
                    "train",
                    "--out",
                    str(darkbay / "phase2_broad_100k"),
                ],
                darkbay / "phase2.log",
                status,
                "darkbay_phase2_generate",
            )
        if not status_complete(darkbay / "phase3" / "data" / "status.json"):
            run_step(
                [
                    str(args.python),
                    str(args.package / "capri.py"),
                    "phase3-prepare",
                    "--family",
                    "darkbay",
                    "--corpus",
                    str(darkbay / "phase2_broad_100k"),
                    "--real-split",
                    str(split),
                    "--run",
                    str(darkbay / "phase3"),
                ],
                darkbay / "phase3_prepare.log",
                status,
                "darkbay_phase3_prepare",
            )
        darkbay_parseq = next_output(darkbay / "parseq")
        run_step(
            [
                str(args.python),
                str(ROOT / "run_parseq_phase3.py"),
                "--phase3-data",
                str(darkbay / "phase3" / "data"),
                "--out",
                str(darkbay_parseq),
                "--real-dir",
                str(DARKBAY_REAL),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
            ],
            darkbay_parseq.with_suffix(".log"),
            status,
            "darkbay_parseq_train",
        )
        write_json(status, {"state": "complete", "root": str(args.root), "updated_unix": time.time()})
    except BaseException as error:
        write_json(
            args.root / "failure.json",
            {"error": repr(error), "traceback": traceback.format_exc(), "updated_unix": time.time()},
        )
        write_json(status, {"state": "failed", "error": repr(error), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
