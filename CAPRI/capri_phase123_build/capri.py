"""CAPRI Phase 1-2-3 unified command-line package."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASE1 = ROOT / "phase1"
PHASE2 = ROOT / "phase2"
PHASE3 = ROOT / "phase3"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root, manifest):
    values = json.loads(manifest.read_text(encoding="utf-8-sig"))
    for relative, expected in values.items():
        path = root / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Integrity mismatch: {path}")


def verify():
    verify_manifest(ROOT, ROOT / "PACKAGE_INTEGRITY.json")
    verify_manifest(PHASE1, PHASE1 / "INTEGRITY.json")
    verify_manifest(PHASE2, PHASE2 / "INTEGRITY.json")
    print("CAPRI PHASE 1-2-3 VERIFIED")


def run(command):
    subprocess.run([sys.executable, *map(str, command)], check=True)


def profile(args):
    run(
        [
            PHASE1 / "capri.py",
            "profile",
            "--references",
            args.references,
            "--out",
            args.out,
            "--name",
            args.name,
            "--kind",
            args.kind,
            "--device",
            args.device,
            "--variant",
            args.variant,
        ]
    )


def broad(args):
    command = [
        PHASE2 / "generate_connected.py",
        "--base",
        PHASE1,
        "--count",
        args.count,
        "--seed",
        args.seed,
        "--partition",
        args.partition,
        "--out",
        args.out,
        "--mixed-fraction",
        args.mixed_fraction,
        "--lengths",
        *args.lengths,
    ]
    command.extend(["--profile", args.profile] if args.profile else ["--family", args.family])
    run(command)


def prepare(args):
    run(
        [
            PHASE3 / "prepare.py",
            "--family",
            args.family,
            "--corpus",
            args.corpus,
            "--split",
            args.real_split,
            "--out",
            Path(args.run) / "data",
        ]
    )


def train(args):
    run([PHASE3 / "reader/coordinator.py", "--root", args.run, "--family", args.family])


def phase3_run(args):
    run_root = Path(args.run)
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True)
    prepare(args)
    train(args)


def targeted(args):
    command = [
        PHASE3 / "targeted.py",
        "--scheme",
        args.scheme,
        "--out",
        args.out,
        "--count",
        args.count,
        "--seed",
        args.seed,
        "--mixed-fraction",
        args.mixed_fraction,
    ]
    if args.profile:
        command.extend(["--profile", args.profile])
    run(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Verify the unified package and its frozen phase snapshots")

    phase1 = sub.add_parser("profile", help="Phase 1: measure and freeze a real-family visual profile")
    phase1.add_argument("--references", required=True)
    phase1.add_argument("--out", required=True)
    phase1.add_argument("--name", required=True)
    phase1.add_argument("--kind", choices=["conformer", "gridded"], default="conformer")
    phase1.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    phase1.add_argument("--variant", choices=["0", "1", "2", "3"], default="0")

    phase2 = sub.add_parser("generate-broad", help="Phase 2: generate broad-font labelled synthetic data")
    phase2_source = phase2.add_mutually_exclusive_group(required=True)
    phase2_source.add_argument("--family")
    phase2_source.add_argument("--profile")
    phase2.add_argument("--count", type=int, default=100000)
    phase2.add_argument("--seed", type=int, required=True)
    phase2.add_argument("--partition", choices=["train", "validation", "test"], default="train")
    phase2.add_argument("--out", required=True)
    phase2.add_argument("--lengths", nargs="+", type=int, default=list(range(2, 9)))
    phase2.add_argument("--mixed-fraction", type=float, default=0.5)

    for name, help_text in (
        ("phase3-prepare", "Phase 3: audit/cache synthetic data and preserve real fit/validation boundaries"),
        ("phase3-run", "Phase 3: prepare, train all reader variants, freeze, infer and aggregate"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--family", required=True)
        command.add_argument("--corpus", required=True)
        command.add_argument("--real-split", required=True)
        command.add_argument("--run", required=True)

    phase3_train = sub.add_parser("phase3-train", help="Phase 3: train/freeze/infer using an existing prepared run")
    phase3_train.add_argument("--family", required=True)
    phase3_train.add_argument("--run", required=True)

    phase3_target = sub.add_parser("generate-targeted", help="Phase 3: generate the final corpus from a reviewed scheme")
    phase3_target.add_argument("--scheme", required=True)
    phase3_target.add_argument("--profile")
    phase3_target.add_argument("--out", required=True)
    phase3_target.add_argument("--count", type=int, default=100000)
    phase3_target.add_argument("--seed", type=int, required=True)
    phase3_target.add_argument("--mixed-fraction", type=float, default=0.5)

    args = parser.parse_args()
    verify()
    if args.command == "verify":
        return
    if args.command == "profile":
        profile(args)
    elif args.command == "generate-broad":
        broad(args)
    elif args.command == "phase3-prepare":
        run_root = Path(args.run)
        if run_root.exists():
            raise FileExistsError(run_root)
        run_root.mkdir(parents=True)
        prepare(args)
    elif args.command == "phase3-train":
        train(args)
    elif args.command == "phase3-run":
        phase3_run(args)
    elif args.command == "generate-targeted":
        targeted(args)


if __name__ == "__main__":
    main()
