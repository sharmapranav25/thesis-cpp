"""Train CRNN and PARSeq sequentially, then aggregate fit-only CryptBB scheme predictions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def groups(text: str) -> str:
    return "".join(
        str(int(any(character in alphabet for character in text)))
        for alphabet in ("0123456789", "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {
        "images": len(rows),
        "predicted_length_histogram": dict(sorted(Counter(row["decoded_length"] for row in rows).items())),
        "predicted_group_combinations": dict(sorted(Counter(groups(row["prediction"]) for row in rows).items())),
        "predicted_group_prevalence": {
            "digits": sum(groups(row["prediction"])[0] == "1" for row in rows) / len(rows),
            "lowercase": sum(groups(row["prediction"])[1] == "1" for row in rows) / len(rows),
            "uppercase": sum(groups(row["prediction"])[2] == "1" for row in rows) / len(rows),
        },
        "mean_confidence": sum(row["confidence"] for row in rows) / len(rows),
        "real_accuracy": None,
    }


def aggregate(root: Path) -> dict:
    fit_rows = {}
    model_summaries = {}
    validation_summaries = {}
    hashes = {}
    for model in ("crnn", "parseq"):
        run = root / f"{model}_plain"
        fit_rows[model] = [json.loads(line) for line in (run / "fit_predictions.jsonl").read_text().splitlines()]
        model_summaries[model] = summarize(run / "fit_predictions.jsonl")
        validation_summaries[model] = summarize(run / "validation_predictions.jsonl")
        hashes[model] = read(run / "status.json")["checkpoint_sha256"]
    count = len(fit_rows["crnn"])
    disagreement = {
        "decoded_length": sum(
            len(fit_rows["crnn"][i]["prediction"]) != len(fit_rows["parseq"][i]["prediction"])
            for i in range(count)
        ) / count,
        "character_groups": sum(
            groups(fit_rows["crnn"][i]["prediction"]) != groups(fit_rows["parseq"][i]["prediction"])
            for i in range(count)
        ) / count,
        "decoded_text": sum(
            fit_rows["crnn"][i]["prediction"] != fit_rows["parseq"][i]["prediction"] for i in range(count)
        ) / count,
    }
    scheme = {
        "schema": 1,
        "stage": "Official CAPRI Phase 3",
        "family": "cryptbb",
        "source_partition": "fit",
        "images": count,
        "models": hashes,
        "model_summaries": model_summaries,
        "model_disagreement": disagreement,
        "validation_predictions": validation_summaries,
        "validation_used_for_scheme": False,
        "alphabet_policy": "All 62 characters retained; absent predictions do not prove exclusion.",
        "real_accuracy": None,
        "ground_truth_status": "No real transcription labels used; filenames are locators only.",
        "status": "estimated_with_model_disagreement",
    }
    write(root / "inferred_scheme.json", scheme)
    return scheme


def main(args: argparse.Namespace) -> None:
    lock = args.root / "coordinator.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))

    def status(state: str, **extra) -> None:
        row = {
            "state": state,
            "pid": os.getpid(),
            "updated_unix": time.time(),
            "run": str(args.root),
            **extra,
        }
        write(args.root / "status.json", row)
        write(args.project / "monitor/capri_phase3_cryptbb_status.json", row)

    try:
        for model in ("crnn", "parseq"):
            tag = f"{model}_plain"
            out = args.root / tag
            if out.exists():
                if (out / "status.json").exists() and read(out / "status.json").get("state") == "complete":
                    continue
                raise RuntimeError(f"Existing incomplete run preserved: {out}")
            command = [
                sys.executable,
                str(args.reader_code / "train.py"),
                "--model",
                model,
                "--data",
                str(args.root / "data"),
                "--out",
                str(out),
                "--epochs",
                "10",
            ]
            with (args.root / f"{tag}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    child_status = read(out / "status.json") if (out / "status.json").exists() else {"state": "initializing"}
                    status("training", active=tag, child_pid=child.pid, child=child_status)
                    time.sleep(15)
                if child.returncode:
                    raise RuntimeError(f"{tag} failed; inspect its log and failure.json")
        status("aggregating_fit_only")
        scheme = aggregate(args.root)
        status("awaiting_scheme_review", scheme=str(args.root / "inferred_scheme.json"), models=list(scheme["models"]))
    except BaseException as error:
        write(args.root / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        status("failed", error=repr(error))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reader-code", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args())
