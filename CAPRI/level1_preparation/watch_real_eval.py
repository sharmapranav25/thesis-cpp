"""Snapshot and evaluate every completed training epoch on a development set."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def snapshot_checkpoint(source, destination):
    state = torch.load(source, map_location="cpu", weights_only=True)
    slim = {
        "config": state["config"],
        "model": state["model"],
        "epoch": state["epoch"],
        "metrics": state["metrics"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(slim, temporary)
    temporary.replace(destination)
    return slim["epoch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("test_manifest", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()

    model_dir = args.root / "PARSeq"
    run_logs = model_dir / "logs" / args.run_name
    run_checkpoints = model_dir / "checkpoints" / args.run_name
    source_checkpoint = run_checkpoints / "last.pt"
    output_root = run_logs / "real_eval_by_epoch"
    watcher_status = output_root / "watcher_status.json"
    metrics_path = output_root / "real_metrics.jsonl"
    processed = set()
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                processed.add(json.loads(line)["epoch"])

    save_json(watcher_status, {
        "state": "watching", "started_utc": utcnow(), "pid": __import__("os").getpid(),
        "run_name": args.run_name, "test_manifest": str(args.test_manifest.resolve()),
        "policy": "development monitoring; repeated real-set evaluation",
        "processed_epochs": sorted(processed),
    })

    terminal_states = {"complete", "cancelled", "failed", "training_failed", "testing_failed"}
    while True:
        training_status = read_json(run_logs / "status.json") or {}
        epoch = int(training_status.get("epoch", 0) or 0)
        if epoch > 0 and epoch not in processed and source_checkpoint.exists():
            snapshot = run_checkpoints / "per_epoch" / f"epoch_{epoch:02d}.pt"
            snap_epoch = snapshot_checkpoint(source_checkpoint, snapshot)
            if snap_epoch != epoch:
                time.sleep(args.poll_seconds)
                continue
            report_dir = output_root / f"epoch_{epoch:02d}"
            log_path = output_root / f"epoch_{epoch:02d}.log"
            save_json(watcher_status, {
                "state": "evaluating", "epoch": epoch, "updated_utc": utcnow(),
                "processed_epochs": sorted(processed), "snapshot": str(snapshot),
            })
            command = [sys.executable, str(args.root / "shared" / "evaluate_detailed.py"),
                       str(args.root), "PARSeq", str(snapshot), str(args.test_manifest), str(report_dir)]
            with log_path.open("w", encoding="utf-8") as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                save_json(watcher_status, {
                    "state": "evaluation_failed", "epoch": epoch, "exit_code": code,
                    "updated_utc": utcnow(), "processed_epochs": sorted(processed),
                    "log": str(log_path),
                })
                time.sleep(args.poll_seconds)
                continue
            report = read_json(report_dir / "report.json")
            record = {
                "epoch": epoch,
                "exact_string_accuracy": report["exact_string_accuracy"],
                "character_error_rate": report["character_error_rate"],
                "position_match_rate_against_gt_length": report["position_match_rate_against_gt_length"],
                "length_accuracy": report["length_accuracy"],
                "exact_correct": report["exact_correct"],
                "samples": report["samples"],
                "report": str((report_dir / "report.json").resolve()),
                "checkpoint": str(snapshot.resolve()),
                "evaluated_utc": utcnow(),
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            processed.add(epoch)
            save_json(watcher_status, {
                "state": "watching", "updated_utc": utcnow(),
                "processed_epochs": sorted(processed), "latest": record,
            })

        state = training_status.get("state")
        expected = int(training_status.get("epochs", 0) or 0)
        if state in terminal_states and (epoch == 0 or epoch in processed or epoch >= expected):
            save_json(watcher_status, {
                "state": "complete", "training_state": state, "finished_utc": utcnow(),
                "processed_epochs": sorted(processed),
            })
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
