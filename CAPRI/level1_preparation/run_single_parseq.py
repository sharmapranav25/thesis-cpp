"""Train one PARSeq experiment, then evaluate its selected checkpoint on real data."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_logged(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--launcher", default="train_v3_first10k.py")
    parser.add_argument("--dataset", default="darkbay_capri_v3_6char_first10k")
    args = parser.parse_args()
    work = args.root / "training_orchestration" / args.run_name
    work.mkdir(parents=True, exist_ok=False)
    status_path = work / "status.json"
    state = {"state": "training", "model": "PARSeq", "run_name": args.run_name,
             "started_utc": utcnow(), "driver_pid": __import__("os").getpid()}
    save(status_path, state)

    train_command = [sys.executable, str(args.root / "PARSeq" / args.launcher),
                     "--run-name", args.run_name, "--device", "cuda"]
    code = run_logged(train_command, work / "PARSeq.train.log")
    state.update(train_exit_code=code, train_finished_utc=utcnow())
    if code:
        state["state"] = "training_failed"
        save(status_path, state)
        raise SystemExit(code)

    state["state"] = "testing"
    save(status_path, state)
    checkpoint = args.root / "PARSeq" / "checkpoints" / args.run_name / "best.pt"
    test_manifest = args.root / "datasets" / args.dataset / "test.csv"
    report_dir = args.root / "PARSeq" / "logs" / args.run_name / "real_eval"
    eval_command = [sys.executable, str(args.root / "shared" / "evaluate_detailed.py"),
                    str(args.root), "PARSeq", str(checkpoint), str(test_manifest), str(report_dir)]
    code = run_logged(eval_command, work / "PARSeq.test.log")
    state.update(state="complete" if code == 0 else "testing_failed", test_exit_code=code,
                 test_finished_utc=utcnow(), checkpoint=str(checkpoint), report=str(report_dir / "report.json"))
    save(status_path, state)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
