"""Atomic run records and bounded run destinations for the decoder pipeline."""
import hashlib
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def run_path(value):
    # Preserve logical paths in frozen manifests even when a run is physically
    # stored on another drive through its explicitly configured directory link.
    p = Path(os.path.abspath(ROOT / value))
    workspace_runs = Path(os.path.abspath(ROOT / 'runs'))
    config_path = ROOT / 'configs/local.json'
    storage = read(config_path).get('decoder_storage') if config_path.exists() else None
    logical = physical = None
    if storage:
        logical = Path(os.path.abspath(ROOT / storage['logical_path']))
        physical = Path(storage['physical_path']).resolve()
        if not logical.is_relative_to(workspace_runs) or logical == workspace_runs:
            raise ValueError('Configured decoder storage must name one workspace run')
        if p.is_relative_to(physical):
            p = logical / p.relative_to(physical)
    if not p.is_relative_to(workspace_runs):
        raise ValueError('New decoder outputs must be under workspace runs/')
    actual = p.resolve()
    if not actual.is_relative_to(workspace_runs.resolve()):
        if logical is None or not p.is_relative_to(logical) or actual != physical / p.relative_to(logical):
            raise ValueError('External decoder storage does not match the configured directory link')
    p.mkdir(parents=True, exist_ok=True)
    return p


def event(run, stage, state, **fields):
    rec = dict(updated_at=now(), stage=stage, state=state, pid=os.getpid(), **fields)
    write(Path(run) / 'status.json', rec)
    with (Path(run) / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


class StageLock:
    def __init__(self, run, stage):
        self.path = Path(run) / (stage + '.lock.json')

    def __enter__(self):
        import psutil
        if self.path.exists():
            old = read(self.path)
            try:
                p = psutil.Process(old['pid'])
                if abs(p.create_time() - old['process_created']) < 1:
                    raise RuntimeError('Stage already running: ' + str(self.path))
            except psutil.NoSuchProcess:
                pass
            # Preserve stale lock as evidence, then atomically claim a new lock.
            self.path.rename(self.path.with_name(self.path.name + '.stale.' + str(time.time_ns())))
        p = psutil.Process()
        with self.path.open('x', encoding='utf-8') as f:
            json.dump(dict(pid=p.pid, process_created=p.create_time(), created_at=now()), f)
        return self

    def __exit__(self, *args):
        self.path.unlink(missing_ok=True)
