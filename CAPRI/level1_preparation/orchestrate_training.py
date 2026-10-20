"""Run the seven DarkBay baselines in waves of at most three GPU processes."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MODELS = ['PARSeq', 'ViTSTR', 'SATRN', 'TrOCR', 'SVTR', 'SVTRv2', 'CRNN']


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--run-name', default='darkbay_capri_v2_100k_seed42')
    args = parser.parse_args()
    if not 1 <= args.parallel <= 3:
        raise ValueError('This RTX A4000 schedule is validated for 1–3 parallel jobs')
    orchestration = args.root / 'training_orchestration' / args.run_name
    orchestration.mkdir(parents=True, exist_ok=False)
    status_path = orchestration / 'status.json'
    test_manifest = args.root / 'datasets' / 'darkbay_capri_v2_100k' / 'test.csv'
    state = {
        'state': 'training', 'started_utc': utcnow(), 'parallel_limit': args.parallel,
        'run_name': args.run_name, 'dataset': 'darkbay_capri_v2_100k',
        'jobs': {name: {'state': 'queued'} for name in MODELS},
    }
    save(status_path, state)
    queue = list(MODELS)
    active = {}
    successful = []
    while queue or active:
        while queue and len(active) < args.parallel:
            name = queue.pop(0)
            command = [sys.executable, str(args.root / name / 'train.py'),
                       '--run-name', args.run_name, '--device', 'cuda']
            log_path = orchestration / f'{name}.train.log'
            log = log_path.open('w', encoding='utf-8')
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            active[name] = (process, log)
            state['jobs'][name] = {'state': 'training', 'pid': process.pid,
                                   'started_utc': utcnow(), 'log': str(log_path)}
            save(status_path, state)
        time.sleep(10)
        for name, (process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            state['jobs'][name].update(state='trained' if code == 0 else 'failed',
                                       exit_code=code, finished_utc=utcnow())
            if code == 0:
                successful.append(name)
            del active[name]
            save(status_path, state)

    state['state'] = 'testing'
    save(status_path, state)
    for name in successful:
        checkpoint = args.root / name / 'checkpoints' / args.run_name / 'best.pt'
        command = [sys.executable, str(args.root / name / 'train.py'), '--device', 'cuda',
                   '--evaluate', str(checkpoint), '--val-manifest', str(test_manifest)]
        log_path = orchestration / f'{name}.test.log'
        state['jobs'][name]['state'] = 'testing'
        state['jobs'][name]['test_log'] = str(log_path)
        save(status_path, state)
        with log_path.open('w', encoding='utf-8') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        state['jobs'][name].update(state='complete' if code == 0 else 'test_failed',
                                   test_exit_code=code, test_finished_utc=utcnow(),
                                   checkpoint=str(checkpoint))
        save(status_path, state)
    state['state'] = 'complete' if len(successful) == len(MODELS) and all(
        state['jobs'][name]['state'] == 'complete' for name in MODELS) else 'completed_with_failures'
    state['finished_utc'] = utcnow()
    save(status_path, state)


if __name__ == '__main__':
    main()
