"""Mark an interrupted orchestration and its incomplete model runs as cancelled."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def update(path, reason):
    if not path.exists():
        return
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('state') not in {'complete', 'failed', 'test_failed'}:
        value['state'] = 'cancelled'
        value['cancelled_utc'] = datetime.now(timezone.utc).isoformat()
        value['reason'] = reason
        for job in value.get('jobs', {}).values():
            if job.get('state') in {'queued', 'training', 'testing'}:
                job['state'] = 'cancelled'
                job['reason'] = reason
        path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('run_name')
    parser.add_argument('reason')
    args = parser.parse_args()
    update(args.root / 'training_orchestration' / args.run_name / 'status.json', args.reason)
    for name in ['PARSeq', 'ViTSTR', 'SATRN', 'TrOCR', 'SVTR', 'SVTRv2', 'CRNN']:
        update(args.root / name / 'logs' / args.run_name / 'status.json', args.reason)


if __name__ == '__main__':
    main()
