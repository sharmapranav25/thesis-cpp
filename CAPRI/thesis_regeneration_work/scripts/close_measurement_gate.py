"""Record a gate only after explicit visual-review evidence and numerical passes."""
import argparse
import json
import hashlib
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    path = args.run / 'gate.json'
    if path.exists():
        raise FileExistsError(path)
    summary = json.loads((args.run / 'summary.json').read_text(encoding='utf-8'))
    review_path = args.run / 'visual_review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    required = {'cryptbb', 'city', 'courier', 'darkbay'}
    numerical = {r['dataset']: r['numerical_gate'] for r in summary['datasets']}
    visual = {k: v['status'] for k, v in review['datasets'].items()}
    passed = set(numerical) == required and set(visual) == required and all(numerical[k] == visual[k] == 'PASS' for k in required)
    receipt = {'method': summary['method'], 'status': 'PASS' if passed else 'FAIL',
               'numerical': numerical, 'visual': visual, 'scope': review['scope'],
               'review_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
               'summary_sha256': hashlib.sha256((args.run/'summary.json').read_bytes()).hexdigest()}
    path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))
    if not passed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
