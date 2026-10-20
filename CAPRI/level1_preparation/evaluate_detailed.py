"""Write per-image predictions and transfer diagnostics for a frozen checkpoint."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch


def edit_distance(a, b):
    last = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, last[j] + 1, last[j - 1] + (x != y)))
        last = row
    return last[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('model')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / 'shared'))
    from adapters import Recognizer
    from runner import ManifestDataset, loader

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = state['config']
    dataset = ManifestDataset(args.manifest, cfg)
    model = Recognizer(cfg).to('cuda')
    model.load_state_dict(state['model'])
    model.eval()
    rows = []
    offset = 0
    with torch.no_grad():
        for images, labels in loader(dataset, cfg):
            predictions = model.predict(images.to('cuda'))
            for pred, label in zip(predictions, labels):
                path = str(dataset.rows[offset][0])
                distance = edit_distance(pred, label)
                rows.append({'path': path, 'label': label, 'prediction': pred,
                             'edit_distance': distance, 'exact': pred == label,
                             'length_correct': len(pred) == len(label)})
                offset += 1
    total_chars = sum(len(row['label']) for row in rows)
    edit_errors = sum(row['edit_distance'] for row in rows)
    positional_correct = sum(
        sum(a == b for a, b in zip(row['label'], row['prediction'])) for row in rows)
    report = {
        'model': args.model,
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_epoch': state['epoch'],
        'samples': len(rows),
        'exact_correct': sum(row['exact'] for row in rows),
        'exact_string_accuracy': sum(row['exact'] for row in rows) / len(rows),
        'character_error_rate': edit_errors / total_chars,
        'edit_similarity': 1 - edit_errors / total_chars,
        'position_match_rate_against_gt_length': positional_correct / total_chars,
        'length_accuracy': sum(row['length_correct'] for row in rows) / len(rows),
        'empty_predictions': sum(not row['prediction'] for row in rows),
        'prediction_lengths': dict(sorted(Counter(len(row['prediction']) for row in rows).items())),
        'best_partial_examples': sorted(rows, key=lambda row: (row['edit_distance'], row['label']))[:20],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
