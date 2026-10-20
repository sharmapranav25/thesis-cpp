import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path

import torch


def infer(model, dataset, loader_fn, cfg, device):
    rows = []
    offset = 0
    with torch.no_grad():
        for images, labels in loader_fn(dataset, cfg):
            distributions = model.net(images.to(device)).softmax(-1)
            predictions, probabilities = model.net.tokenizer.decode(distributions)
            for prediction, probs, label in zip(predictions, probabilities, labels):
                char_probs = [float(x) for x in probs[:len(prediction)].cpu()]
                rows.append({
                    'path': str(dataset.rows[offset][0]), 'label': label,
                    'prediction': prediction, 'char_probs': char_probs,
                    'exact': prediction == label,
                })
                offset += 1
    return rows


def quantile_threshold(values, coverage):
    ordered = sorted(values, reverse=True)
    count = max(1, math.ceil(len(ordered) * coverage))
    return ordered[count - 1]


def evaluate_rule(name, validation, real, score_key):
    result = []
    val_scores = [row[score_key] for row in validation]
    for target in (0.01, 0.05, 0.10, 0.25, 0.50, 1.00):
        threshold = quantile_threshold(val_scores, target)
        selected = [row for row in real if row[score_key] >= threshold]
        correct = sum(row['exact'] for row in selected)
        result.append({
            'rule': name, 'synthetic_target_coverage': target,
            'threshold': threshold, 'real_attempts': len(selected),
            'real_coverage': len(selected) / len(real),
            'real_exact_correct': correct,
            'real_accuracy_when_attempted': correct / len(selected) if selected else None,
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('validation_manifest', type=Path)
    ap.add_argument('real_manifest', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.root / 'shared'))
    from adapters import Recognizer
    from runner import ManifestDataset, loader

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = state['config']
    validation_ds = ManifestDataset(args.validation_manifest, cfg)
    real_ds = ManifestDataset(args.real_manifest, cfg)
    model = Recognizer(cfg).to('cuda')
    model.load_state_dict(state['model'])
    model.eval()
    validation = infer(model, validation_ds, loader, cfg, 'cuda')
    real = infer(model, real_ds, loader, cfg, 'cuda')

    totals = defaultdict(int)
    correct = defaultdict(int)
    for row in validation:
        for position, (char, probability) in enumerate(zip(row['prediction'], row['char_probs'])):
            totals[char] += 1
            if position < len(row['label']) and char == row['label'][position]:
                correct[char] += 1
    global_precision = sum(correct.values()) / max(1, sum(totals.values()))
    # Bayesian shrinkage prevents rare predicted characters from looking perfectly reliable.
    strength = 50
    reliability = {
        char: (correct[char] + strength * global_precision) / (totals[char] + strength)
        for char in cfg['charset']
    }

    for collection in (validation, real):
        for row in collection:
            if len(row['prediction']) != cfg['max_label_length'] or len(row['char_probs']) < cfg['max_label_length']:
                row['raw_min'] = 0.0
                row['char_aware_min'] = 0.0
            else:
                row['raw_min'] = min(row['char_probs'])
                row['char_aware_min'] = min(
                    probability * reliability[char]
                    for char, probability in zip(row['prediction'], row['char_probs'])
                )

    table = evaluate_rule('minimum_raw_character_confidence', validation, real, 'raw_min')
    table += evaluate_rule('character_aware_minimum', validation, real, 'char_aware_min')
    char_table = sorted(({
        'character': char, 'synthetic_predictions': totals[char],
        'synthetic_correct': correct[char],
        'synthetic_precision': correct[char] / totals[char] if totals[char] else None,
        'smoothed_reliability': reliability[char],
    } for char in cfg['charset']), key=lambda x: x['smoothed_reliability'], reverse=True)
    result = {
        'checkpoint': str(args.checkpoint), 'checkpoint_epoch': state['epoch'],
        'calibration_source': str(args.validation_manifest),
        'real_labels_used_for_thresholds': False,
        'synthetic_validation_samples': len(validation), 'real_evaluation_samples': len(real),
        'all_real_exact_correct': sum(row['exact'] for row in real),
        'all_real_exact_accuracy': sum(row['exact'] for row in real) / len(real),
        'coverage_table': table, 'character_reliability': char_table,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (args.output / 'coverage.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys()); writer.writeheader(); writer.writerows(table)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
