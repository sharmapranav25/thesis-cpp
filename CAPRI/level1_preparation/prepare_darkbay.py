"""Audit the DarkBay filename-labeled corpus and create leakage-safe manifests."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image


def inspect(folder: Path):
    rows = []
    dimensions = Counter()
    pixel_hashes = {}
    for path in sorted(folder.glob('*.png')):
        label = path.stem
        if len(label) != 6 or not label.isascii() or not label.isalnum() or label != label.upper():
            raise ValueError(f'Invalid filename label: {path}')
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            dimensions[f'{rgb.width}x{rgb.height}'] += 1
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if digest in pixel_hashes:
            raise ValueError(f'Duplicate pixels: {path} and {pixel_hashes[digest]}')
        pixel_hashes[digest] = str(path.resolve())
        rows.append((str(path.resolve()), label, digest))
    if not rows:
        raise ValueError(f'No PNG files found in {folder}')
    return rows, dimensions


def write_manifest(path: Path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'label'])
        writer.writerows((image, label) for image, label, _ in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-count', type=int, default=80)
    args = parser.parse_args()

    pool, pool_dims = inspect(args.dataset_root / 'pool')
    test, test_dims = inspect(args.dataset_root / 'eval')
    pool_names = {Path(row[0]).name for row in pool}
    test_names = {Path(row[0]).name for row in test}
    pool_hashes = {row[2] for row in pool}
    test_hashes = {row[2] for row in test}
    if pool_names & test_names:
        raise ValueError('Filename overlap between pool and test split')
    if pool_hashes & test_hashes:
        raise ValueError('Pixel overlap between pool and test split')
    if not 0 < args.validation_count < len(pool):
        raise ValueError('validation-count must be between 1 and pool size - 1')

    shuffled = list(pool)
    random.Random(args.seed).shuffle(shuffled)
    validation = sorted(shuffled[:args.validation_count])
    train = sorted(shuffled[args.validation_count:])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / 'train.csv', train)
    write_manifest(args.output_dir / 'validation.csv', validation)
    write_manifest(args.output_dir / 'test.csv', test)
    audit = {
        'dataset': 'DarkBay',
        'label_source': 'PNG filename stem',
        'seed': args.seed,
        'counts': {'train': len(train), 'validation': len(validation), 'test': len(test)},
        'source_counts': {'pool': len(pool), 'eval': len(test)},
        'dimensions': {'pool': dict(pool_dims), 'eval': dict(test_dims)},
        'label_lengths': dict(Counter(len(row[1]) for row in pool + test)),
        'characters': ''.join(sorted(set(''.join(row[1] for row in pool + test)))),
        'duplicate_pixels': {'pool': 0, 'eval': 0},
        'cross_split_overlap': {'filenames': 0, 'pixels': 0},
        'test_role': 'final evaluation only; excluded from training and checkpoint selection',
    }
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
