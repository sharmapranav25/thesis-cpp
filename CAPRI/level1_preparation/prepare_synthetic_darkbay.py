"""Convert the finalized CAPRI-v2 DarkBay corpus into train/validation manifests."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def pixel_hash(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'label', 'pixel_sha256'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('synthetic_root', type=Path)
    parser.add_argument('real_test_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    status = json.loads((args.synthetic_root / 'status.json').read_text(encoding='utf-8'))
    if status.get('state') != 'complete' or status.get('generated') != 100000:
        raise RuntimeError(f'Synthetic corpus is not finalized: {status.get("state")}, '
                           f'{status.get("generated")}/100000 generated, '
                           f'{status.get("audited", 0)}/100000 audited')
    protocol = json.loads((args.synthetic_root / 'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('family') != 'darkbay' or protocol.get('partition') != 'train':
        raise ValueError('Expected finalized DarkBay train corpus')

    train, validation = [], []
    seen_pixels = set()
    labels = Counter()
    manifest = args.synthetic_root / 'manifest.jsonl'
    with manifest.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            label = record['label']
            digest = record['pixel_sha256'].lower()
            image = (args.synthetic_root / record['filename']).resolve()
            if record.get('partition') != 'train':
                raise ValueError(f'{manifest}:{line_number}: non-train record')
            if not 2 <= len(label) <= 8 or set(label) - set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'):
                raise ValueError(f'{manifest}:{line_number}: invalid label {label!r}')
            if digest in seen_pixels:
                raise ValueError(f'{manifest}:{line_number}: duplicate synthetic pixels')
            if not image.is_file():
                raise FileNotFoundError(image)
            seen_pixels.add(digest)
            labels[label] += 1
            row = {'path': str(image), 'label': label, 'pixel_sha256': digest}
            # Label-grouped holdout: duplicate strings cannot cross the boundary.
            bucket = int(hashlib.sha256(('level1-darkbay-val-v1:' + label).encode()).hexdigest()[:8], 16) % 20
            (validation if bucket == 0 else train).append(row)
    if len(train) + len(validation) != 100000:
        raise ValueError(f'Expected 100000 manifest records, got {len(train) + len(validation)}')

    test = []
    test_hashes = set()
    for image in sorted(args.real_test_dir.glob('*.png')):
        label = image.stem
        if len(label) != 6 or set(label) - set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            raise ValueError(f'Invalid real test label filename: {image}')
        digest = pixel_hash(image)
        if digest in test_hashes:
            raise ValueError(f'Duplicate real test pixels: {image}')
        test_hashes.add(digest)
        test.append({'path': str(image.resolve()), 'label': label, 'pixel_sha256': digest})
    overlap = seen_pixels & test_hashes
    if overlap:
        raise ValueError(f'{len(overlap)} synthetic/real pixel hashes overlap')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'train.csv', train)
    write_csv(args.output_dir / 'validation.csv', validation)
    write_csv(args.output_dir / 'test.csv', test)
    audit = {
        'dataset': 'darkbay_capri_v2_100k',
        'synthetic_protocol_sha256': hashlib.sha256((args.synthetic_root / 'protocol.json').read_bytes()).hexdigest(),
        'synthetic_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'counts': {'train': len(train), 'validation': len(validation), 'real_test': len(test)},
        'unique_synthetic_labels': len(labels),
        'duplicate_synthetic_labels': sum(count - 1 for count in labels.values()),
        'synthetic_pixel_duplicates': 0,
        'synthetic_real_pixel_overlap': 0,
        'charset': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        'lengths': [2, 3, 4, 5, 6, 7, 8],
        'validation_rule': 'SHA256(level1-darkbay-val-v1:label) mod 20 == 0; grouped by label',
        'real_test_role': 'evaluated only after checkpoint selection; never used for training or selection',
    }
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
