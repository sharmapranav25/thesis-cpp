"""Freeze equal-compute DarkBay training configs for all prepared models."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('dataset_dir', type=Path)
    args = parser.parse_args()
    batches = {
        'PARSeq': 64,
        'ViTSTR': 32,
        'SATRN': 8,
        'TrOCR': 8,
        'SVTR': 64,
        'SVTRv2': 32,
        'CRNN': 64,
    }
    for name, batch_size in batches.items():
        path = args.root / name / 'config.json'
        backup = args.root / name / 'config.generic.json'
        if not backup.exists():
            shutil.copy2(path, backup)
        cfg = json.loads(path.read_text(encoding='utf-8'))
        cfg.update({
            'experiment': 'darkbay_capri_v2_100k_level1',
            'charset': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            'max_label_length': 8,
            'batch_size': batch_size,
            'effective_batch_size': 64,
            'epochs': 15,
            'learning_rate': 0.0003,
            'weight_decay': 0.01,
            'warmup_fraction': 0.1,
            'gradient_clip': 5.0,
            'seed': 42,
            'num_workers': 4,
            'precision': 'fp32',
            'initialization': 'random',
            'train_manifest': str((args.dataset_dir / 'train.csv').resolve()),
            'val_manifest': str((args.dataset_dir / 'validation.csv').resolve()),
            'test_manifest': str((args.dataset_dir / 'test.csv').resolve()),
            'checkpoint_selection': 'highest synthetic validation exact-string accuracy',
            'real_test_policy': 'evaluate selected best checkpoint once after training',
        })
        path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        print(f'{name}: batch={batch_size}, effective=64, epochs=15')


if __name__ == '__main__':
    main()
