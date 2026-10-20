"""Measure peak CUDA memory for one real-data optimizer step per model."""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('train_manifest', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / 'shared'))
    from adapters import Recognizer
    from runner import ManifestDataset, loader, seed_everything

    results = []
    candidate_batch_sizes = {
        'PARSeq': 64, 'ViTSTR': 32, 'SATRN': 8, 'TrOCR': 8,
        'SVTR': 64, 'SVTRv2': 32, 'CRNN': 64,
    }
    for name in ['PARSeq', 'ViTSTR', 'SATRN', 'TrOCR', 'SVTR', 'SVTRv2', 'CRNN']:
        cfg = json.loads((args.root / name / 'config.json').read_text(encoding='utf-8'))
        cfg['batch_size'] = candidate_batch_sizes[name]
        cfg['effective_batch_size'] = 64
        cfg['max_label_length'] = 8
        cfg['train_manifest'] = str(args.train_manifest.resolve())
        seed_everything(cfg['seed'])
        dataset = ManifestDataset(args.train_manifest, cfg)
        images, labels = next(iter(loader(dataset, cfg, shuffle=False)))
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = Recognizer(cfg).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                      weight_decay=cfg['weight_decay'])
        loss = model.loss(images.to(device), list(labels))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        optimizer.step()
        torch.cuda.synchronize()
        result = {
            'model': name,
            'batch_size': len(labels),
            'input_shape': list(images.shape),
            'loss': float(loss),
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del optimizer, model, images, loss
        gc.collect()
        torch.cuda.empty_cache()
    output = {'gpu': torch.cuda.get_device_name(0), 'models': results}
    (args.root / 'training_memory_benchmark.json').write_text(
        json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
