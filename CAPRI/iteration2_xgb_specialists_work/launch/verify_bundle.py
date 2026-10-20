"""Check the completed replacement bundle, aligned caches and rendered previews."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from PIL import Image
from fit_xgb import (BASE, CHECKPOINT, EXPECTED, REMOVED, SPLITS, SPECIALISTS,
                     dump, load_encoder, load_models, read, score_values, sha, tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    bundle = out/'bundle'
    manifest = read(bundle/'encoder_manifest.json')
    assert sha(CHECKPOINT) == EXPECTED
    assert sha(BASE/'baseline_training/epoch_15/ema.pt') == read(SPECIALISTS/'configuration.json')['baseline_sha256']
    assert sha(bundle/'encoder.pt') == manifest['encoder_sha256']
    encoder, _ = load_encoder(bundle/'encoder.pt')
    assert not any(hasattr(encoder, name) for name in REMOVED)
    assert not any(name.split('.')[0] in REMOVED for name in encoder.state_dict())
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 2179108
    specs = read(bundle/'heads.json')['tasks']
    expected_tasks = tasks()
    assert {task['name'] for task in specs} == {task['name'] for task in expected_tasks}
    by_name = {task['name']: task for task in specs}
    for task in expected_tasks:
        assert all(by_name[task['name']][key] == value for key, value in task.items())
    models = load_models(out, specs)
    rounds = []
    for task in specs:
        if 'constant' not in task:
            count = models[task['name']].num_boosted_rounds()
            assert count == task['best_iteration'] + 1 and count <= 2000
            rounds.append(count)
    caches = {}
    for split in SPLITS:
        directory = out/'features'/split
        info = read(directory/'manifest.json')
        ids = read(directory/'row_manifest.json')['ids']
        assert len(ids) == info['count']
        assert hashlib.sha256('\n'.join(ids).encode()).hexdigest() == info['row_identity_sha256']
        assert info['checkpoint_sha256'] == EXPECTED
        for key, expected_hash in info['feature_hashes'].items():
            path = directory/f'{key}.npy'
            assert sha(path) == expected_hash
            data = np.load(path, mmap_mode='r')
            assert list(data.shape) == info['shapes'][key] and data.dtype == np.float16
        caches[split] = dict(count=info['count'], hashes_verified=True,
                             labels_sha256=sha(directory/'labels.npz'))
    scores = read(out/'synthetic_results.json')
    for split in ('selection', 'seen_test', 'generalisation'):
        values = dict(np.load(out/'evaluation'/f'{split}_predictions.npz'))
        labels = dict(np.load(out/'features'/split/'labels.npz'))
        assert score_values(values, labels, specs) == scores[split]
    predictions = [json.loads(line) for line in (out/'real_inference/predictions.jsonl').read_text().splitlines()]
    expected_counts = dict(CryptBB=600, City=2000, DarkBay=999)
    assert len(predictions) == 3599 and Counter(row['family'] for row in predictions) == expected_counts
    assert len({(row['family'], row['index']) for row in predictions}) == 3599
    for row in predictions:
        assert len(row['effects']) == 37 and len(row['text_features']) == 16 and len(row['font']) == 7
        assert type(row['font']['italic']) is bool
    cli = read(out/'cli_prediction.json')
    assert len(cli['effects']) == 37 and len(cli['text_features']) == 16 and len(cli['font']) == 7
    assert cli['neural_feature_heads_used'] is False
    dataset = out/'dataset'
    report = read(dataset/'report.json')
    plan = read(dataset/'sampling_plan.json')
    assert report['stage'] == 'complete' and not report['failures']
    rows = [json.loads(line) for line in (dataset/'labels.jsonl').read_text().splitlines()]
    assert len(rows) == 300 and Counter(row['family'] for row in rows) == dict.fromkeys(expected_counts, 100)
    profiles = {family: read(dataset/'profiles'/f'{family}.json') for family in expected_counts}
    for row in rows:
        assert row['renderer_usable'] is True
        assert row['length'] == len(row['text']) == row['requested']['length']
        assert row['prototype_id'] == plan['prototype_ids'][row['family']][row['index']]
        assert row['requested'] == profiles[row['family']]['joint_distribution'][row['prototype_id']]['description']
        with Image.open(dataset/row['image']) as image:
            assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() == row['pixel_sha256']
    assert len({row['pixel_sha256'] for row in rows}) == 300
    files = {str(path.relative_to(bundle)).replace('\\', '/'): dict(bytes=path.stat().st_size, sha256=sha(path))
             for path in bundle.rglob('*') if path.is_file() and path.name != 'checksums.json'}
    dump(bundle/'checksums.json', dict(source_checkpoint_sha256=EXPECTED, files=files,
                                      torch=torch.__version__, xgboost=xgb.__version__))
    verification = dict(passed=True, source_checkpoints_unchanged=True, encoder_heads_absent=True,
        encoder_parameters=2179108, encoder_frozen=True, task_routing_verified=True,
        heads=len(specs), saved_boosters=len(models), selected_rounds_median=float(np.median(rounds)),
        selected_rounds_max=max(rounds), selected_models_at_round_limit=sum(count == 2000 for count in rounds),
        caches=caches, benchmark_scores_replay_exact=True, cli_inference_verified=True,
        real_predictions=len(predictions), real_italic_counts=dict(Counter(row['family'] for row in predictions if row['font']['italic'])),
        preview_images=len(rows), preview_hashes_lengths_prototypes_verified=True,
        preview_failures=0, bundle_bytes=sum(value['bytes'] for value in files.values()))
    dump(out/'verification.json', verification)
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
