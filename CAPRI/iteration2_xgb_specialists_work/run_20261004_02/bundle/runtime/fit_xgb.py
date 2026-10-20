"""Fit 107 XGBoost replacements on the frozen specialist encoder's own queries."""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
import xgboost as xgb
import yaml
from PIL import Image
from feature_bank.iter2 import data as D, metrics as M
from feature_bank.iter2.generate import CATEGORIES, STROKES, TEXT_HEADS, effect_heads
from feature_bank.iter2.model import Iter2Model
from feature_bank.iter2.train import leak_check
if __package__:
    from .feature_encoder import Iter2FeatureEncoder, REMOVED, load_encoder
else:
    from feature_encoder import Iter2FeatureEncoder, REMOVED, load_encoder

BASE = PROJECT / 'runs/iter2_baseline_15_20261003_04'
SPECIALISTS = Path('C:/Users/pranav/Documents/ChatGPT/thesis/iteration2_specialists_work/run_20261003_01')
CHECKPOINT = SPECIALISTS / 'final/hybrid.pt'
EXPECTED = '198f89c05877a2e5053215ed53cc3e9be33d4a2a9dabc94f6e7842af415f6ccf'
SPLITS = ('train', 'selection', 'calibration', 'seen_test', 'generalisation')
PARAMS = dict(tree_method='hist', device='cuda', learning_rate=.05, max_depth=6,
              subsample=.8, colsample_bytree=.8, min_child_weight=1., reg_lambda=1.,
              nthread=8, seed=42)
ROUNDS, PATIENCE = 2000, 50
FONT_NAMES = ('category', 'stroke', 'weight', 'width', 'x_height', 'slant', 'italic')
GEOMETRY = set(TEXT_HEADS[:9])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def status(out, stage, **values):
    value = dict(stage=stage, time=time.strftime('%Y-%m-%d %H:%M:%S'), **values)
    dump(out / 'status.json', value)
    with (out / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def tasks():
    result = []
    # Fonts first, for early selection-only diagnostic feedback.
    for index, name in enumerate(FONT_NAMES):
        result.append(dict(name='font_' + name, group='font', head=name, index=index,
            kind='multiclass' if index < 2 else 'binary' if index == 6 else 'regression',
            feature='T', slot=17 + index))
    result.append(dict(name='length', group='length', head='length', index=0,
                       kind='multiclass', feature='T', slot=16))
    for group, names, feature in [('txt', TEXT_HEADS, 'T'), ('eff', effect_heads(), 'C')]:
        for index, name in enumerate(names):
            result.append(dict(name=f'{group}_presence_{name}', group=group, head=name,
                               index=index, kind='binary', feature=feature, slot=index))
            if group == 'eff' or name in GEOMETRY:
                result.append(dict(name=f'{group}_level_{name}', group=group, head=name,
                                   index=index, kind='regression', feature=feature, slot=index))
    assert len(result) == 107
    return result


def target(labels, task):
    if task['group'] in ('eff', 'txt'):
        y = labels[task['group'] + ('_present' if task['kind'] == 'binary' else '_level')][:, task['index']]
        valid = y >= 0 if task['kind'] == 'binary' else y > 0
    elif task['group'] == 'font':
        y = labels['font'][:, task['index']]
        valid = y >= 0  # Font zero is a valid measured label.
    else:
        y = labels['length']
        valid = y >= 0
    return y, valid


def matrix(features, task, valid=None):
    value = features[task['feature']][:, task['slot']]
    return np.asarray(value if valid is None else value[valid], dtype=np.float32, order='C')


def calibrate(probability, y):
    best = (-1., -1., 1.000001)
    for threshold in np.unique(np.r_[np.quantile(probability, np.linspace(0, 1, 401)), 1.000001]):
        predicted = probability >= threshold
        fpr = np.sum(predicted & (y == 0)) / max(1, np.sum(y == 0))
        if fpr <= .1:
            candidate = (M.f1(predicted, y == 1), float(np.sum(predicted & (y == 1))/max(1, np.sum(y == 1))), float(threshold))
            if candidate[:2] > best[:2]:
                best = candidate
    return best[2]


def export(out, model_config, training):
    directory = out / 'bundle'
    directory.mkdir(exist_ok=True)
    if (directory / 'encoder.pt').exists():
        manifest = read(directory / 'encoder_manifest.json')
        assert manifest['source_checkpoint_sha256'] == sha(CHECKPOINT) == EXPECTED
        assert manifest['encoder_sha256'] == sha(directory / 'encoder.pt')
        return
    full = Iter2Model(**model_config).eval()
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    full.load_state_dict(state, strict=True)
    encoder = Iter2FeatureEncoder(**model_config).eval()
    stripped = {key: value for key, value in state.items() if key.split('.')[0] not in REMOVED}
    encoder.load_state_dict(stripped, strict=True)
    assert not any(key.split('.')[0] in REMOVED for key in encoder.state_dict())
    assert not any(hasattr(encoder, name) for name in REMOVED)
    saved = {}
    hooks = [full.eff_pool.register_forward_hook(lambda m, a, v: saved.__setitem__('C', v.detach().clone())),
             full.txt_pool.register_forward_hook(lambda m, a, v: saved.__setitem__('T', v.detach().clone()))]
    items, _ = D.items_of(training)
    batch = D.collate(items[:8])
    images = torch.from_numpy(batch['image']).float() / 255
    columns = torch.from_numpy(batch['cols'])
    with torch.inference_mode():
        full(images, columns)
        actual = encoder(images, columns)
    differences = {}
    for key in ('C', 'T'):
        differences[key] = float((actual[key] - saved[key]).abs().max())
        torch.testing.assert_close(actual[key], saved[key], rtol=0, atol=0)
    for hook in hooks:
        hook.remove()
    torch.save(dict(config=model_config, state_dict=encoder.state_dict(),
                    source_checkpoint_sha256=EXPECTED, feature_keys=['C', 'T']), directory / 'encoder.pt')
    # Preserve the sequence reader separately for optional diagnostics. It is
    # absent from the exported feature encoder and never used by XGBoost.
    reader = {key: value for key, value in state.items() if key.split('.')[0] in ('col_pool', 'ctc_up', 'ctc')}
    torch.save(dict(config=model_config, state_dict=reader, optional=True), directory / 'optional_ctc_reader.pt')
    original_parameters = sum(p.numel() for p in full.parameters())
    parameters = sum(p.numel() for p in encoder.parameters())
    manifest = dict(source_checkpoint=str(CHECKPOINT), source_checkpoint_sha256=EXPECTED,
        encoder_sha256=sha(directory / 'encoder.pt'), removed_modules=list(REMOVED),
        original_neural_parameters=original_parameters, feature_encoder_parameters=parameters,
        removed_neural_parameters=original_parameters-parameters, feature_equivalence_max_abs=differences,
        C='37 learned effect-branch query vectors, 128 dimensions each; includes eff_block and learned pooling',
        T='24 learned transformer queries, 160 dimensions each: 16 text + length + seven font',
        reader='Optional CTC sidecar; no reader or neural prediction head is executed in feature inference')
    dump(directory / 'encoder_manifest.json', manifest)
    dump(out / 'export_checks.json', manifest)
    status(out, 'encoder_exported', parameters=parameters, removed_parameters=original_parameters-parameters,
           exact_feature_equivalence=differences)
    del full, encoder, state, stripped, reader
    gc.collect()


def extract(out, datasets, model_config):
    model, _ = load_encoder(out / 'bundle/encoder.pt', 'cuda')
    metadata = {}
    start = time.perf_counter()
    for split in SPLITS:
        directory = out / 'features' / split
        directory.mkdir(parents=True, exist_ok=True)
        done = directory / 'manifest.json'
        items, widths = D.items_of(datasets[split])
        ids = [str(s.cdir) + ':' + str(index) + ':' + s.meta['sha256'][index] for s, index in items]
        row_hash = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
        if done.exists():
            metadata[split] = read(done)
            assert metadata[split]['row_identity_sha256'] == row_hash
            assert metadata[split]['checkpoint_sha256'] == EXPECTED
            continue
        count = len(items)
        features = {key: np.lib.format.open_memmap(directory / f'{key}.npy', mode='w+', dtype=np.float16, shape=shape)
                    for key, shape in [('C', (count, 37, 128)), ('T', (count, 24, 160))]}
        labels = {key: np.stack([s.ix[key][index] for s, index in items]) for key in D.LABEL_KEYS}
        np.savez(directory / 'labels.npz', **labels)
        dump(directory / 'row_manifest.json', dict(ids=ids, row_identity_sha256=row_hash))
        order = np.argsort(widths, kind='stable')
        batches, current = [], []
        for index in order:
            if current and (len(current) >= 128 or (len(current)+1)*(int(widths[index])+D.MARGIN) > D.PIXEL_BUDGET):
                batches.append(current)
                current = []
            current.append(int(index))
        if current:
            batches.append(current)
        completed, last = 0, time.perf_counter()
        for indices in batches:
            batch = D.collate([items[index] for index in indices])
            images = torch.from_numpy(batch['image']).cuda().float()/255
            cols = torch.from_numpy(batch['cols']).cuda()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                values = model(images, cols)
            for key in features:
                features[key][indices] = values[key].float().cpu().numpy().astype(np.float16)
            completed += len(indices)
            if time.perf_counter()-last >= 30 or completed == count:
                status(out, 'extracting', split=split, completed=completed, total=count,
                       seconds=round(time.perf_counter()-start, 1))
                last = time.perf_counter()
        for value in features.values():
            value.flush()
        metadata[split] = dict(count=count, row_identity_sha256=row_hash, checkpoint_sha256=EXPECTED,
            shapes={key: list(value.shape) for key, value in features.items()},
            feature_hashes={key: sha(directory/f'{key}.npy') for key in features})
        dump(done, metadata[split])
        del features, labels
        gc.collect()
    del model
    torch.cuda.empty_cache()
    dump(out / 'feature_manifest.json', dict(splits=metadata, frozen=True, routing='Effects: own CNN effect query; text/font/length: own transformer query'))


def probability(model, values):
    return model.inplace_predict(np.asarray(values, dtype=np.float32, order='C'))


def predict_task(model, task, values):
    if 'constant' in task:
        return np.full(len(values), task['constant'], dtype=np.float32)
    prediction = np.asarray(probability(model, values))
    if task['kind'] == 'multiclass':
        index = (prediction >= .5).astype(int) if prediction.ndim == 1 else prediction.argmax(1)
        return np.asarray(task['classes'])[index]
    if task['kind'] == 'regression':
        return np.rint(np.clip(prediction, 0, 10)).astype(int)
    return prediction


class Progress(xgb.callback.TrainingCallback):
    def __init__(self, out, task, number):
        self.out, self.task, self.number = out, task, number
        self.last = time.perf_counter()

    def after_iteration(self, model, epoch, evals_log):
        if time.perf_counter()-self.last >= 30:
            status(self.out, 'fitting_xgboost', completed=self.number, total=107,
                   head=self.task['name'], boosting_round=epoch+1, maximum_rounds=ROUNDS)
            self.last = time.perf_counter()
        return False


def fit(out, reuse=None):
    features = {split: {key: np.load(out/'features'/split/f'{key}.npy', mmap_mode='r') for key in ('C', 'T')}
                for split in ('train', 'selection', 'calibration')}
    labels = {split: dict(np.load(out/'features'/split/'labels.npz')) for split in features}
    specs = []
    start = time.perf_counter()
    for number, task in enumerate(tasks()):
        metadata = out / 'head_specs' / (task['name']+'.json')
        if metadata.exists():
            entry = read(metadata)
            if 'model_file' in entry:
                assert entry['model_sha256'] == sha(out/'bundle/boosters'/entry['model_file'])
            specs.append(entry)
            continue
        prior_path = reuse/'head_specs'/(task['name']+'.json') if reuse else None
        prior = read(prior_path) if prior_path and prior_path.exists() else None
        if prior and task['kind'] != 'regression':
            entry = dict(prior, reused_fit_from=str(reuse))
            if 'model_file' in entry:
                source = reuse/'bundle/boosters'/entry['model_file']
                assert sha(source) == entry['model_sha256']
                shutil.copy2(source, out/'bundle/boosters'/entry['model_file'])
            dump(metadata, entry)
            specs.append(entry)
            status(out, 'head_reused', completed=number+1, total=107, head=entry['name'],
                   selection_score=entry['selection_score'])
            continue
        status(out, 'fitting_xgboost', completed=number, total=107, head=task['name'])
        task = task.copy()
        y, valid = target(labels['train'], task)
        ys, selection_valid = target(labels['selection'], task)
        assert valid.any() and selection_valid.any(), task['name']
        classes = np.unique(y[valid])
        task.update(classes=classes.tolist(), labelled_train=int(valid.sum()),
                    labelled_selection=int(selection_valid.sum()),
                    label_counts={str(int(value)): int(np.sum(y[valid] == value)) for value in classes})
        if len(classes) == 1:
            task['constant'] = float(classes[0])
            task['threshold'] = .5
            model = None
        else:
            xtrain = matrix(features['train'], task, valid)
            xselection = matrix(features['selection'], task, selection_valid)
            parameters = dict(PARAMS)
            if task['kind'] == 'regression':
                parameters.update(objective='reg:squarederror', eval_metric='mae')
                bins = np.bincount(y[valid].astype(int), minlength=11)
                weights = len(y[valid]) / (np.count_nonzero(bins)*bins[y[valid].astype(int)])
                train_target, selection_target = y[valid], ys[selection_valid]
            else:
                assert np.isin(ys[selection_valid], classes).all(), task['name']
                train_target = np.searchsorted(classes, y[valid])
                selection_target = np.searchsorted(classes, ys[selection_valid])
                weights = None
                if len(classes) == 2:
                    parameters.update(objective='binary:logistic', eval_metric='logloss')
                else:
                    parameters.update(objective='multi:softprob', eval_metric='mlogloss', num_class=len(classes))
            dselection = xgb.DMatrix(xselection, label=selection_target)
            variants = ['balanced', 'unweighted'] if task['kind'] == 'regression' else ['default']
            candidates = []
            for variant in variants:
                candidate_file = out/'head_candidates'/f'{task["name"]}_{variant}.json'
                status(out, 'fitting_xgboost', completed=number, total=107, head=task['name'], weighting=variant)
                if variant == 'balanced' and prior:
                    source = reuse/'bundle/boosters'/prior['model_file']
                    assert sha(source) == prior['model_sha256']
                    shutil.copy2(source, candidate_file)
                    candidate = xgb.Booster()
                    candidate.load_model(candidate_file)
                    candidate.set_param(dict(device='cpu'))
                    best_iteration = prior['best_iteration']
                else:
                    dtrain = xgb.DMatrix(xtrain, label=train_target, weight=weights if variant == 'balanced' else None)
                    candidate = xgb.train(parameters, dtrain, num_boost_round=ROUNDS, evals=[(dselection, 'selection')],
                        early_stopping_rounds=PATIENCE, callbacks=[Progress(out, task, number)], verbose_eval=False)
                    best_iteration = int(candidate.best_iteration)
                    candidate = candidate[:best_iteration+1]
                    candidate.set_param(dict(device='cpu'))
                    candidate.save_model(candidate_file)
                    del dtrain
                selected = predict_task(candidate, task, xselection)
                if task['kind'] == 'regression':
                    quality = float(np.mean(abs(selected-selection_target) <= 1))
                    mae = float(np.mean(abs(selected-selection_target)))
                else:
                    quality = M.f1(selected >= .5, selection_target == 1) if task['kind'] == 'binary' else float(np.mean(selected == ys[selection_valid]))
                    mae = 0.
                candidates.append(dict(weighting=variant, selection_score=quality, selection_mae=mae,
                    best_iteration=best_iteration, candidate_file=str(candidate_file), model_sha256=sha(candidate_file)))
                del candidate
            chosen = max(candidates, key=lambda value: (value['selection_score'], -value['selection_mae']))
            task.update(best_iteration=chosen['best_iteration'], weighting=chosen['weighting'], candidates=candidates)
            model = xgb.Booster()
            model.load_model(chosen['candidate_file'])
            model.set_param(dict(device='cpu'))
            task['model_file'] = task['name']+'.json'
            shutil.copy2(chosen['candidate_file'], out/'bundle/boosters'/task['model_file'])
            task['model_sha256'] = sha(out/'bundle/boosters'/task['model_file'])
            del xtrain, xselection, dselection
        if task['kind'] == 'binary':
            yc, calibration_valid = target(labels['calibration'], task)
            pc = predict_task(model, task, matrix(features['calibration'], task, calibration_valid))
            # Font italic uses .5, matching both saved neural comparators.
            task['threshold'] = .5 if task['group'] == 'font' else calibrate(pc, yc[calibration_valid])
        selected = predict_task(model, task, matrix(features['selection'], task, selection_valid))
        if task['kind'] == 'binary':
            task['selection_score'] = M.f1(selected >= .5, ys[selection_valid] == 1)
            task['selection_calibrated_f1'] = M.f1(selected >= task['threshold'], ys[selection_valid] == 1)
        elif task['kind'] == 'regression':
            task['selection_score'] = float(np.mean(abs(selected-ys[selection_valid]) <= 1))
            task['selection_mae'] = float(np.mean(abs(selected-ys[selection_valid])))
        else:
            task['selection_score'] = float(np.mean(selected == ys[selection_valid]))
        dump(metadata, task)
        specs.append(task)
        dump(out/'bundle/heads.json', dict(tasks=specs, params=PARAMS, maximum_rounds=ROUNDS,
              early_stopping_rounds=PATIENCE, all_neural_feature_heads_replaced=True))
        status(out, 'head_complete', completed=number+1, total=107, head=task['name'],
               selection_score=round(task['selection_score'], 5),
               best_iteration=task.get('best_iteration'), seconds=round(time.perf_counter()-start, 1))
        del model
        gc.collect()
    assert len(specs) == 107
    dump(out/'bundle/heads.json', dict(tasks=specs, params=PARAMS, maximum_rounds=ROUNDS,
              early_stopping_rounds=PATIENCE, all_neural_feature_heads_replaced=True))
    return specs


def load_models(out, specs, bundle=None):
    bundle = Path(bundle) if bundle is not None else out/'bundle'
    models = {}
    for task in specs:
        if 'constant' in task:
            continue
        path = bundle/'boosters'/task['model_file']
        assert sha(path) == task['model_sha256']
        model = xgb.Booster()
        model.load_model(path)
        model.set_param(dict(device='cpu', nthread=4))
        models[task['name']] = model
    return models


def boosted(features, specs, models):
    return {task['name']: predict_task(models.get(task['name']), task, matrix(features, task)) for task in specs}


def decode_font(values, index=0):
    font = {name: int(values['font_'+name][index]) for name in FONT_NAMES[:-1]}
    font.update(category=CATEGORIES[font['category']], stroke=STROKES[font['stroke']],
                italic=bool(values['font_italic'][index] >= .5))
    return font


def score_values(values, labels, specs):
    result = dict(effects={}, text_heads={}, font={}, reading={})
    for task in specs:
        y, valid = target(labels, task)
        predicted = values[task['name']][valid]
        truth = y[valid]
        group, name = task['group'], task['head']
        if group in ('eff', 'txt'):
            block = result['effects' if group == 'eff' else 'text_heads'].setdefault(name, {})
            if task['kind'] == 'binary':
                decision = predicted >= task['threshold']
                block.update(f1=M.f1(predicted >= .5, truth == 1),
                             calibrated_f1=M.f1(decision, truth == 1),
                             recall=float(np.sum(decision & (truth == 1))/max(1, np.sum(truth == 1))),
                             fpr=float(np.sum(decision & (truth == 0))/max(1, np.sum(truth == 0))),
                             threshold=task['threshold'], positive_n=int(np.sum(truth == 1)),
                             negative_n=int(np.sum(truth == 0)), unknown_n=int(np.sum(~valid)))
            else:
                block.update(lvl_exact=float(np.mean(predicted == truth)),
                             lvl_pm1=float(np.mean(abs(predicted-truth) <= 1)),
                             lvl_mae=float(np.mean(abs(predicted-truth))), lvl_n=int(valid.sum()))
        elif group == 'font':
            result['font'][name + '_pm1' if task['kind'] == 'regression' else name + '_f1' if task['kind'] == 'binary' else name] = (
                float(np.mean(abs(predicted-truth) <= 1)) if task['kind'] == 'regression' else
                M.f1(predicted >= .5, truth == 1) if task['kind'] == 'binary' else float(np.mean(predicted == truth)))
        else:
            result['reading']['length_head'] = float(np.mean(predicted == truth))
    result['summary'] = dict(
        eff_f1=float(np.mean([value['f1'] for value in result['effects'].values()])),
        txt_f1=float(np.mean([value['f1'] for value in result['text_heads'].values()])),
        eff_calibrated_f1=float(np.mean([value['calibrated_f1'] for value in result['effects'].values()])),
        txt_calibrated_f1=float(np.mean([value['calibrated_f1'] for value in result['text_heads'].values()])),
        eff_lvl_pm1=float(np.mean([value['lvl_pm1'] for value in result['effects'].values()])),
        txt_lvl_pm1=float(np.mean([value['lvl_pm1'] for value in result['text_heads'].values() if 'lvl_pm1' in value])),
        length=result['reading']['length_head'], font_cat=result['font']['category'],
        font_ord_pm1=float(np.mean([result['font'][name+'_pm1'] for name in FONT_NAMES[2:6]])))
    return result


def evaluate(out, specs, models):
    results = {}
    for split in ('selection', 'seen_test', 'generalisation'):
        path = out/'evaluation'/f'{split}.json'
        if path.exists():
            results[split] = read(path)
            continue
        features = {key: np.load(out/'features'/split/f'{key}.npy', mmap_mode='r') for key in ('C', 'T')}
        labels = dict(np.load(out/'features'/split/'labels.npz'))
        values = boosted(features, specs, models)
        results[split] = score_values(values, labels, specs)
        np.savez(out/'evaluation'/f'{split}_predictions.npz', **values)
        dump(path, results[split])
        status(out, 'synthetic_evaluation', split=split, summary=results[split]['summary'], font=results[split]['font'])
    dump(out/'synthetic_results.json', results)
    return results


def infer_real(out, specs, models):
    directory = out/'real_inference'
    directory.mkdir(exist_ok=True)
    if (directory/'report.json').exists():
        report = read(directory/'report.json')
        assert report['state'] == 'complete' and report['total'] == 3599
        return directory
    source = SPECIALISTS/'real_inference'
    manifest = read(source/'manifest.json')
    dump(directory/'manifest.json', manifest)
    # Select identities only; no old feature predictions, CTC strings or real
    # answer fields enter this extraction or the new family profiles.
    rows = [{key: value[key] for key in ('family', 'index', 'source')}
            for value in map(json.loads, (source/'predictions.jsonl').read_text().splitlines())]
    assert Counter(row['family'] for row in rows) == manifest['counts']
    encoder, _ = load_encoder(out/'bundle/encoder.pt', 'cuda')
    lookup = {task['name']: task for task in specs}
    results = []
    for begin in range(0, len(rows), 32):
        records = rows[begin:begin+32]
        images, columns = [], []
        for row in records:
            with Image.open(row['source']) as image:
                image = image.convert('RGB')
                width = max(8, min(960, round(image.width*96/image.height)))
                images.append(np.asarray(image.resize((width, 96), Image.Resampling.LANCZOS), dtype=np.uint8))
                columns.append((width+7)//8)
        columns = np.asarray(columns, dtype=np.int64)
        width = ((int(columns.max())*8 + D.MARGIN + 7)//8)*8
        pixels = np.stack([np.pad(image, ((0, 0), (0, width-image.shape[1]), (0, 0)), mode='edge') for image in images])
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            encoded = encoder(torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy()).cuda().float()/255,
                              torch.from_numpy(columns).cuda())
        features = {key: value.float().cpu().numpy().astype(np.float16) for key, value in encoded.items()}
        values = boosted(features, specs, models)
        for i, row in enumerate(records):
            result = dict(row, effects={}, text_features={})
            for group, names, block in [('eff', effect_heads(), 'effects'), ('txt', TEXT_HEADS, 'text_features')]:
                for name in names:
                    key = f'{group}_presence_{name}'
                    probability = float(values[key][i])
                    threshold = lookup[key]['threshold']
                    ordinal = values.get(f'{group}_level_{name}')
                    result[block][name] = dict(probability=probability, present=probability >= threshold,
                        level=int(ordinal[i]) if ordinal is not None else 0, threshold=threshold)
            result.update(font=decode_font(values, i), length=int(values['length'][i]))
            results.append(result)
        if len(results) % 512 < 32 or len(results) == len(rows):
            status(out, 'real_inference', completed=len(results), total=len(rows))
    assert len(results) == 3599
    with (directory/'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for row in results:
            stream.write(json.dumps(row)+'\n')
    dump(directory/'report.json', dict(state='complete', total=len(results), counts=manifest['counts'],
         source_checkpoint_sha256=EXPECTED, method='Pruned frozen feature encoder plus 107 XGBoost heads',
         neural_feature_heads_used=False, real_answers_used=False))
    del encoder
    torch.cuda.empty_cache()
    return directory


def previews(out, directory):
    from feature_bank.iter2.controller import build_profiles, generate_profiles
    if not (out/'profiles/index.json').exists():
        build_profiles(directory/'predictions.jsonl', directory/'manifest.json', out/'profiles')
    if not (out/'dataset/report.json').exists():
        status(out, 'generating_previews', count_per_family=100)
        try:
            generate_profiles(out/'profiles', out/'dataset', 100, 12, 42, 12)
        except RuntimeError:
            if not (out/'dataset/report.json').exists():
                raise
    report = read(out/'dataset/report.json')
    if report['stage'] != 'complete':
        repair = Path('C:/Users/pranav/Documents/ChatGPT/thesis/iteration2_specialists_work/launch/repair_previews.py')
        subprocess.run([sys.executable, str(repair), '--dataset', str(out/'dataset'), '--additional-attempts', '24'], check=True)
        report = read(out/'dataset/report.json')
    assert report['stage'] == 'complete' and not report['failures']
    assert all(value['generated'] == 100 and value['unique_pixels'] == 100 for value in report['audit'].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--reuse-fits-from', help='Reuse checked first-pass extraction/classification and balanced ordinal candidates')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ('bundle/boosters', 'head_specs', 'head_candidates', 'evaluation'):
        (out/name).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha(CHECKPOINT) == EXPECTED
    config = yaml.safe_load((BASE/'training_config.yaml').read_text())['model']
    protocol = dict(source_checkpoint=str(CHECKPOINT), source_checkpoint_sha256=EXPECTED,
        data=str(BASE), params=PARAMS, maximum_rounds=ROUNDS, early_stopping_rounds=PATIENCE,
        fitting='Synthetic train only; frozen learned head query features; no retraining of CNN/transformer',
        selection='Early stopping per head; regression compares exactly two weighting candidates by selection within-one, then lower selection MAE; first candidate wins exact ties. No neural fallback or benchmark tuning.',
        calibration='Effect/text presence: calibration max F1 with FPR <= .10. Font italic: .5 for comparable F1.',
        ordinals='Effect/text measured level >0 only; font labels >=0 including zero; balanced and unweighted regression candidates',
        removal='All neural effect/text/font/length prediction MLPs removed; CTC optional in separate sidecar',
        unsupported='Seven binary text attributes have no measured ordinal labels and receive no invented intensity model',
        real_training=False, solver_training=False, total_heads=107,
        reused_first_pass=str(Path(args.reuse_fits_from).resolve()) if args.reuse_fits_from else None)
    reuse = Path(args.reuse_fits_from).resolve() if args.reuse_fits_from else None
    if reuse:
        old = read(reuse/'protocol.json')
        assert old['source_checkpoint_sha256'] == EXPECTED and old['params'] == PARAMS
        assert not (reuse/'evaluation/seen_test.json').exists() and not (reuse/'evaluation/generalisation.json').exists(), 'First-pass benchmarks must be unscored'
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json') == protocol, 'Output protocol differs; use a fresh directory'
    else:
        dump(out/'protocol.json', protocol)
    datasets = {split: D.open_split(BASE, split) for split in SPLITS}
    assert {split: sum(map(len, value)) for split, value in datasets.items()} == dict(train=100000, selection=2000, calibration=2000, seen_test=2000, generalisation=2500)
    leak = leak_check(datasets['train'], [s for split in SPLITS[1:] for s in datasets[split]])
    dump(out/'leak_check.json', leak)
    export(out, config, datasets['train'])
    if args.export_only:
        status(out, 'export_verified')
        return
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.8)
    extract(out, datasets, config)
    specs = fit(out, reuse)
    models = load_models(out, specs)
    evaluate(out, specs, models)
    directory = infer_real(out, specs, models)
    previews(out, directory)
    assert sha(CHECKPOINT) == EXPECTED
    status(out, 'complete', heads=107, source_checkpoint_unchanged=True, real_images=3599, previews=300)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--out' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--out')+1]).resolve()
            if output.exists():
                status(output, 'failed', error=repr(error))
        raise
