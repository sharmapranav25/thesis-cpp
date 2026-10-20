"""Audit Method 1 across all four datasets; no target labels or VLM calls."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageStat
from sourcefree_synth.data import prepare_split, load_image, write_json, sha256_file, inventory
from sourcefree_synth.pixel_statistics import measure, aggregate, intersection, sample_canvases

def panel(rows, path, title):
    cell_w, cell_h = 360, 176
    canvas = Image.new('RGB', (cell_w*3, 42 + cell_h*((len(rows)+2)//3)), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill='black')
    for i, row in enumerate(rows):
        x, y = (i % 3)*cell_w+10, (i//3)*cell_h+42
        im = load_image(row['path'])
        im.thumbnail((340, 88))
        canvas.paste(im, (x, y))
        draw.text((x, y+91), f"{row['pixel_sha256'][:10]}  {row['width']}x{row['height']}  Y={row['brightness_mean']:.1f}", fill='black')
        start = x
        for color in row['palette']:
            width = int(round(330*color['fraction']))
            if width:
                draw.rectangle((start, y+112, start+width-1, y+135), fill=tuple(round(v) for v in color['rgb']))
                start += width
        if start < x+330:
            draw.rectangle((start, y+112, x+329, y+135), fill='#bbbbbb', outline='black')
        draw.text((x, y+140), f"Top 6 bins cover {100*row['palette_covered_fraction']:.1f}%; gray = remainder", fill='black')
    canvas.save(path)

def run_dataset(name, config, folder):
    folder.mkdir()
    start = time.perf_counter()
    write_json(folder / 'status.json', {'status': 'RUNNING'})
    all_inputs = inventory(config['input'])
    write_json(folder / 'input_receipts.json', all_inputs)
    split = prepare_split(config['input'], config['exclude'], seed=42)
    write_json(folder / 'split.json', split)
    measured, matrices = {}, {}
    checks = {'finite': True, 'histogram_mass': True, 'ranges': True, 'dimensions': True}
    crosscheck = []
    for subset in ('fit', 'validation'):
        measured[subset], matrices[subset] = [], []
        selected = {r['pixel_sha256'] for r in sorted(split[subset], key=lambda r: r['pixel_sha256'])[:8]}
        for i, record in enumerate(split[subset]):
            image = load_image(record['path'])
            row, hist = measure(image)
            numeric = np.array(row['rgb_mean']+row['rgb_std']+row['brightness_quantiles']+
                               [row['brightness_std'], row['color_entropy_bits']])
            checks['finite'] &= bool(np.isfinite(numeric).all() and np.isfinite(hist).all())
            checks['histogram_mass'] &= abs(row['histogram_sum']-1) <= 1e-10
            checks['ranges'] &= bool((hist >= 0).all() and (hist <= 1).all() and
                                     0 <= row['palette_covered_fraction'] <= 1+1e-10 and
                                     all(0 <= c <= 255 for c in row['rgb_mean']))
            checks['dimensions'] &= (row['width'], row['height']) == (record['width'], record['height'])
            if record['pixel_sha256'] in selected:
                independent = ImageStat.Stat(image)
                error = max(float(np.max(np.abs(np.array(row['rgb_mean'])-independent.mean))),
                            float(np.max(np.abs(np.array(row['rgb_std'])-independent.stddev))))
                crosscheck.append({'pixel_sha256': record['pixel_sha256'], 'subset': subset, 'max_abs_error': error})
            measured[subset].append(dict(record, **row, subset=subset))
            matrices[subset].append(hist)
            if (i+1) % 500 == 0:
                print(f'{name} {subset}: {i+1}/{len(split[subset])}', flush=True)
        matrices[subset] = np.stack(matrices[subset])
        with (folder / f'{subset}_measurements.jsonl').open('w', encoding='utf-8') as stream:
            for row in measured[subset]:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        np.savez_compressed(folder / f'{subset}_histograms.npz',
                            histogram=matrices[subset],
                            pixel_sha256=np.array([r['pixel_sha256'] for r in measured[subset]]))
        if subset == 'fit':
            frozen = aggregate(measured['fit'], matrices['fit'])
            write_json(folder / 'frozen_fit_profile.json', frozen)
            frozen_hash = sha256_file(folder / 'frozen_fit_profile.json')
    validation = aggregate(measured['validation'], matrices['validation'])
    write_json(folder / 'validation_profile.json', validation)
    fit_ids = {r['pixel_sha256'] for r in split['fit']}
    val_ids = {r['pixel_sha256'] for r in split['validation']}
    checks['split_disjoint'] = not bool(fit_ids & val_ids)
    checks['independent_pillow_agreement'] = len(crosscheck) == 16 and max(r['max_abs_error'] for r in crosscheck) <= 1e-6
    checks['fit_profile_unchanged_after_validation'] = sha256_file(folder / 'frozen_fit_profile.json') == frozen_hash
    fit_dims = {(m['width'], m['height']) for m in frozen['dimensions']}
    val_dims = {(m['width'], m['height']) for m in validation['dimensions']}
    sampled = sample_canvases(frozen, 64, 1337)
    checks['canvas_sampler_uses_only_observed_joint_modes'] = set(sampled).issubset(fit_dims)
    write_json(folder / 'sampled_canvases.json', {'seed': 1337, 'canvases': sampled, 'source': 'frozen_fit_profile.json', 'classification': 'MEASURED empirical distribution sampled with a fixed RNG seed'})
    ordered = sorted(measured['fit'], key=lambda r: r['pixel_sha256'])
    examples = [ordered[i] for i in np.linspace(0, len(ordered)-1, 9).astype(int)]
    examples += [min(ordered, key=lambda r: r['brightness_mean']), max(ordered, key=lambda r: r['brightness_mean']), max(ordered, key=lambda r: r['color_entropy_bits'])]
    val_order = sorted(measured['validation'], key=lambda r: r['pixel_sha256'])
    val_examples = [val_order[i] for i in np.linspace(0, len(val_order)-1, 9).astype(int)]
    panel(examples, folder / 'fit_review.png', name + ': fitting images and whole-image palette mass')
    panel(val_examples, folder / 'validation_review.png', name + ': held-out validation images and palette mass')
    write_json(folder / 'visual_selection.json', {'fit': [r['pixel_sha256'] for r in examples], 'validation': [r['pixel_sha256'] for r in val_examples], 'policy': '9 evenly spaced hash-sorted fit examples + darkest/brightest/max-entropy fit cases; 9 evenly spaced validation cases. Duplicates possible if an extreme is also selected.'})
    result = {'dataset': name, 'numerical_gate': 'PASS' if all(checks.values()) else 'FAIL',
              'visual_gate': 'PENDING', 'stage_gate': 'PENDING',
              'scope': 'Whole-image pixel measurements; not full family regeneration.',
              'checks': checks, 'independent_checks': crosscheck,
              'input_files': split['input_count'], 'unique_images': split['unique_count'],
              'duplicates_removed': split['duplicates_removed'], 'fit_count': len(fit_ids), 'validation_count': len(val_ids),
              'fit_dimensions': frozen['dimensions'], 'validation_dimensions': validation['dimensions'],
              'unseen_validation_dimensions': sorted(val_dims-fit_dims),
              'fit_validation_histogram_intersection': intersection(frozen['mean_rgb_histogram'], validation['mean_rgb_histogram']),
              'fit_validation_mean_rgb_absolute_shift': np.abs(np.array(frozen['mean_image_rgb_mean'])-validation['mean_image_rgb_mean']).tolist(),
              'mean_fit_brightness': frozen['mean_image_brightness'],
              'frozen_fit_profile_sha256': frozen_hash, 'elapsed_seconds': time.perf_counter()-start,
              'labels_read': False, 'images_quantitatively_sampled': False,
              'pixel_equality_exclusion_only': True, 'vlm_runtime_calls': 0}
    write_json(folder / 'result.json', result)
    write_json(folder / 'status.json', {'status': 'MEASUREMENTS_COMPLETE_VISUAL_REVIEW_PENDING' if result['numerical_gate']=='PASS' else 'FAILED'})
    lines = [f'# Method 1 — {name}', '', f"Numerical gate: **{result['numerical_gate']}**; visual gate: **PENDING**.", '',
             f"Files {result['input_files']}; unique images {result['unique_images']}; fit {len(fit_ids)}; validation {len(val_ids)}.",
             f"All input images were inspected quantitatively; {result['duplicates_removed']} exact pixel duplicates excluded from weighting.",
             '', f"Histogram intersection (descriptive): {result['fit_validation_histogram_intersection']:.6f}.",
             f"Runtime: {result['elapsed_seconds']:.2f} seconds.", '',
             '## Evidence', '', '- [Per-input hashes](input_receipts.json)', '- [Split](split.json)',
             '- [Frozen fitting profile](frozen_fit_profile.json)', '- [Validation profile](validation_profile.json)',
             '- [Checks and diagnostics](result.json)', '- [Canvas samples from measured dimensions](sampled_canvases.json)',
             '- [Visual sample selection](visual_selection.json)', '',
             '![Fit samples and palettes](fit_review.png)', '', '![Validation samples and palettes](validation_review.png)', '',
             'No alphabet, text length, font, foreground mask or background role is inferred. These panels are samples, not a full visual audit.']
    (folder / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'dataset': name, 'gate': result['numerical_gate'], 'files': result['input_files'], 'unique': result['unique_images'], 'intersection': result['fit_validation_histogram_intersection']}), flush=True)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.baseline / 'equivalence.json').read_text(encoding='utf-8'))
    if baseline['status'] != 'PASS':
        raise RuntimeError('Baseline equivalence gate has not passed')
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'configs/datasets.json').read_text(encoding='utf-8'))
    write_json(args.out / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__,
               'script_sha256': sha256_file(Path(__file__)), 'method_sha256': sha256_file(ROOT / 'sourcefree_synth/pixel_statistics.py'),
               'data_module_sha256': sha256_file(ROOT / 'sourcefree_synth/data.py'),
               'config_sha256': sha256_file(ROOT / 'configs/datasets.json'), 'protocol_sha256': sha256_file(ROOT / 'docs/VALIDATION.md'),
               'baseline_receipt_sha256': sha256_file(args.baseline / 'equivalence.json'), 'seed': 42, 'device': 'CPU'})
    results = []
    for name, paths in config.items():
        try:
            results.append(run_dataset(name, paths, args.out / name))
        except Exception as exc:
            folder = args.out / name
            folder.mkdir(exist_ok=True)
            record = {'dataset': name, 'numerical_gate': 'FAIL', 'stage_gate': 'FAIL', 'error': repr(exc)}
            write_json(folder / 'failure.json', record)
            results.append(record)
            print(json.dumps(record), flush=True)
    write_json(args.out / 'summary.json', {'method': 1, 'stage_gate': 'PENDING_VISUAL_REVIEW', 'datasets': results})
    if any(r['numerical_gate'] != 'PASS' for r in results):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
