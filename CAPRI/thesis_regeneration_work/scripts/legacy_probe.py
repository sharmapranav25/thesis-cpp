"""Exercise the imported repository, without editing it or using sourcefree_synth."""
from __future__ import annotations

import argparse
import csv
import contextlib
import dataclasses
import hashlib
import importlib.util
import io
import json
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(r'E:\pranav\Final sem\CaptchaThesis')
SAVED = REPO / 'system1_synthetic_generator/system1_production/final_generators'
FONT_DIR = Path(r'E:\pranav\model 1\.venv\Lib\site-packages\matplotlib\mpl-data\fonts\ttf')
DEJAVU = FONT_DIR / 'DejaVuSans-Bold.ttf'
sys.path.insert(0, str(REPO))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixels(image):
    return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')


def source_receipt():
    paths = list((REPO / 'system1_synthetic_generator').rglob('*.py'))
    paths += list((REPO / 'experiments/gen_ablation').glob('*.py'))
    paths.append(REPO / 'system1_synthetic_generator/previously_produced_generators/gen_gateonly/.font_whitelist.json')
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(paths)}


def sheet(images, captions, path, title, columns=4):
    width, height = 270, 110
    out = Image.new('RGB', (width * columns, height * ((len(images) + columns - 1) // columns) + 35), '#eeeeee')
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), title, fill='black')
    for i, (image, caption) in enumerate(zip(images, captions)):
        x, y = (i % columns) * width + 8, (i // columns) * height + 35
        display = image.convert('RGB').copy()
        display.thumbnail((width - 16, 78))
        out.paste(display, (x, y))
        draw.text((x, y + 83), caption, fill='black')
    out.save(path)


def saved_generators(out):
    from experiments.gen_ablation.families import family_charset, FAMILY_LENGTH
    charset = family_charset('cryptbb')
    rng = random.Random(2026)
    labels = [''.join(rng.choice(charset) for _ in range(FAMILY_LENGTH['cryptbb'])) for _ in range(16)]
    results = []
    for cluster in range(3):
        source = SAVED / f'cluster_{cluster}_generator.py'
        spec = importlib.util.spec_from_file_location(f'imported_cluster_{cluster}', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_get_font = module.get_font
        for mode in ('as_is', 'same_font_local_path'):
            folder = out / f'cluster_{cluster}_{mode}'
            folder.mkdir()
            if mode == 'same_font_local_path':
                # Only the environment's path to the EXACT requested generic font changes.
                module.get_font = lambda m=module: ImageFont.truetype(str(DEJAVU), m.FONT_SIZE)
            else:
                module.get_font = original_get_font
            font = module.get_font()
            font_info = {'type': type(font).__name__, 'name': font.getname(),
                         'size': getattr(font, 'size', None), 'path': str(getattr(font, 'path', 'embedded'))}
            if len(font_info['path']) > 200:
                font_info['path'] = 'Pillow embedded default font'
            images, rows = [], []
            start = time.perf_counter()
            for i, label in enumerate(labels):
                seed = 10000 + i
                random.seed(seed)
                np.random.seed(seed)
                image = module.generate_captcha(label)
                image.save(folder / f'{i:03d}.png')
                with Image.open(folder / f'{i:03d}.png') as check:
                    check.verify()
                images.append(image)
                rows.append({'filename': f'{i:03d}.png', 'label': label, 'seed': seed,
                             'width': image.width, 'height': image.height, 'pixel_sha256': pixels(image)})
            with (folder / 'labels.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'label'])
                writer.writeheader()
                writer.writerows({k: r[k] for k in ['filename', 'label']} for r in rows)
            sheet(images, labels, folder / 'preview.png', f'Imported cluster {cluster}: {mode}')
            info = {'cluster': cluster, 'mode': mode, 'source': str(source), 'source_sha256': digest(source),
                    'font': font_info, 'elapsed_seconds': time.perf_counter() - start,
                    'count': len(rows), 'samples': rows, 'vlm_called': False,
                    'real_images_used_to_refit': False, 'source_files_modified': False}
            save_json(folder / 'result.json', info)
            results.append(info)
            print(f'Saved cluster {cluster} / {mode}: {len(rows)} images, font={font_info["name"]}, size={font_info["size"]}', flush=True)
        module.get_font = original_get_font
    save_json(out / 'saved_generators.json', results)


def universal(out, remap=False):
    from experiments.gen_ablation.generate_lmdb import generate_one, UniversalCaptchaGenerator
    from experiments.gen_ablation.families import family_charset, FAMILY_LENGTH
    import gen_gateonly.generator as gm
    charset = family_charset('cryptbb')
    results = []
    font_modes = ('whitelist_paths_mapped',) if remap else ('as_is', 'generic_fonts_configured')
    for font_mode in font_modes:
        for condition in ('c1_random', 'c2_curriculum'):
            folder = out / f'{condition}_{font_mode}'
            folder.mkdir()
            random.seed(1337)
            np.random.seed(1337)
            context = contextlib.nullcontext()
            mapping = []
            if remap:
                from fontTools.ttLib import TTFont
                whitelist = REPO / 'system1_synthetic_generator/previously_produced_generators/gen_gateonly/.font_whitelist.json'
                original_data = json.loads(whitelist.read_text())
                for existing in sorted(set(original_data['whitelist'])):
                    local = FONT_DIR / Path(existing).name
                    if local.is_file():
                        with TTFont(str(local), lazy=True) as font:
                            cmap = font.getBestCmap() or {}
                            if all(ord(c) in cmap for c in charset):
                                mapping.append({'original': existing, 'local': str(local), 'sha256': digest(local)})
                mapped_data = dict(original_data, whitelist=sorted({r['local'] for r in mapping}))
                original_open = open
                def redirected_open(file, *args, **kwargs):
                    if isinstance(file, (str, Path)) and Path(file).resolve() == whitelist.resolve():
                        return io.StringIO(json.dumps(mapped_data))
                    return original_open(file, *args, **kwargs)
                context = patch('builtins.open', redirected_open)
                save_json(folder / 'font_path_mapping.json', mapping)
            with context:
                gen = UniversalCaptchaGenerator(seed=1337, font_dirs=[str(FONT_DIR)] if font_mode != 'as_is' else None)
            current = {'sample': 0, 'attempts': []}
            render_history = {}
            capture = {}
            originals = {}

            # Observer wrappers copy stage outputs but preserve RNG, arguments and return values.
            for name, stage in [('generate_background', '01_background'), ('apply_background_noise', '02_background_noise'),
                                ('render_text_on_image', '03_text'), ('apply_foreground_noise', '04_foreground_noise'),
                                ('apply_post_processing', '05_post_processing')]:
                original = getattr(gm, name)
                originals[name] = original
                def wrapped(*args, _original=original, _stage=stage, **kwargs):
                    result = _original(*args, **kwargs)
                    image = result[0] if isinstance(result, tuple) else result
                    capture[_stage] = image.copy()
                    return result
                setattr(gm, name, wrapped)

            render_pipeline = gen._render_pipeline
            generate = gen.generate
            def observed_pipeline(schema, text):
                capture.clear()
                result = render_pipeline(schema, text)
                entry = {'sample': current['sample'], 'label': text,
                         'schema': dataclasses.asdict(schema),
                         'degenerate': gm._is_degenerate(result[2]),
                         'quality_meta': result[2], 'stages': dict(capture), 'pixel_sha256': pixels(result[0])}
                current['attempts'].append(entry)
                render_history[entry['pixel_sha256']] = entry
                return result
            gen._render_pipeline = observed_pipeline
            def observed_generate(*args, **kwargs):
                current['requested_label'] = kwargs.get('text')
                current['requested_schema'] = dataclasses.asdict(kwargs['schema_obj'])
                result = generate(*args, **kwargs)
                current['internal_returned_label'] = result[1]
                return result
            gen.generate = observed_generate
            images, rows, errors = [], [], []
            count = 8 if font_mode == 'as_is' else 16
            started = time.perf_counter()
            try:
                for i in range(count):
                    current.update(sample=i, attempts=[])
                    try:
                        image, exported_label = generate_one(condition, charset, gen=gen, length=FAMILY_LENGTH['cryptbb'])
                        chosen = render_history.get(pixels(image))
                        filename = f'{i:03d}.png'
                        image.save(folder / filename)
                        with Image.open(folder / filename) as check:
                            check.verify()
                        row = {'filename': filename, 'label': exported_label,
                               'internal_returned_label': current['internal_returned_label'],
                               'label_alignment': exported_label == current['internal_returned_label'],
                               'attempts': len(current['attempts']), 'width': image.width, 'height': image.height,
                               'returned_from_sample': chosen['sample'] if chosen else None,
                               'returned_degenerate': chosen['degenerate'] if chosen else None,
                               'retry_changed_schema': any(e['schema'] != current['requested_schema'] for e in current['attempts'][1:]),
                               'pixel_sha256': pixels(image)}
                        rows.append(row)
                        images.append(image)
                        if i < 2 and chosen:
                            frames = list(chosen['stages'].values())
                            captions = list(chosen['stages'])
                            sheet(frames, captions, folder / f'{i:03d}_process.png',
                                  f'Imported {condition}, returned label {exported_label}; {len(current["attempts"])} attempts', columns=3)
                        if not row['label_alignment']:
                            print(f'LABEL MISMATCH: {condition} sample {i}: exported={exported_label}, actual={current["internal_returned_label"]}', flush=True)
                    except Exception as exc:
                        errors.append({'sample': i, 'error': type(exc).__name__ + ': ' + str(exc)})
                        print(f'{condition}/{font_mode} sample {i} failed: {exc}', flush=True)
                    print(f'{condition}/{font_mode}: sample {i+1}/{count}; attempts={len(current["attempts"])}', flush=True)
            finally:
                for name, original in originals.items():
                    setattr(gm, name, original)
            if images:
                sheet(images, [r['label'] for r in rows], folder / 'preview.png', f'Imported {condition}: {font_mode}')
            with (folder / 'labels.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'label'])
                writer.writeheader()
                writer.writerows({k: r[k] for k in ['filename', 'label']} for r in rows)
            result = {'condition': condition, 'font_mode': font_mode, 'font_dirs': gen.font_manager.font_dirs,
                      'discovered_fonts': len(gen.font_manager._font_catalog), 'elapsed_seconds': time.perf_counter()-started,
                      'requested': count, 'written': len(rows), 'errors': errors, 'samples': rows,
                      'label_mismatches': sum(not r['label_alignment'] for r in rows),
                      'returned_degenerate': sum(r['returned_degenerate'] is True for r in rows),
                      'samples_with_schema_substitution': sum(r['retry_changed_schema'] for r in rows),
                      'source_files_modified': False, 'vlm_called': False, 'real_images_used_to_refit': False,
                      'font_whitelist_paths_remapped_in_memory': remap,
                      'source': str(REPO/'experiments/gen_ablation/generate_lmdb.py')}
            save_json(folder / 'result.json', result)
            results.append(result)
            print(f'SUMMARY {condition}/{font_mode}: {len(rows)} images; {result["discovered_fonts"]} fonts; '
                  f'{result["label_mismatches"]} label mismatches; {result["returned_degenerate"]} degenerate returns', flush=True)
    save_json(out / 'universal_generators.json', results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--phase', choices=['saved', 'universal', 'universal_portable'], required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    if REPO == out or REPO in out.parents:
        raise ValueError('Outputs must be outside the imported repository')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    before = source_receipt()
    save_json(out / 'source_hashes_before.json', before)
    try:
        if args.phase == 'saved':
            saved_generators(out)
        else:
            universal(out, remap=args.phase == 'universal_portable')
    finally:
        after = source_receipt()
        changed = [p for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
        save_json(out / 'source_integrity.json', {'files_checked': len(before), 'changed_files': changed})
        print('Imported source integrity:',len(before),'Python files checked; changed:',changed,flush=True)
        if changed:
            raise RuntimeError('Imported source changed during test')


if __name__ == '__main__':
    main()
