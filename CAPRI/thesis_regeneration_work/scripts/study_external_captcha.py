"""Decode only image members from the pinned public ZIP; test unchanged Method 1."""
import hashlib
import io
import json
import platform
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import PIL
from PIL import Image, ImageOps
from sourcefree_synth.data import write_json, sha256_file, inventory
from study_pixels import run_dataset

def main():
    package = ROOT / 'data/external/outlined_text_captcha'
    source = package / 'source'
    receipt = json.loads((source / 'download_receipt.json').read_text(encoding='utf-8-sig'))
    archive = source / 'train.zip'
    if sha256_file(archive) != receipt['sha256']:
        raise RuntimeError('Archive identity changed')
    if json.loads((ROOT / 'runs/pixel_statistics_01/gate.json').read_text())['status'] != 'PASS':
        raise RuntimeError('The existing Method 1 gate has not passed')
    images = package / 'images'
    images.mkdir(parents=True, exist_ok=False)
    members, skipped = [], []
    with zipfile.ZipFile(archive) as handle:
        entries = handle.infolist()
        if len(entries) > 20000 or sum(e.file_size for e in entries) > 200_000_000:
            raise RuntimeError('Unexpected archive expansion')
        for index, entry in enumerate(entries):
            suffix = Path(entry.filename).suffix.lower()
            if entry.is_dir() or suffix not in {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'}:
                skipped.append({'archive_index': index, 'suffix': suffix, 'reason': 'not an image member'})
                continue
            # Never extract archive paths. Never parse the filename for a text label.
            if entry.file_size > 10_000_000:
                raise RuntimeError('Unexpected image member size')
            raw = handle.read(entry)
            digest = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as original:
                original.load()
                raw_mode = original.mode
                oriented = ImageOps.exif_transpose(original)
                rgba = oriented.convert('RGBA')
                image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
            destination = images / f'{index:05d}_{digest[:16]}.png'
            image.save(destination)
            members.append({'archive_index': index, 'archive_image_sha256': digest,
                            'opaque_image_file': destination.name, 'decoded_size': list(image.size),
                            'original_mode': raw_mode, 'original_filename_recorded': False})
    if len(members) < 20:
        raise RuntimeError('Insufficient decoded images')
    write_json(package / 'extraction_receipt.json', {'download': receipt, 'image_members': len(members),
               'members': members, 'skipped': skipped, 'labels_read': False,
               'policy': 'Decode pixels only; save RGB PNG with opaque index and content hash; never execute archive content.'})
    print(json.dumps({'decoded_images': len(members), 'nonimage_members_skipped': len(skipped)}), flush=True)
    out = ROOT / 'runs/external_pixels_outlined_01'
    out.mkdir(exist_ok=False)
    # Exclude all four previous pools and declared external eval pools for an independence check.
    configs = json.loads((ROOT / 'configs/datasets.json').read_text(encoding='utf-8'))
    exclusions = sorted({p for c in configs.values() for p in [c['input']]+c['exclude']})
    write_json(out / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__,
               'method_sha256': sha256_file(ROOT / 'sourcefree_synth/pixel_statistics.py'),
               'study_runner_sha256': sha256_file(ROOT / 'scripts/study_pixels.py'),
               'external_runner_sha256': sha256_file(Path(__file__)),
               'download_receipt_sha256': sha256_file(source / 'download_receipt.json'),
               'seed': 42, 'device': 'CPU', 'external_exclusion_paths': exclusions,
               'method_settings_changed': False, 'dataset_selection': 'public convenience sample; entire downloaded train archive used'})
    try:
        result = run_dataset('outlined_text_captcha', {'input': str(images), 'exclude': exclusions}, out / 'outlined_text_captcha')
        write_json(out / 'summary.json', {'method': 1, 'study': 'external generalization probe',
                   'stage_gate': 'PENDING_VISUAL_REVIEW', 'datasets': [result], 'upstream_split': 'train only'})
        if result['numerical_gate'] != 'PASS':
            raise SystemExit(1)
    except Exception as exc:
        write_json(out / 'failure.json', {'error': repr(exc)})
        raise

if __name__ == '__main__':
    main()
