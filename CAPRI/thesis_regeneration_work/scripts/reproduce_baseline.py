"""Replay historical probes against the byte-identical vendored repository."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import legacy_probe as probe
import numpy as np
import PIL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--historical-root', type=Path, default=Path(r'E:\pranav\Final sem\imported_regeneration_test'))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original_repo = probe.REPO
    vendor = ROOT / 'vendor/imported_repo'
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != original_repo.resolve()]
    sys.path.insert(0, str(vendor))
    probe.REPO = vendor
    probe.SAVED = vendor / 'system1_synthetic_generator/system1_production/final_generators'
    manifest = json.loads((ROOT / 'vendor/PROVENANCE.json').read_text(encoding='utf-8'))
    before = {r['relative_path']: probe.digest(Path(r['original_path'])) for r in manifest['files']}
    expected = {r['relative_path']: r['sha256'] for r in manifest['files']}
    actual = {rel: probe.digest(vendor / rel) for rel in expected}
    if before != expected or actual != expected:
        raise RuntimeError('Source or vendor no longer matches the frozen provenance')
    started = time.perf_counter()
    for name in ('saved', 'universal'):
        (out / name).mkdir()
    probe.saved_generators(out / 'saved')
    probe.universal(out / 'universal', remap=True)
    rows = []
    for branch, historical in [('saved', 'cryptbb_saved_01'), ('universal', 'cryptbb_universal_portable_01')]:
        for current in sorted((out / branch).glob('*/result.json')):
            old_path = args.historical_root / historical / current.parent.name / 'result.json'
            old = json.loads(old_path.read_text(encoding='utf-8'))
            new = json.loads(current.read_text(encoding='utf-8'))
            a = [(r['label'], r['width'], r['height'], r['pixel_sha256']) for r in old['samples']]
            b = [(r['label'], r['width'], r['height'], r['pixel_sha256']) for r in new['samples']]
            rows.append({'case': current.parent.name, 'historical': str(old_path), 'samples': len(b),
                         'exact_match': a == b, 'label_mismatches': new.get('label_mismatches', 0),
                         'returned_degenerate': new.get('returned_degenerate', 0),
                         'schema_substitutions': new.get('samples_with_schema_substitution', None)})
    changed = [rel for rel in expected if probe.digest(vendor / rel) != expected[rel]
               or probe.digest(original_repo / rel) != expected[rel]]
    module_paths = {name: str(module.__file__) for name, module in sys.modules.items()
                    if (name.startswith('gen_gateonly') or name.startswith('experiments.gen_ablation'))
                    and getattr(module, '__file__', None)}
    isolated = all(vendor.resolve() in Path(p).resolve().parents for p in module_paths.values())
    result = {'status': 'PASS' if all(r['exact_match'] for r in rows) and not changed and isolated else 'FAIL',
              'scope': 'Exact historical reproduction, not family fidelity or readiness for training.',
              'cases': rows, 'total_samples': sum(r['samples'] for r in rows),
              'original_or_vendor_changed': changed, 'vendor_import_isolation': isolated,
              'module_paths': module_paths, 'elapsed_seconds': time.perf_counter()-started,
              'python': platform.python_version(), 'pillow': PIL.__version__, 'numpy': np.__version__,
              'vlm_calls': 0, 'training': False, 'source_manifest_sha256': probe.digest(ROOT / 'vendor/PROVENANCE.json')}
    probe.save_json(out / 'equivalence.json', result)
    lines = ['# Baseline reproduction', '', 'Status: **' + result['status'] + '**', '',
             result['scope'], '', '| Case | Samples | Exact pixels + labels + dimensions |', '|---|---:|---|']
    lines += [f"| {r['case']} | {r['samples']} | {r['exact_match']} |" for r in rows]
    lines += ['', f"Total: {result['total_samples']} images. VLM calls: 0. Training: none.",
              f"Runtime: {result['elapsed_seconds']:.2f} seconds. Python {result['python']}; Pillow {result['pillow']}; NumPy {result['numpy']}.",
              '', 'Source hashes and import locations: [equivalence.json](equivalence.json).',
              '', 'Visual review is a separate artifact; exact reproduction preserves any original visual weaknesses.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'samples': result['total_samples'], 'changed': changed}), flush=True)
    if result['status'] != 'PASS':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
