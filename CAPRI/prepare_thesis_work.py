"""Prepare a local, reviewable overlay; never writes to E: or imports vendor code."""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
STAGE = HERE / 'thesis_regeneration_work'
TARGET = Path(r'E:\pranav\Final sem\sourcefree_synth')
REPO = Path(r'E:\pranav\Final sem\CaptchaThesis')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    STAGE.mkdir(exist_ok=True)
    existing = {}
    for p in TARGET.rglob('*'):
        rel = p.relative_to(TARGET)
        if p.is_file() and not set(rel.parts) & {'runs', '__pycache__', '.git'}:
            dest = STAGE / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            existing[rel.as_posix()] = sha(p)
    for name in ('docs/experiments', 'docs/methods', 'docs/archive', 'scripts', 'configs'):
        (STAGE / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TARGET / 'README.md', STAGE / 'docs/archive/README_before_import.md')
    legacy = HERE / 'test_imported_regeneration.py'
    shutil.copyfile(legacy, STAGE / 'scripts/legacy_probe.py')
    vendor = STAGE / 'vendor/imported_repo'
    selected = list((REPO / 'system1_synthetic_generator/previously_produced_generators/gen_gateonly').glob('*.py'))
    selected += [REPO / 'system1_synthetic_generator/previously_produced_generators/gen_gateonly/.font_whitelist.json']
    selected += list((REPO / 'system1_synthetic_generator/system1_production/final_generators').glob('cluster_*_generator.py'))
    selected += [p for p in (REPO / 'experiments/__init__.py', REPO / 'experiments/gen_ablation/__init__.py') if p.is_file()]
    selected += [REPO / 'experiments/gen_ablation' / f for f in ('families.py', 'generate_lmdb.py')]
    selected += list((REPO / 'system5_clean_baseline/charsets').glob('charset_*.txt'))
    records = []
    for p in sorted(set(selected)):
        rel = p.relative_to(REPO)
        dest = vendor / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        records.append({'relative_path': rel.as_posix(), 'original_path': str(p), 'sha256': sha(p)})
    (STAGE / 'vendor/PROVENANCE.json').write_text(json.dumps({
        'source_root': str(REPO), 'files': records,
        'original_files_changed': False, 'vendor_modifications': [],
        'license_note': 'No LICENSE/NOTICE file was found by the initial filename search. Local research copy only; no redistribution permission inferred.',
        'legacy_probe_source': str(legacy), 'legacy_probe_sha256': sha(legacy)
    }, indent=2) + '\n', encoding='utf-8')
    (HERE / 'thesis_publish_state.json').write_text(json.dumps({'target': str(TARGET), 'known_existing': existing}, indent=2), encoding='utf-8')
    print(json.dumps({'stage': str(STAGE), 'vendor_files': len(records), 'existing_files_copied': len(existing)}))

if __name__ == '__main__':
    main()
