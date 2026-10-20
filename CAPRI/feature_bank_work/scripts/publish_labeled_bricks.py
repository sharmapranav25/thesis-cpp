import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,digest
src=Path(__file__).resolve().parents[1];dst=Path('E:/pranav/Final sem/sourcefree_synth')
paths=['feature_bank/labeled_bricks.py','scripts/make_labeled_brick_pilot.py','scripts/publish_labeled_bricks.py',
       'tests/test_labeled_bricks.py','docs/feature_bank/LABELED_BRICKS_03.md']
for n in ('01','02','03'):
    paths += [p.relative_to(src).as_posix() for p in (src/f'runs/feature_bank_labeled_bricks_{n}').rglob('*') if p.is_file()]
for rel in paths:
    if (dst/rel).exists() and digest(dst/rel)!=digest(src/rel):raise FileExistsError(rel)
old={}
for r in read(dst/'vendor/PROVENANCE.json')['files']:
    for p in (Path(r['original_path']),dst/'vendor/imported_repo'/r['relative_path']):
        if digest(p)!=r['sha256']:raise ValueError('Original source hash mismatch')
        old[p]=r['sha256']
plan=dst/'docs/FEATURE_BANK_CLASSIFICATION_PLAN.md';archive=dst/'docs/archive/FEATURE_BANK_CLASSIFICATION_PLAN_before_labeled_bricks03.md'
if archive.exists():raise FileExistsError(archive)
print('Publishing',len(paths),'new files',flush=True)
for rel in paths:
    target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copy2(src/rel,target)
shutil.copy2(plan,archive)
with plan.open('a',encoding='utf-8') as f:
    f.write('\n\n## Local-label mechanism pilot — 2026-09-28\n\n[Small labeled-brick pilot and revisions](feature_bank/LABELED_BRICKS_03.md): 64 images / 1,920 local crops from eight synthetic base groups; straight lines, curves and dots have final contribution masks and conservative local labels. Effect-color shortcuts found in the first design were corrected before training. Only 1,139 unique crops; no training or full-corpus gate passed. Unknown features and ambiguous fragments remain unknown/uncertain. Earlier revisions are retained.\n')
for rel in paths:
    if digest(dst/rel)!=digest(src/rel):raise ValueError('Published hash mismatch')
for p,h in old.items():
    if digest(p)!=h:raise ValueError('Historical source changed')
write(dst/'runs/feature_bank_labeled_bricks_03/publication.json',dict(files_verified=len(paths),historical_source_paths_unchanged=len(old)))
print('Publication verified; historical source hashes unchanged')
