"""Publish only the new region demonstration, preserving prior runs."""
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import digest,read,write
src=Path(__file__).resolve().parents[1]
dst=Path('E:/pranav/Final sem/sourcefree_synth')
paths=['feature_bank/regions.py','scripts/demonstrate_regions.py','scripts/publish_regions.py',
       'tests/test_regions.py','docs/feature_bank/REGIONS_02.md']
paths += [p.relative_to(src).as_posix() for p in (src/'runs/feature_bank_regions_02').rglob('*') if p.is_file()]
for rel in paths:
    target=dst/rel
    if target.exists() and digest(target)!=digest(src/rel): raise FileExistsError(str(target))
protected=[]
for row in read(dst/'vendor/PROVENANCE.json')['files']:
    for p in (Path(row['original_path']),dst/'vendor/imported_repo'/row['relative_path']):
        if digest(p)!=row['sha256']: raise ValueError('Historical source mismatch')
        protected.append((p,row['sha256']))
plan=dst/'docs/FEATURE_BANK_CLASSIFICATION_PLAN.md'
before=digest(plan)
archive=dst/'docs/archive/FEATURE_BANK_CLASSIFICATION_PLAN_before_regions02.md'
if archive.exists(): raise FileExistsError(str(archive))
for rel in paths:
    target=dst/rel; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): shutil.copy2(src/rel,target)
archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(plan,archive)
with plan.open('a',encoding='utf-8') as f:
    f.write('\n\n## User clarification and region demonstration — 2026-09-28\n\nThe intended recognition path is local bricks → spatially connected feature evidence → library functions → later style approximation. The requested demonstration uses **10 pieces across and 3 down**, with overlapping crops and whole-image context retained. [Region demonstration and integrity protocol](feature_bank/REGIONS_02.md) · [Interactive explorer](../runs/feature_bank_regions_02/explore.html). This implements geometry only; no classifier, matching, or reconstruction gate has passed. Synthetic base groups must be split before cropping, and labels must describe visible evidence within each crop.\n')
for rel in paths:
    if digest(dst/rel)!=digest(src/rel): raise ValueError('Published hash mismatch')
for p,h in protected:
    if digest(p)!=h: raise ValueError('Historical file changed')
write(dst/'runs/feature_bank_regions_02/publication.json',dict(files_verified=len(paths),
    source_paths_unchanged=len(protected),plan_preimage_sha256=before,archive=str(archive)))
print('Published',len(paths),'files; preserved',len(protected),'original/vendor sources')
