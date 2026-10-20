"""Publish a reviewed candidate additively; archive changed code and master plan."""
import argparse
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,digest
src=Path(__file__).resolve().parents[1];dst=Path('E:/pranav/Final sem/sourcefree_synth')
parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
paths=['feature_bank/labeled_bricks.py','scripts/make_labeled_brick_pilot.py','tests/test_labeled_bricks.py',
 'scripts/build_brick_corpus.py','scripts/audit_brick_corpus.py','scripts/publish_corpus_candidate.py',
 'docs/feature_bank/CORPUS_CANDIDATE_01.md','docs/feature_bank/FEATURE_PARAMETERS.md']
for name in ['feature_bank_labeled_bricks_04','feature_bank_corpus_candidate_01']:
    paths += [p.relative_to(src).as_posix() for p in (src/'runs'/name).rglob('*') if p.is_file()]
planrel='docs/FEATURE_BANK_CLASSIFICATION_PLAN.md'
preconditions=src/'corpus_publication_preconditions.json'
if args.prepare:
    write(preconditions,{rel:digest(dst/rel) if (dst/rel).exists() else None for rel in paths+[planrel]})
    print('Prepared publication preconditions:',len(paths),'files');sys.exit(0)
expected=read(preconditions)
for rel in paths+[planrel]:
    actual=digest(dst/rel) if (dst/rel).exists() else None
    if actual!=expected[rel]:raise ValueError('Destination changed: '+rel)
protected={}
for r in read(dst/'vendor/PROVENANCE.json')['files']:
    for p in (Path(r['original_path']),dst/'vendor/imported_repo'/r['relative_path']):
        if digest(p)!=r['sha256']:raise ValueError('Historical source hash mismatch')
        protected[p]=r['sha256']
archive=dst/'docs/archive/before_corpus_candidate_01'
if archive.exists():raise FileExistsError(archive)
for rel in paths+[planrel]:
    if expected[rel] is not None:
        backup=archive/rel;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst/rel,backup)
print('Publishing',len(paths),'files',flush=True)
for rel in paths:
    target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/rel,target)
with (dst/planrel).open('a',encoding='utf-8') as stream:
    stream.write('\n\n## Candidate corpus and parameter proposal — 2026-09-28\n\n[Label repair and expanded corpus](feature_bank/CORPUS_CANDIDATE_01.md): 5,600 synthetic parents, 168,000 bricks, 119,292 retained unique crops; grouped partitions, withheld font/combination, two quality rejections preserved. Integrity/replay checks passed; corpus acceptance remains partial and no training started. [Density and tilt after classification](feature_bank/FEATURE_PARAMETERS.md) records the user-proposed extension, its feature-specific definitions and validation gates; estimators are not yet implemented.\n')
for rel in paths:
    if digest(dst/rel)!=digest(src/rel):raise ValueError('Publication hash mismatch: '+rel)
for p,h in protected.items():
    if digest(p)!=h:raise ValueError('Historical source changed')
write(dst/'runs/feature_bank_corpus_candidate_01/publication.json',dict(files_verified=len(paths),historical_source_paths_unchanged=len(protected),
    archived_preimages=str(archive),plan_preimage_sha256=expected[planrel]))
print('Verified publication; original and vendor source hashes unchanged')
