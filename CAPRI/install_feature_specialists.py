"""Additive installation of the audited first feature-specialist package."""
from pathlib import Path
import json,shutil,hashlib
HERE=Path(__file__).parent;src=HERE/'feature_specialists';ROOT=Path('E:/pranav/Final sem/sourcefree_synth');dst=ROOT/'feature_specialists_v2'
assert not dst.exists(),'Refuse to overwrite any existing implementation'
prep=src/'runs/curves_v1';audit=json.loads((prep/'corpus_audit.json').read_text());assert audit['passed'] and audit['images']==20000
dst.mkdir();run=dst/'runs/curves_v1';run.mkdir(parents=True)
for name in ('PLAN.md','corpus.py','train_curves.py','test_curves.py'):shutil.copy2(src/name,dst/name)
for path in prep.iterdir():
    if path.suffix=='.npz' or path.name.endswith('_records.json') or path.name in ('fonts.json','corpus_audit.json','training_preview.png'):shutil.copy2(path,run/path.name)
for before,after in [('protocol.json','corpus_preparation_protocol.json'),('events.jsonl','corpus_preparation_events.jsonl'),('preparation_train_curves.py','corpus_preparation_runner.py')]:shutil.copy2(prep/before,run/after)
receipt={'canonical':str(dst),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.iterdir() if p.is_file()},'corpus':audit,'preparation_note':'Data generated under preserved preparation runner. Training runner adds progress reporting, resume grid normalization and final Markdown reporting; corpus.py unchanged.'}
(run/'installation.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
(ROOT/'docs/V2_FEATURE_SPECIALIST_PLAN.md').write_text((src/'PLAN.md').read_text(),encoding='utf8')
message='D084: User requests independent per-feature training/pop/view selection. Added feature_specialists_v2, starting CURVES ONLY; others queued. CorpusG0passed20000images/10000pairedgroups, no duplicatepixels, disjointseeds,12orientationbins,9auditedfonts. Raw versus localcontrastpop for8grids then top2grid combination; selected configuration retrained3seeds; calibration separate;4lockedstrata unused untilfinal. Strict syntheticgates predeclared, no automaticpromotion. No shared5classclassifier or rushedrobustnessclaim. Original/vendor/defaultrelease unchanged. Finiteworker launch next; continuousautomation remains paused. Read V2_FEATURE_SPECIALIST_PLAN.md.'
for name in ('STYLE_DECISIONS.md','STYLE_HANDOFF.md','V2_TIER1_PLAN.md'):
    with (ROOT/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\n'+message+'\n')
with (ROOT/'docs/THESIS_PAPERS.md').open('a',encoding='utf8') as f:f.write('\n\nIndependent curve-specialist reset: original paired synthetic-corpus design, deterministic local-contrast pop candidate, spatialCNN, per-grid/raw-pop search and grouped validation. No new external paper directly used or reproduction claimed. Uses standard PyTorch components, including GroupNorm; exact code/source hashes and limitations in feature_specialists_v2 and V2_FEATURE_SPECIALIST_PLAN.md. Results pending.\n')
print(json.dumps({'installed':str(dst),'audit':audit},indent=2))
