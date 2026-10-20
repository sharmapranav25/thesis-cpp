"""Install reviewed new Phase3 source/artifacts and append canonical records."""
from pathlib import Path
import hashlib,json,shutil
BASE=Path(__file__).resolve().parent
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
dest=PROJECT/'style_pipeline/phase3_reader'
run=PROJECT/'runs/capri_phase3_city_01'
assert not dest.exists() and not run.exists()
shutil.copytree(BASE/'phase3_reader',dest,ignore=shutil.ignore_patterns('__pycache__'))
run.mkdir();shutil.copytree(BASE/'phase3_city_data_01',run/'data')
shutil.copy2(dest/'PROTOCOL.md',PROJECT/'docs/CAPRI_PHASE3_CITY_01.md')
provenance=json.loads(Path('E:/pranav/Final sem/Models/level 1/PROVENANCE.json').read_text())
up=Path('E:/pranav/Final sem/Models/level 1/upstream')
files=[p for repo in ['parseq','OpenOCR'] for p in (up/repo).rglob('*.py')]
provenance['source_hashes']={str(p.relative_to(up)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
provenance['license_hashes']={str(p.relative_to(up)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [up/'parseq/LICENSE',up/'parseq/strhub/models/crnn/LICENSE',up/'OpenOCR/LICENSE']}
(run/'upstream_provenance.json').write_text(json.dumps(provenance,indent=2))
entry='''\n\n## Official Phase 3 — City scheme reader comparison01 (2026-09-29)\nCurrent user request supersedes count-only and optional-refinement phase definitions: Phase1 CV+frozen visual style, Phase2 broad typography/variable length, Phase3 synthetic-only OCR reader -> frozen fit-only scheme estimation -> targeted dataset. Historical optional refiners remain outside official Phase3. See CAPRI_PHASE3_CITY_01.md and runs/capri_phase3_city_01. Source style_pipeline/phase3_reader; original/vendor/datasets preserved. Full independent100k rehash and1599fit/400validation integrity passed; all case groups covered. Metadata count shortcut probe100% from count-dependent layout; NO shortcut-resistant count claim. Four GPU preflight tests passed after correcting nested Lightning device metadata in test. Planned six matched ten-epoch runs (CRNN/SVTR/PARSeq, plain/aux, seed731); no existing weights loaded. Real ground truth not independently verified: filenames never labels, no real accuracy claim. Actual live stage is runs/capri_phase3_city_01/status.json and monitor/capri_phase3_city_status.json. At installation, prepared; coordinator launch is separate. Final dataset still pending. Research automation not resumed.\n'''
for name in ['STYLE_HANDOFF.md','STYLE_DECISIONS.md']:
    with (PROJECT/'docs'/name).open('a',encoding='utf8') as f:f.write(entry)
plan=PROJECT/'docs/STYLE_PIPELINE_PLAN.md'
plan.write_text('# Current official Phase 3 — 2026-09-29\n\nThe current user request supersedes every older phase/count-only definition below. Phase1 extracts visual style with CV and frozen feature inference; Phase2 generates broad-font variable-length synthetic text; Phase3 trains synthetic-only readers, freezes them to estimate real FIT length/group distributions, then generates a separate targeted dataset. Historical optional image refiners are outside official Phase3 and must not be reintroduced. See CAPRI_PHASE3_CITY_01.md for the City protocol and runs/capri_phase3_city_01/status.json for actual state. No real transcription labels in training or scheme estimation; validation never informs the scheme.\n\n---\n\n'+plan.read_text(encoding='utf8'),encoding='utf8')
with (PROJECT/'docs/THESIS_PAPERS.md').open('a',encoding='utf8') as f:
    f.write('\n\n## Official Phase3 City reader comparison01 — implementation prepared\n'+(dest/'PROTOCOL.md').read_text(encoding='utf8').split('References and deviations:')[1])
(PROJECT/'monitor/phase3_city.html').write_text('''<!doctype html><meta charset="utf-8"><title>CAPRI official Phase3 City</title><style>body{font:16px system-ui;margin:32px;background:#111827;color:#e5e7eb}pre{white-space:pre-wrap}</style><h1>CAPRI official Phase3 — City</h1><p>Actual finite-job status. Predictions are not real accuracy.</p><pre id="s">Loading…</pre><script>async function poll(){try{let r=await fetch('capri_phase3_city_status.json?'+Date.now());document.getElementById('s').textContent=JSON.stringify(await r.json(),null,2)}catch(e){document.getElementById('s').textContent=String(e)}}poll();setInterval(poll,10000)</script>''',encoding='utf8')
print('Installed new Phase3 implementation and data audit:',run)
