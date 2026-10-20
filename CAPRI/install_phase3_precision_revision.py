"""Preserve failed run01 and install the tested bf16 comparison revision."""
from pathlib import Path
import shutil,json
base=Path(__file__).resolve().parent
project=Path('E:/pranav/Final sem/sourcefree_synth')
old=project/'runs/capri_phase3_city_01';new=project/'runs/capri_phase3_city_02';dest=project/'style_pipeline/phase3_reader_v2'
assert json.loads((old/'status.json').read_text())['state']=='failed'
assert not new.exists() and not dest.exists()
shutil.copytree(base/'phase3_reader',dest,ignore=shutil.ignore_patterns('__pycache__'))
new.mkdir();shutil.copytree(old/'data',new/'data');shutil.copy2(old/'upstream_provenance.json',new/'upstream_provenance.json')
note='''\n\n## City Phase3 precision revision02\nRun01 stopped on the first full CRNN batch: float16 scaled gradients were nonfinite. No success/checkpoint claimed; source and failure are preserved. Revision02 uses bfloat16 autocast with unscaled gradients and float32 CTC, consistently across all six candidates. Full128-image actual-data tests with three optimizer updates per architecture passed before launch. Also fixed local-module precedence in source search path, which the batch test exposed. Source style_pipeline/phase3_reader_v2; outputs runs/capri_phase3_city_02. All data/protocol controls otherwise unchanged; tenepochs each, no real labels. Actual status remains monitor/capri_phase3_city_status.json. Final dataset pending.\n'''
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md','THESIS_PAPERS.md','CAPRI_PHASE3_CITY_01.md']:
    with (project/'docs'/name).open('a',encoding='utf8') as f:f.write(note)
print('Installed preserved precision revision',new)
