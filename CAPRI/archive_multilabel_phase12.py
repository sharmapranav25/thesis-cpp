from pathlib import Path
import shutil
here=Path(__file__).parent;root=Path('E:/pranav/Final sem/sourcefree_synth');dst=root/'runs/style_multilabel_phase12_01'
assert not dst.exists()
shutil.copytree(here/'multilabel_phase12_01',dst);shutil.copy2(here/'multilabel_phase12.py',dst/'runner.py');shutil.copy2(dst/'REVIEW.md',root/'docs/V2_MULTILABEL_REAL_PHASE12.md')
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (root/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\nMulti-label real Phase1/2 experiment complete:6909 real images inferred,105 renders across fivefamilies. See V2_MULTILABEL_REAL_PHASE12.md and runs/style_multilabel_phase12_01. CachedCV/font profiles plus fresh15head inference; only7passedheads map compatible effects, straight legacy settings remain. Darkbay falsegrid68.7%, outlined curve0.1% show transfer failures; no promotion. City dot0% promising qualitative correction. Original release integrity verified. All outputs and failed serialization recovery documented; no real fitting.\n')
print('Archived experiment',dst)
