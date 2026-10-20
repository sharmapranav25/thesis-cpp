from pathlib import Path
import shutil
here=Path(__file__).parent;root=Path('E:/pranav/Final sem/sourcefree_synth');dst=root/'runs/style_transformer_phase12_01'
assert not dst.exists()
shutil.copytree(here/'transformer_phase12_01',dst);shutil.copy2(here/'transformer_phase12.py',dst/'runner.py');shutil.copy2(dst/'REVIEW.md',root/'docs/V2_TRANSFORMER_REAL_PHASE12.md')
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (root/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\nTransformer Phase1/2 test complete by user request:6909real predictions,105matched renders, seed733. Same seven mapping heads as CNN, explicitly exploratory including failedheads; no joint parameter training. City/Courier falsegrids remain, no clear visual improvement. See V2_TRANSFORMER_REAL_PHASE12.md and runs/style_transformer_phase12_01. Production unchanged.\n')
print(dst)
