from pathlib import Path
import json,shutil
here=Path(__file__).parent;src=here/'stroke_grid_10x3_v2';root=Path('E:/pranav/Final sem/sourcefree_synth');dst=root/'feature_specialists_v2/strokes_grid_10x3_v2'
result=json.loads((src/'run/results.json').read_text())
assert not dst.exists(),'Preserve previous artifacts'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))
report=(src/'PLAN.md').read_text()+'\n\n## Completed outcome\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n\nNo production promotion. This trial changes both view and colour distribution; old/new scores are not an isolated grid comparison.\n'
(root/'docs/V2_STROKE_GRID_10X3.md').write_text(report)
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (root/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\nUser-requested10x3 raw RGB multicolour stroke retraining completed. Shared per-patch detector, stitched mask, no feature-pop, fresh synthetic seeds and font holdouts. See V2_STROKE_GRID_10X3.md for protocol/results. Failed diagnostic gate remains truthful; no production promotion or automatic loop. Prior full-dataset stroke inference exposed background/text mask contamination.\n')
print(report[-1800:])
