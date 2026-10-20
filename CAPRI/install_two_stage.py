from pathlib import Path
import shutil,subprocess,json
root=Path('E:/pranav/Final sem/sourcefree_synth');src=Path(__file__).parent/'stroke_two_stage';dst=root/'feature_specialists_v2/strokes_two_stage_v1'
assert not dst.exists(),'Preserve existing experiments'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))
plan='''# Two-stage strokes v1: controlled learnability trial

User authorized locating non-text strokes first, then classifying straight versus curved. This finite experiment trains a small dense detector with exact synthetic masks and a separate shape classifier with synthetic masks. End-to-end evaluation uses predicted masks, never target masks. Oracle-mask accuracy is separately labeled. Full-image 128x64 context is retained; no broad grid search until learnability established.

2400 train, 600 selection, 600 fresh locked examples, seeds101m/102m/103m; disjoint font groups (six train, seventh additionally selection, last two locked). Equal absent/straight/curve cases. Four clean detector epochs then twelve text-clutter epochs. Shape trains sixteen epochs; best checkpoints selected on selection only. No real fitting, target generators/fonts, VLM or production promotion.

Gate: locked mean positive mask IoU >=0.70, absent-image false positive rate <=0.05, end-to-end shape accuracy on positive images >=0.85. Mask threshold0.5 and minimum12pixels fixed before evaluation. Report localization, detection, oracle shape and predicted-mask shape separately. Passing this diagnostic does not establish robust acceptance.

Scope limits: zero or one target stroke, bright plain backgrounds and dark foreground, single quadratic curve recipe, no instance grouping or simultaneous multiple shapes, no texture/grid/occlusion robustness. These limitations are deliberate ablations, not a completed replacement pipeline. Shape transfer from clean true masks to predicted masks may fail and must be measured. No new paper consulted. Historical curve run and dots preserved. Dots may continue concurrently. Training logs and per-job status live in feature_specialists_v2/strokes_two_stage_v1/run; monitor/strokes_two_stage_status.json mirrors this job alone.
'''
(dst/'PLAN.md').write_text(plan);(root/'docs/V2_TWO_STAGE_STROKES.md').write_text(plan)
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (root/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\nStarted authorized finite two-stage stroke learnability trial: see V2_TWO_STAGE_STROKES.md. Reason: v1 curve classifier collapsed under whole-image supervision. Explicit masks teach localization; oracle versus predicted-mask shape tests isolate failure. Preserve dot worker, old runs, release, and paused automation. No robust success claimed.\n')
dashboard=root/'monitor/specialists.html';html=dashboard.read_text(encoding='utf8')
if 'strokes_two_stage_status.json' not in html:dashboard.write_text(html+'\n<p><a href="strokes_two_stage_status.json">Two-stage stroke experiment: live status</a></p>',encoding='utf8')
run=dst/'run';run.mkdir(exist_ok=True)
with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
 p=subprocess.Popen(['E:/pranav/model 1/.venv/Scripts/python.exe',str(dst/'train.py')],cwd=dst,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
(run/'launch.json').write_text(json.dumps({'launcher_pid':p.pid}));print('Started two-stage stroke trial launcher',p.pid)
