from pathlib import Path
import shutil,subprocess,json
root=Path('E:/pranav/Final sem/sourcefree_synth');src=Path(__file__).parent/'multilabel_v1';dst=root/'feature_multilabel_v1'
assert not dst.exists(),'Preserve existing experiments'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','run'))
shutil.copy2(src/'PLAN.md',root/'docs/V2_MULTILABEL_PLAN.md')
note='\n\nUser shifts focus to joint multi-label feature learning. New finite feature_multilabel_v1 run:15operational features, raw RGB, multiscale1x1/2x2/5x3/10x3 shared feature maps, independent feature attention/heads. Synthetic-only13400image corpus, counterfactual visibility masks for label validity, heldout fonts/combinations/recipes,3seeds, early stopping and per-feature gates. See V2_MULTILABEL_PLAN.md. No success/promotion claimed; old specialists and release preserved. Dot specialist completed but failed strict synthetic gate on cluster layout/rasterizer; do not call prior highAP a robustness pass. Automation remains paused.\n'
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (root/'docs'/name).open('a',encoding='utf8') as f:f.write(note)
with (root/'docs/THESIS_PAPERS.md').open('a',encoding='utf8') as f:f.write('\n\nMulti-label v1: no new primary paper directly consulted or reproduced. Standard shared CNN/BCE supervision/attention pooling used as original baseline; exact architecture and limits in docs/V2_MULTILABEL_PLAN.md and feature_multilabel_v1/train.py. Results pending; no fabricated paper attribution.\n')
run=dst/'run';run.mkdir()
dashboard=root/'monitor/specialists.html'
with dashboard.open('a',encoding='utf8') as f:f.write('\n<p><a href="multilabel_status.json">Multi-label training: latest status</a></p>')
with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
 p=subprocess.Popen(['E:/pranav/model 1/.venv/Scripts/python.exe',str(dst/'train.py')],cwd=dst,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
(run/'launch.json').write_text(json.dumps({'launcher_pid':p.pid}));print('Launched finite multi-label run',p.pid,str(run))
