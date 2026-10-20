from pathlib import Path
import shutil,subprocess,json,hashlib
root=Path('E:/pranav/Final sem/sourcefree_synth');src=Path(__file__).parent/'transformer_multilabel_v1';dst=root/'feature_transformer_multilabel_v1'
assert not dst.exists(),'Preserve experiments'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','run'));run=dst/'run';run.mkdir();hashes={}
for part in ['train','selection','calibration']:
 for ext in ['.npz','_records.json']:
  origin=root/'feature_multilabel_v1/run'/(part+ext);shutil.copy2(origin,run/origin.name);hashes[origin.name]=hashlib.sha256(origin.read_bytes()).hexdigest()
(run/'reused_data_hashes.json').write_text(json.dumps(hashes,indent=2));shutil.copy2(src/'PLAN.md',root/'docs/V2_TRANSFORMER_MULTILABEL_PLAN.md')
note='\n\nUser requests transformer instead of CNN. Launched finite feature_transformer_multilabel_v1: pure8x8patch transformer,15feature tokens, same train/selection/calibration corpus and optimizer/stopping rules,3seeds. Fresh304m..307m test images; frozenCNN replay on same fresh tests for comparison. Historical CNN preserved, no real fitting, no renderer-parameter extension yet. Read V2_TRANSFORMER_MULTILABEL_PLAN.md. No success/promotion claim; automation paused.\n'
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (root/'docs'/name).open('a',encoding='utf8') as f:f.write(note)
with (root/'docs/THESIS_PAPERS.md').open('a',encoding='utf8') as f:f.write('\n\n## Vision Transformer architectural reference (2026-09-28)\nDosovitskiy et al., An Image is Worth16x16Words: Transformers for Image Recognition at Scale (ICLR2021), https://arxiv.org/abs/2010.11929. Primary abstract consulted. Borrowed high-level idea: image-patch sequences processed by a pure transformer. Implementation: feature_transformer_multilabel_v1/train.py Model. Original small adaptation:8x8patches,96dimensions,3encoder layers,4heads,15feature tokens and binary heads instead of a single classification token. No pretrained weights or large-scale pretraining; no claimed paper reproduction or equivalent performance. Training/evaluation pending; results recorded in V2_TRANSFORMER_MULTILABEL_RESULTS.md when complete.\n')
with (root/'monitor/specialists.html').open('a',encoding='utf8') as f:f.write('\n<p><a href="transformer_multilabel_status.json">Transformer multi-label live status</a></p>')
with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
 p=subprocess.Popen(['E:/pranav/model 1/.venv/Scripts/python.exe',str(dst/'train.py')],cwd=dst,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
(run/'launch.json').write_text(json.dumps({'launcher_pid':p.pid}));print('Launched transformer worker',p.pid)
