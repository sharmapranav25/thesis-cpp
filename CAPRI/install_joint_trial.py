from pathlib import Path
import shutil,subprocess,json,hashlib
root=Path('E:/pranav/Final sem/sourcefree_synth');src=Path(__file__).parent/'joint_mapping_trial';dst=root/'feature_joint_mapping_01'
assert not dst.exists(),'Preserve historical runs'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(src/'PLAN.md',root/'docs/V2_JOINT_MAPPING_PLAN.md')
note='\n\nUser requests BOTH architectures and clarifies joint feature-to-library mapping. Implemented gridded pseudo-token and visualConformer-inspired bidirectional models with15feature heads plus9renderer-control heads, exact synthetic control tracing tested against20render replays. Train synthetic only, same data, fresh404m..407m tests,3seeds each. Finite workers include fullreal6909 inference and matched median-vs-learned render comparisons afterward. Do not call this fullGriddedTNP/Conformer reproduction or all-parameter solution. See V2_JOINT_MAPPING_PLAN.md. Oldrelease/history retained, no promotion/automation restart.\n'
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (root/'docs'/name).open('a',encoding='utf8') as f:f.write(note)
with (root/'docs/THESIS_PAPERS.md').open('a',encoding='utf8') as f:f.write('\n\nJoint mapping01 now IMPLEMENTS original adaptations of the researched ideas (previously research-only): Ashman et al.ICML2025 gridded pseudo-token local encoder/transformer processor, https://proceedings.mlr.press/v267/ashman25a.html ; Peng et al.ICCV2021 visualConformer bidirectional feature exchange, https://github.com/pengzhiliang/Conformer . Exact code: feature_joint_mapping_01/models.py; joint supervision controls.py/train.py; renderer adapter generate.py. No source copied, no temporal/probabilistic NP decoder, no paper-scale reproduction.9parameter heads and15binary heads are our adaptation. Tests verified targets and both gradient paths; training outcomes pending and will be saved separately for each model.\n')
for kind in ['gridded','conformer']:
 run=dst/('run_'+kind);run.mkdir();hashes={}
 for part in ['train','selection','calibration']:
  for ext in ['.npz','_records.json']:
   origin=root/'feature_multilabel_v1/run'/(part+ext);shutil.copy2(origin,run/origin.name);hashes[origin.name]=hashlib.sha256(origin.read_bytes()).hexdigest()
 (run/'reused_data_hashes.json').write_text(json.dumps(hashes,indent=2))
 with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
  p=subprocess.Popen(['E:/pranav/model 1/.venv/Scripts/python.exe',str(dst/'train.py'),'--kind',kind],cwd=dst,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
 (run/'launch.json').write_text(json.dumps({'launcher_pid':p.pid,'kind':kind}));print('Launched',kind,p.pid)
 with (root/'monitor/specialists.html').open('a',encoding='utf8') as f:f.write('\n<p><a href="joint_'+kind+'_status.json">Joint '+kind+' mapping: live status</a></p>')
