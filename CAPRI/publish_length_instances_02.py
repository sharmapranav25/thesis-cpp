from pathlib import Path
import shutil,json,hashlib
root=Path(__file__).resolve().parent;src=root/'length_instances_02';project=Path('E:/pranav/Final sem/sourcefree_synth');dest=project/'runs/style_length_instances_02'
assert json.loads((src/'status.json').read_text())['state']=='complete'
assert not dest.exists();shutil.copytree(src,dest,ignore=shutil.ignore_patterns('__pycache__'))
manifest={str(p.relative_to(dest)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()};(dest/'INTEGRITY.json').write_text(json.dumps(manifest,indent=2))
docs=project/'docs';shutil.copy2(src/'REVIEW.md',docs/'STYLE_LENGTH_INSTANCES_02.md')
note='''

## Length instance/count comparison02 completed — failed pilot

User-authorized test completed:2architectures x3seeds,3300synthetic train,440selection,440calibration and3x440tests. Explicit glyph width/count clamp removed; width/span sampled independently, residual metadata shortcut audit retained. Primary2–8mean exact: CNN58.21/59.17/54.64%(ordinary/font/layout), hybridglobal56.67/56.43/52.38%, detected-instance count10.83/11.79/11.90%. Box precision~15%, recall13–16%atIoU.5: localisation itself failed. No runtimeVLM, target generator/font assets or real-label fitting.

320real style-fit images (64/family), no eval directory/label loading. DarkBay ensemble counts mostly10; median instance count12onall64. High seed agreement did not establish correctness. Assistant inspected fixed first12DarkBay andCity boards, found duplicates/background detections and missed glyphs. Real accuracy remains unmeasured. No pipeline promotion and no automatic length update;35-font Phase2 remains separate. Read STYLE_LENGTH_INSTANCES_02.md and runs/style_length_instances_02. Models checkpoint-selected using global count head, not optimum detection score; report this comparison limitation.

Original runnable workspace: C:/Users/pranav/Documents/ChatGPT/thesis/length_instances_02. Archived experiment.py uses a sibling CAPRI_typography_v3 font directory; when reproducing from archive supply that layout or explicitly change the source in a new version. Do not overwrite the archived run. No ongoing training left by this experiment.
'''
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md','STYLE_PIPELINE_PLAN.md']:
 with (docs/name).open('a',encoding='utf8') as f:f.write(note)
with (docs/'THESIS_PAPERS.md').open('a',encoding='utf8') as f:
 f.write('''

## DETR-inspired length pilot implemented — negative result

Carion etal.(ECCV2020), https://arxiv.org/abs/2005.12872 : query/set matching idea now implemented in runs/style_length_instances_02/experiment.py as a small CNN+2-layer decoder with12queries, character/no-object logits, boxes and an auxiliary count head. Not original DETR; no pretrained detection weights. CRAFT and CTC remain research candidates, not implemented in this trial.3seeds, synthetic-only training; detected-instance counts and localisation failed. See STYLE_LENGTH_INSTANCES_02.md for numbers and checkpoint-selection caveat. Real seed consistency also failed as a reliability signal. No automatic length integration.
''')
(project/'monitor/length_instances_02_status.json').write_text(json.dumps(dict(state='complete_failed_validation',report=str(docs/'STYLE_LENGTH_INSTANCES_02.md'),promoted=False,real_accuracy_verified=False),indent=2))
print(dest)
