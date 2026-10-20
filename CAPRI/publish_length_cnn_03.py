from pathlib import Path
import json,hashlib,shutil
root=Path(__file__).resolve().parent;source=root/'length_existing_cnn_03';project=Path('E:/pranav/Final sem/sourcefree_synth');dest=project/'runs/style_length_existing_cnn_03';docs=project/'docs'
assert json.loads((source/'status.json').read_text())['state']=='complete';assert not dest.exists()
shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__'));manifest={str(p.relative_to(dest)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()};(dest/'INTEGRITY.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(source/'RESULTS.md',docs/'STYLE_LENGTH_EXISTING_CNN_03.md')
note='''

## Simple CNN on existing100k DarkBay corpus — completed

User explicitly requested direct count training on the existing synthetic corpus. Three-convolution CNN, no transformer or instance detection, lengths2–8; synthetic-only fitting. Text-hash split89714/5144/5142. Epoch1 selected; patience2 stopped after3epochs; total123.9s. Locked synthetic exact99.4166%, MAE0.01050. This corpus retains the existing count/layout dependency and shared fonts/profiles; not a matched comparison against the previous repaired-count corpus.

Frozen full DarkBay pool inference799images, eval directory excluded, no real labels. Predicted4:3,5:18,6:357,7:6,8:415. Assistant first24review found clear overcounting to8on visible six-glyph examples, including high scores. Not real accuracy; no pipeline promotion or forced-six correction. See STYLE_LENGTH_EXISTING_CNN_03.md and runs/style_length_existing_cnn_03. Checkpoint preserved; original corpus/recogniser untouched. No training process left running by this trial.
'''
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (docs/name).open('a',encoding='utf8') as f:f.write(note)
(project/'monitor/length_existing_cnn_03_status.json').write_text(json.dumps(dict(state='complete',synthetic_exact=.9941656942823804,real_images=799,real_accuracy_verified=False,promoted=False,report=str(docs/'STYLE_LENGTH_EXISTING_CNN_03.md')),indent=2))
print(dest)
