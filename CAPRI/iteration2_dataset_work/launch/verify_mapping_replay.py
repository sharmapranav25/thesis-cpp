import sys,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
project=Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0,str(project));sys.path.insert(0,str(project.parent/'captcha-effects'))
from captcha_effects.text import FontBank
from captcha_effects.captcha import render_captcha
from prediction_render_mapping import measure_distance
base=Path('C:/Users/pranav/Documents/ChatGPT/thesis')
report=json.loads((base/'prediction_mapping_audit_02/report.json').read_text())
old=json.loads((base/'iter2_target_recreation_01/report.json').read_text())
bank=FontBank()
for i,(row,goal) in enumerate(zip(report['real'],old['results'])):
    size=Image.open(row['source']).size
    a=render_captcha(np.random.default_rng(9000+i),bank,size,effects_spec=row['effects_spec'],text_spec=row['text_spec'])
    b=render_captcha(np.random.default_rng(9000+i),bank,size,effects_spec=row['effects_spec'],text_spec=row['text_spec'])
    assert np.array_equal(a['image'],b['image'])
    assert abs(measure_distance(a,goal)-row['after_distance'])<1e-9
    assert row['after_distance']<=row['before_distance']+1e-9
    row['final_usable']=a['text']['usable']
    Image.fromarray(a['image']).save(base/'prediction_mapping_audit_02'/f'{row["family"]}_{i:02d}_corrected.png')
assert hashlib.sha256(Path(old['checkpoint']).read_bytes()).hexdigest()==old['checkpoint_sha256']
result=dict(replayed=6,deterministic=True,no_distance_regressions=True,checkpoint_unchanged=True,usable=[r['final_usable'] for r in report['real']])
(base/'prediction_mapping_audit_02/verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
