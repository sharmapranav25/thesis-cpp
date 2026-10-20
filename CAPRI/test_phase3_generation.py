"""Synthetic fixtures only; no inferred real scheme or final data claim."""
from pathlib import Path
import itertools,json,subprocess,sys
import numpy as np
import phase3_targeted_generation as g
root=Path(__file__).resolve().parent/'phase3_generation_fixture_01'
root.mkdir(exist_ok=False)
cells=[dict(length=n,groups=''.join(map(str,b))) for n in range(2,9) for b in itertools.product([0,1],repeat=3) if 0<sum(b)<=n]
for cell in cells:cell['sampling_weight']=1/len(cells)
rng=np.random.default_rng(17)
for cell in cells:
    for _ in range(25):
        t=g.draw_text(rng,cell['length'],cell['groups']);assert len(t)==cell['length'] and g.group_key(t)==cell['groups'] and g.text_partition(t)=='train'
fixture=dict(synthetic_fixture=True,synthetic_reading_gate=dict(passed=True),source_partition='fit',validation_used_for_scheme=False,joint_distribution=cells)
(root/'fixture_scheme.json').write_text(json.dumps(fixture))
subprocess.run([sys.executable,str(Path(g.__file__)),'--worker','--scheme',str(root/'fixture_scheme.json'),'--out',str(root/'images_fixture'),'--count','210','--seed','991001'],check=True)
rows=[json.loads(l) for l in (root/'images_fixture/manifest.jsonl').open()]
assert len(rows)==210 and len({f for r in rows for f in r['font_ids']})==35
for r in rows:
    im=g.load_image(root/'images_fixture'/r['filename']);assert g.pixels(im)==r['pixel_sha256'];assert ''.join(x['character'] for x in r['recipe']['glyphs'])==r['label']
    for glyph in r['recipe']['glyphs']:
        x0,y0,x1,y1=glyph['box'];assert 0<=x0<x1<=im.width and 0<=y0<y1<=im.height
print('PASS: all48 feasible length/group cells sampled; 210 renders,35 fonts, exact replay20, label/hash/geometry verified. Fixture only.')
