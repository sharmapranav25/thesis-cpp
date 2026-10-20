import json,sys,subprocess,hashlib,shutil,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).parent;release=ROOT/'phase12_release_01';sys.path.insert(0,str(release))
from feature_bank.core import read,write,load,pixel_hash
from style_pipeline.render import render
from pipeline import make_text,partition_for
import numpy as np
versions={n:importlib.metadata.version(n) for n in ['torch','numpy','Pillow','opencv-python-headless','scikit-learn','scikit-image','joblib','scipy']}
write(release/'environment-tested.json',versions)
(release/'requirements.txt').write_text('\n'.join(f'{n}=={v.split("+")[0]}' for n,v in versions.items())+'\n')
subprocess.run([sys.executable,'-B',str(ROOT/'seal_phase12.py')],check=True)
tests=ROOT/'phase12_validation_01';tests.mkdir(exist_ok=False);index=read(release/'profiles/index.json');results={}
for family in index:
    output=tests/family
    subprocess.run([sys.executable,'-B',str(release/'pipeline.py'),'generate','--family',family,'--out',str(output),'--count','42','--seed','5501','--partition','train','--recipes'],check=True)
    rows=[json.loads(line) for line in (output/'manifest.jsonl').read_text().splitlines()]
    assert len(rows)==42 and set(read(output/'status.json')['cells'].values())=={2}
    for r in rows:
        text=r['label'];assert len(text)==r['length'] and 2<=len(text)<=8 and partition_for(text)=='train'
        if r['bucket']=='numeric':assert text.isdigit()
        elif r['bucket']=='alphabetic':assert text.isalpha()
        else:assert any(c.isalpha() for c in text) and any(c.isdigit() for c in text)
        assert len(r['recipe']['glyphs'])==len(text) and r['minimum_glyph_visibility']>=.55
    # Exact parity with earlier direct Phase1+fonts renderer, length5 seed180000/181000.
    p=read(release/'profiles'/f'{family}.json');fonts=read(release/'assets/fonts.json');text=''.join(np.random.default_rng(180000).choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),5));im,_,_=render(p['entries'][0]['style'],text,str(release/fonts[p['font_index']]['path']),181000,p['variant'])
    old=load(Path('E:/pranav/Final sem/sourcefree_synth/runs/style_direct_bank_01')/f'{family}_00.png');assert pixel_hash(im)==pixel_hash(old)
    results[family]=dict(samples=42,cells=21,exact_historical_baseline_parity=True,retries=read(output/'status.json')['retries'])
# Cross-partition label guarantees (including limited two-digit vocabulary).
rng=np.random.default_rng(4);sets={}
for part in ('train','validation','test'):
    values=set()
    for bucket in ('numeric','alphabetic','alphanumeric'):
        for length in range(2,9):
            for _ in range(100):values.add(make_text(rng,bucket,length,part)[0])
    sets[part]=values
assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
# Fresh CV+learned inference route on explicitly declared fitting references.
p=read(release/'profiles/city.json');write(tests/'references.json',dict(partition='style_fit',images=[e['reference_path_provenance'] for e in p['entries'][:2]]))
subprocess.run([sys.executable,'-B',str(release/'pipeline.py'),'profile','--references',str(tests/'references.json'),'--name','city_smoke','--out',str(tests/'fresh_profile'),'--device','cpu'],check=True)
fresh=read(tests/'fresh_profile/profile.json')
for a,b in zip(fresh['entries'],p['entries']):assert a['style']['classifier_features']==b['style']['classifier_features']
# Relocation: entrypoint runs with its own root from an unrelated working directory.
relocated=tests/'relocated_release';shutil.copytree(release,relocated)
subprocess.run([sys.executable,'-B',str(relocated/'pipeline.py'),'generate','--family','city','--out',str(tests/'relocated_output'),'--count','21','--seed','6501','--partition','validation','--mode','diverse'],cwd=tests,check=True)
write(tests/'RESULTS.json',dict(families=results,text_partition_disjoint=True,fresh_feature_inference_matches_cache=True,relocation_passed=True,training_started=False));print('ALL VALIDATIONS PASSED',flush=True)
