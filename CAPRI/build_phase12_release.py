"""Stage a portable release; never mutate historical research inputs."""
import sys,json,shutil,hashlib,importlib.metadata
from pathlib import Path
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');STAGE=Path(__file__).parent/'feature_bank_work';OUT=Path(__file__).parent/'phase12_release_01'
sys.path.insert(0,str(ROOT))
from feature_bank.core import read,write,load,pixel_hash,digest
from style_pipeline.measure import extract_style
OUT.mkdir(exist_ok=False)
provenance={}
def cp(source,dest):
    dest=OUT/dest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);provenance[str(dest.relative_to(OUT)).replace('\\','/')]=dict(source=str(source),sha256=digest(dest))
for package,names,base in [('feature_bank',['core.py','regions.py'],ROOT),('feature_bank',['descriptors.py','views.py'],STAGE),('style_pipeline',['measure.py','regions_text.py','font_match.py','render.py'],ROOT)]:
    for name in names:cp(base/package/name,package+'/'+name)
    (OUT/package/'__init__.py').write_text('')
for f in ('straight_interference','curved_interference','dot_scatter'):cp(STAGE/'runs/feature_bank_exploratory_01'/f'{f}_LR.joblib','assets/'+f+'_LR.joblib')
cp(STAGE/'runs/feature_bank_full_reference_01/thresholds.json','assets/thresholds.json')
cp(ROOT/'runs/style_phase3_font_05/font_matcher.pt','assets/font_matcher.pt')
fonts=read(ROOT/'runs/style_phase3_font_05/fonts.json')
for i,f in enumerate(fonts):
    assert digest(f['path'])==f['sha256'];assert digest(f['license_path'])==f['license_sha256']
    fontpath=f'assets/fonts/{i:02d}'+Path(f['path']).suffix;licensepath=f'assets/fonts/{i:02d}_LICENSE.txt';cp(f['path'],fontpath);cp(f['license_path'],licensepath);f['path']=fontpath;f['license_path']=licensepath
write(OUT/'assets/fonts.json',fonts)
prior=read(ROOT/'runs/style_direct_bank_01/results.json');old=read(ROOT/'runs/style_phase1_05/results.json')['families'];bank={}
for family in prior:
    if family in old:
        folder=ROOT/'runs'/('style_frozen_'+family+('_02' if family=='darkbay' else '_01'));preds=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()];lookup={p['pixel_sha256']:p for p in preds};refs=read(ROOT/'runs/style_phase1_05'/f'{family}_measurements.json')['fit'][:96];rows=[lookup[r['pixel_sha256']] for r in refs];variant=old[family]['selected']['variant']
    elif family=='roncuevas':
        folder=ROOT/'runs/style_frozen_inference_01';lookup={p['pixel_sha256']:p for p in [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]};rows=[lookup[r['pixel_sha256']] for r in read(ROOT/'runs/style_phase1_unseen_01/split.json')['fit'][:96]];variant=0
    else:
        folder=ROOT/'runs'/('style_phase3_new_family_01' if family=='pymlex' else 'style_phase3_new_family_02');rows=[r for r in read(folder/'predictions.json') if r['partition']=='train'];variant=0
    entries=[]
    for r in rows:
        im=load(r['path']);assert pixel_hash(im)==r['pixel_sha256'];s=extract_style(im);s['classifier_features']={f:v['image_state'] for f,v in r['predictions'].items()};s['font_effect']=prior[family]['effect'];entries.append(dict(style=s,reference_pixel_sha256=r['pixel_sha256'],reference_path_provenance=r['path'],feature_predictions=r['predictions']))
    profile=dict(schema=1,family=family,variant=variant,font_index=next(i for i,f in enumerate(fonts) if f['family']==prior[family]['font']),font_effect=prior[family]['effect'],font_probabilities=prior[family]['font_probabilities'],entries=entries,source='historical exposed fit references; fresh CV, cached frozen learned features, original direct-bank font selection',feature_source=str(folder),real_labels_used=False)
    write(OUT/'profiles'/f'{family}.json',profile);bank[family]=dict(profile=f'profiles/{family}.json',references=len(entries),exposure='historically exposed development; NOT untouched test')
    print('packed',family,len(entries),flush=True)
write(OUT/'profiles/index.json',bank);write(OUT/'PROVENANCE.json',provenance)
versions={name:importlib.metadata.version(name) for name in ['torch','numpy','Pillow','opencv-python','scikit-learn','scikit-image','joblib','scipy']}
write(OUT/'environment-tested.json',versions)
(OUT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
print('STAGED',OUT)
