import collections,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,new_run,load,digest
from feature_bank.views import image_views
from profile_five_datasets import generate,FEATURES

root=Path(__file__).resolve().parents[1];previous=root/'runs/feature_bank_full_reference_01'
out=new_run(root/'runs/feature_bank_cryptbb_reference_01')
rows=[json.loads(s) for s in (previous/'cryptbb.jsonl').read_text().splitlines()]
source=read(previous/'sources.json')[0];manifest=read(source['manifest'])
assert {r['pixel_sha256'] for r in rows}=={r['pixel_sha256'] for r in manifest['fit']+manifest['validation']}
assert len(rows)==source['count']==600
shutil.copy2(previous/'cryptbb.jsonl',out/'cryptbb.jsonl');shutil.copy2(previous/'thresholds.json',out/'thresholds.json')
summary=dict(images=len(rows),features={},combinations={})
for f in FEATURES:
    summary['features'][f]=dict(collections.Counter(r['predictions'][f]['image_state'] for r in rows))
    summary['features'][f]['positive_brick_fraction']=float(np.mean([s=='present' for r in rows for s in r['predictions'][f]['states']]))
for r in rows:
    key='+'.join(f for f in FEATURES if r['predictions'][f]['image_state']=='present') or 'no_confident_positive'
    summary['combinations'][key]=summary['combinations'].get(key,0)+1
write(out/'summaries.json',dict(cryptbb=summary))
write(previous/'interruption.json',dict(state='STOPPED_AT_USER_REQUEST',completed_cryptbb_images=600,
    superseded_by=str(out),other_datasets_not_completed=True))
chosen=sorted(rows,key=lambda r:r['pixel_sha256'])[:8]
canvas=Image.new('RGB',(1200,1080),'#eef1f5');draw=ImageDraw.Draw(canvas)
titles=['Original','Feature pop','Line regions','Curve regions','Dot regions']
for j,title in enumerate(titles):draw.text((j*240+8,8),title,fill='black')
for i,r in enumerate(chosen):
    im=load(r['path']);views=[im,Image.fromarray(image_views(im)['feature_pop']).convert('RGB')]
    for f in FEATURES:
        overlay=im.copy();d=ImageDraw.Draw(overlay)
        for region,state in zip(r['regions'],r['predictions'][f]['states']):
            if state=='present':d.rectangle(region['core_box'],outline=(255,65,0),width=1)
        views.append(overlay)
    for j,view in enumerate(views):
        view.thumbnail((228,98));x=j*240+6;y=34+i*128;canvas.paste(view,(x,y))
        draw.text((x,y+100),r['pixel_sha256'][:10],fill='black')
canvas.save(out/'predictions.png')
generate(out,dict(cryptbb=summary))
write(out/'status.json',dict(state='CRYPTBB_DESCRIPTIVE_COMPLETE',images=600,bricks=18000,
    accuracy_measured=False,other_datasets_stopped=True,generated_attempts=32,
    generation_failures=len(read(out/'generation_failures.json'))))
snapshot=out/'code_snapshot';snapshot.mkdir();shutil.copy2(__file__,snapshot/'report_cryptbb.py')
print(json.dumps(summary,indent=2));print('Generation failures:',len(read(out/'generation_failures.json')))
