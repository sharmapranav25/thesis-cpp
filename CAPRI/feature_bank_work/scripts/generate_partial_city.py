"""User-requested generation from completed City predictions only."""
import json,shutil
import numpy as np
from PIL import Image,ImageDraw
from profile_five_datasets import ROOT,FEATURES,generate
from feature_bank.core import read,write,new_run,load
from feature_bank.views import image_views

previous=ROOT/'runs/feature_bank_city_reference_01'
out=new_run(ROOT/'runs/feature_bank_city_partial_generation_01')
rows=[];ignored=0
for line in (previous/'city.jsonl').read_text().splitlines():
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:ignored+=1
assert rows and len({r['pixel_sha256'] for r in rows})==len(rows)
with (out/'city.jsonl').open('w') as stream:
    for r in rows:stream.write(json.dumps(r)+'\n')
shutil.copy2(previous/'thresholds.json',out/'thresholds.json')
summary=dict(images=len(rows),features={},combinations={})
for f in FEATURES:
    summary['features'][f]={s:sum(r['predictions'][f]['image_state']==s for r in rows) for s in ('present','absent','uncertain')}
    summary['features'][f]['positive_brick_fraction']=float(np.mean([s=='present' for r in rows for s in r['predictions'][f]['states']]))
for r in rows:
    key='+'.join(f for f in FEATURES if r['predictions'][f]['image_state']=='present') or 'no_confident_positive'
    summary['combinations'][key]=summary['combinations'].get(key,0)+1
write(out/'summaries.json',dict(city=summary))
write(previous/'interruption.json',dict(state='STOPPED_AT_USER_REQUEST',saved_images=len(rows),ignored_incomplete_lines=ignored,
    continued_as=str(out),total_registered=1999))
generate(out,dict(city=summary))
canvas=Image.new('RGB',(1200,1080),'#eef1f5');draw=ImageDraw.Draw(canvas)
for j,title in enumerate(['Original','Feature pop','Line regions','Curve regions','Dot regions']):draw.text((j*240+8,8),title,fill='black')
for i,r in enumerate(sorted(rows,key=lambda r:r['pixel_sha256'])[:8]):
    im=load(r['path']);views=[im,Image.fromarray(image_views(im)['feature_pop']).convert('RGB')]
    for f in FEATURES:
        overlay=im.copy();d=ImageDraw.Draw(overlay)
        for region,state in zip(r['regions'],r['predictions'][f]['states']):
            if state=='present':d.rectangle(region['core_box'],outline=(255,65,0),width=1)
        views.append(overlay)
    for j,view in enumerate(views):
        view.thumbnail((228,98));x=j*240+6;y=34+i*128;canvas.paste(view,(x,y));draw.text((x,y+100),r['pixel_sha256'][:10],fill='black')
canvas.save(out/'predictions.png')
write(out/'status.json',dict(state='PARTIAL_REFERENCE_GENERATION_COMPLETE',reference_images=len(rows),registered_images=1999,
    generated_attempts=32,generation_failures=len(read(out/'generation_failures.json')),accuracy_measured=False,style_recreation_verified=False))
snapshot=out/'code_snapshot';snapshot.mkdir();shutil.copy2(__file__,snapshot/'generate_partial_city.py')
shutil.copy2(ROOT/'scripts/profile_five_datasets.py',snapshot/'profile_five_datasets.py')
print(json.dumps(summary,indent=2));print('Generation failures:',len(read(out/'generation_failures.json')))
