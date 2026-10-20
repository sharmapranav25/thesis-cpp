import sys
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import new_run,write,read,pixel_hash
from feature_bank.views import image_views,VERSION

root=Path(__file__).resolve().parents[1];out=new_run(root/'runs/feature_bank_feature_pop_01')
corpus=root/'runs/feature_bank_corpus_candidate_01'
parents=read(corpus/'parents.json')
# Synthetic training samples only; no target evaluation labels/images needed.
chosen=[]
for size in (0,1,2,3):
    chosen.append(next(p for p in parents if p['partition']=='train' and len(p['enabled'])==size and p['accepted']))
names=['rgb','clahe','feature_pop','canny','white_tophat','black_tophat']
canvas=Image.new('RGB',(1440,540),'#edf1f5');draw=ImageDraw.Draw(canvas)
for col,name in enumerate(names):draw.text((col*240+8,8),name,fill='black')
for row,parent in enumerate(chosen):
    with Image.open(corpus/parent['file']) as im:views=image_views(im)
    folder=out/parent['id'].replace('/','_');folder.mkdir()
    for name,arr in views.items():Image.fromarray(arr).save(folder/(name+'.png'))
    for col,name in enumerate(names):
        im=Image.fromarray(views[name]).convert('RGB');im.thumbnail((228,96))
        canvas.paste(im,(col*240+6,row*128+34));draw.text((col*240+6,row*128+134),parent['id'],fill='black')
canvas.save(out/'comparison.png')
write(out/'protocol.json',dict(version=VERSION,selection=[p['id'] for p in chosen],
    masks_or_labels_used_by_transform=False,real_target_access=False,training_started=False,
    meaning='Diagnostic contrast/structure responses; letters also respond. Not segmentation or predictions.',
    settings='Fixed synthetic engineering settings; no learned/tuned evaluation parameters'))
for rel in ['feature_bank/views.py','scripts/demonstrate_feature_pop.py','tests/test_views.py']:
    p=out/'code_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/rel).read_bytes())
print(out)
