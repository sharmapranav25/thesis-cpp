from pathlib import Path
import sys,json,copy,hashlib
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).parent;release=R/'phase12_release_02';sys.path.insert(0,str(release))
from feature_bank.core import read,write,load,pixel_hash
from style_pipeline.measure import extract_style
from style_pipeline.background_blocks import measure_blocks
from style_pipeline.render import render
out=R/'background_blocks_fix_01';out.mkdir(exist_ok=False)
fixtures={}
white=Image.new('RGB',(200,80),'white');fixtures['plain']=white
lines=white.copy();d=ImageDraw.Draw(lines)
for i in range(10):d.line((0,i*7,199,79-i*6),fill=(i*20,30,150),width=2)
fixtures['colored_lines']=lines
gradient=np.repeat(np.linspace(30,230,200).astype('uint8')[None,:,None],80,0);fixtures['smooth_gradient']=Image.fromarray(np.repeat(gradient,3,2))
rect=white.copy();d=ImageDraw.Draw(rect)
for box,color in [((10,10,40,35),(200,100,180)),((75,35,105,65),(100,200,120)),((150,8,180,35),(80,100,200))]:d.rectangle(box,fill=color)
fixtures['rectangles']=rect
checks={}
for name,im in fixtures.items():
    evidence=measure_blocks(np.asarray(im),np.zeros((im.height,im.width),dtype='uint8'));checks[name]=evidence;assert (evidence['state']=='supported')==(name=='rectangles');im.save(out/(name+'.png'))
fonts=read(release/'assets/fonts.json');stats={};panels=[]
for name in read(release/'profiles/index.json'):
    path=release/'profiles'/f'{name}.json';p=read(path);oldcounts=[];newcounts=[]
    for entry in p['entries']:
        old=entry['style'];blockgrid=np.array([b['rgb_mean'] for b in old['blocks']]).reshape(3,10,3);v=float(blockgrid.std((0,1)).mean());oldcounts.append(int(np.clip(v,6,35)) if v>8 and old['saturation_mean']>.08 and len(old['background_palette'])>=3 else 0)
        im=load(entry['reference_path_provenance']);assert pixel_hash(im)==entry['reference_pixel_sha256'];fresh=extract_style(im);old['background_block_evidence']=fresh['background_block_evidence'];newcounts.append(fresh['background_block_evidence']['count'])
    write(path,p);stats[name]=dict(references=len(oldcounts),old_triggered=sum(c>0 for c in oldcounts),new_supported=sum(c>0 for c in newcounts),old_blocks=sum(oldcounts),new_blocks=sum(newcounts))
    records=[json.loads(line) for line in (R/'phase12_validation_01'/name/'manifest.jsonl').read_text().splitlines()];r=next(v for v in records if v['bucket']=='numeric' and v['length']==5);entry=p['entries'][r['profile_entry']];im=load(entry['reference_path_provenance']);oldim=load(R/'phase12_validation_01'/name/r['filename']);new,mask,recipe=render(entry['style'],r['label'],str(release/fonts[r['font_index']]['path']),r['render_seed'],p['variant']);new.save(out/(name+'_after.png'));oldim.save(out/(name+'_before.png'));write(out/(name+'_recipe.json'),recipe)
    if name=='courier':assert not any(op['kind']=='block' for op in recipe['operations'])
    panel=Image.new('RGB',(1050,180),'#f1f3f5');draw=ImageDraw.Draw(panel);draw.text((8,5),name,fill='black')
    for col,(image,label) in enumerate([(im,'Real reference'),(oldim,'Before'),(new,'Corrected')]):
        image=image.copy();image.thumbnail((330,120));panel.paste(image,(10+350*col,35));draw.text((10+350*col,20),label,fill='black')
    panels.append(panel)
    print(name,stats[name],flush=True)
for k in range(2):
    board=Image.new('RGB',(1050,720),'white')
    for i,p in enumerate(panels[k*4:k*4+4]):board.paste(p,(0,i*180))
    board.save(out/f'comparison_{k+1}.png')
write(out/'fixture_results.json',checks);write(out/'reference_audit.json',stats)
print('VALIDATED')
