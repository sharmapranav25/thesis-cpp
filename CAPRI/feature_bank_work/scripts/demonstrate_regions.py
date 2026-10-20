"""Five-dataset geometry demo, not classification or real-data fitting."""
import base64
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,digest,load,pixel_hash,new_run,FEATURES
from feature_bank.regions import regions,overlap_edges,verify_crops,check_group_partitions,SCALES,HALO_FRACTION

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
parser=argparse.ArgumentParser()
parser.add_argument('--columns',type=int)
parser.add_argument('--rows',type=int)
parser.add_argument('--out',default='runs/feature_bank_regions_01')
args=parser.parse_args()
if (args.columns is None)!=(args.rows is None): raise ValueError('Specify both rows and columns')
if args.columns is not None: SCALES=((args.rows,args.columns),)
OUT=new_run(ROOT/args.out)
selection=PROJECT/'runs/feature_bank_01'
manifest=read(selection/'manifest.json')
protocol=dict(version='regions-v1',scales=SCALES,halo_per_side=HALO_FRACTION,
    selection='lowest decoded pixel SHA256 among the 40 previously frozen validation members per dataset',
    no_fitting=True,no_predictions=True,no_annotation_files_read=True,
    geometry_not_feature_connections=True,source_image_is_split_unit=False,
    split_unit='base generation group including all variants, crops and paired renders; duplicate parent hashes also grouped',
    labels='all unknown; no inheritance from whole images',
    settings_status='PROPOSED fixed engineering demonstration, not tuned or accuracy-validated')
write(OUT/'protocol.json',protocol)
items=[]; checks=[]

def add_image(sid,image,role,parent_group,source_record):
    folder=OUT/sid; folder.mkdir()
    image.save(folder/'original.png')
    rs=regions(*image.size,scales=SCALES); validation=verify_crops(image,rs)
    parent=pixel_hash(image)
    for r in rs:
        crop=image.crop(r['box']); crop.save(folder/(r['id']+'.png'))
        r.update(parent_pixel_sha256=parent,base_group=parent_group,partition=role,
                 file=sid+'/'+r['id']+'.png',pixel_sha256=pixel_hash(crop),
                 labels={f:'unknown' for f in FEATURES},label_status='NOT_ANNOTATED_NOT_A_NEGATIVE',
                 training_eligible=False)
    check_group_partitions(rs)
    graph=overlap_edges(rs)
    record=dict(id=sid,role=role,source=source_record,parent_pixel_sha256=parent,base_group=parent_group,
                size=list(image.size),regions=rs,geometric_links=graph,checks=validation)
    write(folder/'regions.json',record); items.append(record); checks.append(dict(id=sid,**validation))
    # Static presentation: nearest-neighbor enlargement only, exports stay native.
    rows,cols=SCALES[0]; cellw=max(155,980//cols); panelw=max(1000,40+cellw*cols)
    panel=Image.new('RGB',(panelw,350+rows*155),'#f3f5f7'); d=ImageDraw.Draw(panel)
    d.text((20,12),sid+' | fixed grid demonstration; no feature predictions',fill='black')
    original=image.resize((image.width*3,image.height*3),Image.Resampling.NEAREST)
    original.thumbnail((950,225),Image.Resampling.NEAREST)
    panel.paste(original,(20,40)); sx=original.width/image.width; sy=original.height/image.height
    for r in rs[:rows*cols]:
        x0,y0,x1,y1=r['core_box']; d.rectangle((20+x0*sx,40+y0*sy,20+x1*sx-1,40+y1*sy-1),outline='#e66c00',width=2)
    d.text((20,280),f'{cols} across x {rows} down = {rows*cols} base bricks. Overlapping crops below; borders are display-only.',fill='black')
    for i,r in enumerate(rs[:rows*cols]):
        x=20+(i%cols)*cellw;y=310+(i//cols)*155
        d.text((x,y),r['id'],fill='black')
        d.text((x,y+15),str(r['box']),fill='black')
        crop=image.crop(r['box']); factor=min((cellw-15)/crop.width,103/crop.height)
        crop=crop.resize((round(crop.width*factor),round(crop.height*factor)),Image.Resampling.NEAREST)
        panel.paste(crop,(x,y+33))
    d.text((20,panel.height-22),'All crops retain native pixels. Unknown feature labels; no model predictions.',fill='black')
    panel.save(folder/'overview.png')

for name in sorted({r['dataset'] for r in manifest['images']}):
    row=min((r for r in manifest['images'] if r['dataset']==name),key=lambda r:r['pixel_sha256'])
    image=load(selection/row['image'])
    if pixel_hash(image)!=row['pixel_sha256']: raise ValueError('Frozen evaluation image changed')
    add_image(name,image,'evaluation_only',row['pixel_sha256'],dict(id=row['id'],manifest_sha256=digest(selection/'manifest.json')))

# Same crop code on one existing pilot image per backend. No new training labels.
for backend in ('gregwar','securimage','mobicms','simple','python'):
    source=PROJECT/'runs/feature_bank_pilot_01'
    rows=read(source/backend/'results.json')
    row=min((r for r in rows if r['status']=='RENDERED'),key=lambda r:r['pixel_sha256'])
    add_image('synthetic_'+backend,load(source/row['image']),'pilot_only',row['base_group'],
              dict(id=row['id'],run='feature_bank_pilot_01',training_eligible=False))
write(OUT/'manifest.json',dict(protocol=protocol,images=items,checks=checks,
      code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'feature_bank/regions.py']}))

data=[]
for row in items:
    r=dict(row)
    r['image']='data:image/png;base64,'+base64.b64encode((OUT/row['id']/'original.png').read_bytes()).decode()
    data.append(r)
html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>CAPTCHA bricks demonstration</title>
<style>body{font:16px system-ui;max-width:1100px;margin:28px auto;padding:0 20px;background:#f5f7f9;color:#152235}select,button{font:inherit;padding:8px}canvas{max-width:100%;background:white}#crop{image-rendering:pixelated;min-width:240px;max-width:100%}.controls{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0}#bricks{display:flex;flex-wrap:wrap;gap:6px;margin:20px 0}.active{outline:3px solid #d56910}small{color:#475569}section{margin:24px 0}h1{font-size:26px}</style>
<h1>One image, connected pieces</h1><p>Select a brick to see its position and overlapping neighbors. This demonstrates where evidence could connect. It does not identify any feature yet.</p>
<div class="controls"><label>Image <select id="dataset"></select></label><label>Grid <select id="scale">__OPTIONS__</select></label></div>
<p id="role"></p><canvas id="canvas" aria-label="Original image with brick boundaries and selected overlapping crop"></canvas><div id="bricks"></div>
<section><h2 id="title"></h2><img id="crop" alt="Selected native-pixel crop enlarged for display"><p id="box"></p><p id="neighbors"></p></section>
<p><b>Orange:</b> selected crop, including overlap. <b>Blue:</b> neighboring crops sharing source pixels. Gray lines divide the base bricks. A shared border or shared pixels do not establish a continuous line, glyph, or texture.</p>
<p>Every feature label is unknown. These evaluation images and their pieces are excluded from training. Display enlargement does not change the saved native-pixel crops.</p>
<script>const data=__DATA__;const $=id=>document.getElementById(id);let selected=0,img=new Image(),request=0;
data.forEach((r,i)=>$('dataset').add(new Option(r.id,i)));
function render(){let row=data[+$('dataset').value],rs=row.regions.filter(r=>r.scale.join(',')===$('scale').value);selected=Math.min(selected,rs.length-1);let r=rs[selected],links=row.geometric_links.filter(e=>e.a===r.id||e.b===r.id),neighborIds=links.map(e=>e.a===r.id?e.b:e.a);let c=$('canvas'),s=Math.min(4,1000/row.size[0]);c.width=Math.round(row.size[0]*s);c.height=Math.round(row.size[1]*s);let ctx=c.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(img,0,0,c.width,c.height);
for(let q of rs){ctx.strokeStyle='#687482';ctx.lineWidth=1;let [x,y,xx,yy]=q.core_box;ctx.strokeRect(x*s,y*s,(xx-x)*s,(yy-y)*s)}
for(let q of rs.filter(q=>neighborIds.includes(q.id)).concat([r])){ctx.strokeStyle=q.id===r.id?'#d56910':'#1473aa';ctx.lineWidth=q.id===r.id?3:1;let [x,y,xx,yy]=q.box;ctx.strokeRect(x*s,y*s,(xx-x)*s,(yy-y)*s)}
$('bricks').replaceChildren();rs.forEach((q,i)=>{let b=document.createElement('button');b.textContent=q.id;b.className=i===selected?'active':'';b.onclick=()=>{selected=i;render()};$('bricks').append(b)});
let temp=document.createElement('canvas'),[x,y,xx,yy]=r.box;temp.width=xx-x;temp.height=yy-y;temp.getContext('2d').drawImage(img,x,y,xx-x,yy-y,0,0,xx-x,yy-y);$('crop').src=temp.toDataURL();$('title').textContent=r.id;$('box').textContent='Native box [x0,y0,x1,y1): '+r.box.join(', ')+' | '+(xx-x)+' × '+(yy-y)+' pixels';$('neighbors').textContent='Geometric neighbors: '+neighborIds.join(', ');$('role').textContent=row.role+' — no classifications or audited labels';}
function load(){const id=++request;img=new Image();img.onload=()=>{if(id===request)render()};img.src=data[+$('dataset').value].image} $('dataset').onchange=()=>{selected=0;load()};$('scale').onchange=()=>{selected=0;render()};load();</script></html>'''
options=''.join(f'<option value="{r},{c}">{c} across × {r} down</option>' for r,c in SCALES)
(OUT/'explore.html').write_text(html.replace('__OPTIONS__',options).replace('__DATA__',json.dumps(data).replace('</','<\\/')),encoding='utf-8')
write(OUT/'status.json',dict(state='GEOMETRY_DEMO_COMPLETE_NOT_CLASSIFICATION',real_images=5,synthetic_pilot_images=5,
    regions_per_image=sum(r*c for r,c in SCALES),total_regions=len(items)*sum(r*c for r,c in SCALES),checks=checks,model_training=False,feature_predictions=False))
print(OUT)
