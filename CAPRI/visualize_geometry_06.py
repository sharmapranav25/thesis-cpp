from pathlib import Path
import json,math
import numpy as np,torch
from PIL import Image,ImageDraw
import run_v2_tier1_06 as v6
HERE=Path(__file__).parent;OUT=HERE/'v2_tier1_06';torch.set_num_threads(4)
model=v6.GeometryModel(True).cuda().eval();model.load_state_dict(torch.load(OUT/'geometry_with_neighbors.pt',weights_only=True)['state_dict'])
fonts=json.loads((OUT/'fonts.json').read_text());examples=[]
for i in range(100):
    item=v6.base.generate(fonts[:6],71000000+i)
    if item[2][0]>0 or item[2][1]>0:examples.append(item)
    if len(examples)==6:break
canvas=Image.new('RGB',(800,6*170),'white');draw=ImageDraw.Draw(canvas)
for row,(im,m,y,full,record) in enumerate(examples):
    x=torch.tensor(im.transpose(2,0,1)[None]/255,dtype=torch.float32,device='cuda')
    with torch.no_grad():a,z=model(x);pred=model.last_geometry[0].cpu().numpy()
    truth,_=v6.geometry_targets(m[None]);truth=truth[0]
    panels=[Image.fromarray(im).resize((256,128)) for _ in range(3)]
    for panel,g,color in [(panels[1],truth,'cyan'),(panels[2],pred,'yellow')]:
        d=ImageDraw.Draw(panel)
        for index in range(33,63):
            x0,y0,x1,y1=v6.BOUNDS[index];p,cx,cy,c,s,coherence,density=g[index]
            if p<.5 or coherence<.3:continue
            px=2*(x0+cx*(x1-x0-1));py=2*(y0+cy*(y1-y0-1));theta=.5*math.atan2(s,c);dx=9*math.cos(theta);dy=9*math.sin(theta)
            d.line((px-dx,py-dy,px+dx,py+dy),fill='black',width=5);d.line((px-dx,py-dy,px+dx,py+dy),fill=color,width=2)
        for i in range(1,10):d.line((i*256/10,0,i*256/10,128),fill=(120,120,120),width=1)
        for j in range(1,3):d.line((0,j*128/3,256,j*128/3),fill=(120,120,120),width=1)
    for col,panel in enumerate(panels):canvas.paste(panel,(col*265,row*170+22))
    draw.text((0,row*170+3),f"seed {record['seed']} | image / target tilt / predicted tilt",fill='black')
    draw.text((0,row*170+151),f"global curve target={y[1]:.0f} predicted score={a.sigmoid()[0,1]:.2f}; arrows filtered by stroke>=.5, coherence>=.3",fill='black')
canvas.save(OUT/'geometry_preview.png');print('saved geometry_preview.png')
