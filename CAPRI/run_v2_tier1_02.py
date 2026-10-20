"""Tier1 presence/localization pilot; synthetic supervision only."""
import os
os.environ['OMP_NUM_THREADS']='4'
import sys,json,time,hashlib,shutil
from pathlib import Path
import numpy as np
import cv2,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');RELEASE=ROOT/'releases/phase12_v2';OUT=Path(__file__).parent/'v2_tier1_02'
FEATURES=['straight_lines','curves','dots','rectangles','grain'];GRIDS=[(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(5,3),(10,3)]
def write(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2),encoding='utf8')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def generate(fonts,seed,style=0):
    rng=np.random.default_rng(seed);w,h=128,64;bg=rng.integers(130,256,3);a=np.broadcast_to(bg,(h,w,3)).copy().astype(float)
    # Background gradient is a nuisance here, not a labeled grain feature.
    if rng.random()<.35:a+=np.linspace(-20,20,w)[None,:,None]
    active=rng.random(5)<.45
    if seed%10==0:active[:]=False
    layers=[];params=[]
    for k in range(5):
        mask=Image.new('L',(w,h));d=ImageDraw.Draw(mask);count=0;thickness=int(rng.integers(1,4))
        if active[k]:
            count=int(rng.integers(1,6))
            if k==0:
                for _ in range(count):d.line(tuple(rng.integers([0,0,0,0],[w,h,w,h])),fill=255,width=thickness)
            elif k==1:
                for _ in range(count):
                    if style==1:
                        pts=rng.integers([0,0],[w,h],(3,2));t=np.linspace(0,1,60)[:,None];xy=(1-t)**2*pts[0]+2*(1-t)*t*pts[1]+t*t*pts[2]
                    else:
                        x=np.linspace(0,w-1,60);xy=np.stack([x,rng.uniform(10,54)+rng.uniform(3,18)*np.sin(x/w*rng.uniform(3,9)+rng.uniform(0,6))],1)
                    d.line([tuple(map(int,p)) for p in xy],fill=255,width=thickness)
            elif k==2:
                count=int(rng.integers(8,100))
                for _ in range(count):
                    x,y=rng.integers([0,0],[w,h]);r=int(rng.integers(0,2));d.ellipse((x-r,y-r,x+r,y+r),fill=255)
            elif k==3:
                for _ in range(count):
                    x,y=rng.integers([0,0],[w-8,h-8]);bw,bh=rng.integers([5,5],[30,35]);d.rectangle((x,y,min(w-1,x+bw),min(h-1,y+bh)),fill=255)
            else:
                noise=rng.random((h,w))<rng.uniform(.15,.7);mask=Image.fromarray(np.uint8(noise)*255);count=int(noise.sum())
        layers.append(np.asarray(mask)/255);params.append(dict(count=count,width=thickness))
    order=[4,3,0,1,2];visible=[m.copy() for m in layers]
    for pos,k in enumerate(order):
        for later in order[pos+1:]:visible[k]*=1-layers[later]
        color=rng.integers(0,256,3);a=a*(1-layers[k][:,:,None])+color*layers[k][:,:,None]
    textmask=Image.new('L',(w,h));d=ImageDraw.Draw(textmask);font=ImageFont.truetype(str(RELEASE/fonts[int(rng.integers(len(fonts)))]['path']),int(rng.integers(20,38)));n=int(rng.integers(2,9));text=''.join(rng.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'),n));span=rng.uniform(.45,.9);centers=np.linspace(w*(1-span)/2,w*(1+span)/2,n)
    for c,x in zip(text,centers):
        box=font.getbbox(c);d.text((int(x-(box[2]-box[0])/2),int(rng.integers(7,25))-box[1]),c,font=font,fill=255,stroke_width=int(rng.random()<.15))
    alpha=np.asarray(textmask)/255;color=rng.integers(0,160,3);a=a*(1-alpha[:,:,None])+color*alpha[:,:,None]
    for k in range(5):visible[k]*=1-alpha
    image=np.uint8(np.clip(a,0,255));full=np.stack(layers).astype('float32');visible=np.stack(visible).astype('float32');labels=(visible.sum((1,2))>=3).astype('float32')
    if rng.random()<.2:image=255-image
    return image,visible,labels,full,dict(seed=seed,text=text,full_coverage=full.mean((1,2)).tolist(),visible_coverage=visible.mean((1,2)).tolist(),parameters=params,render_recipe=style)
class Multi(nn.Module):
    def __init__(self,multiview):
        super().__init__();self.multiview=multiview;self.encoder=nn.Sequential(nn.Conv2d(3,24,3,2,1),nn.BatchNorm2d(24),nn.ReLU(),nn.Conv2d(24,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,48,3,2,1),nn.BatchNorm2d(48),nn.ReLU());self.score=nn.Sequential(nn.Linear(52,32),nn.Tanh(),nn.Linear(32,1));self.presence=nn.Linear(48,5);self.location=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,5,1));self.context=nn.Linear(48,48)
        boxes=[];grids=[]
        for cols,rows in ([(10,3)] if multiview=='single_grid' else GRIDS if multiview else [(1,1)]):
            for j in range(rows):
                for i in range(cols):
                    x0,x1=i/cols,(i+1)/cols;y0,y1=j/rows,(j+1)/rows;boxes.append([x0,y0,x1-x0,y1-y0]);yy,xx=torch.meshgrid(torch.linspace(y0*2-1,y1*2-1,24),torch.linspace(x0*2-1,x1*2-1,48),indexing='ij');grids.append(torch.stack([xx,yy],-1))
        self.register_buffer('grid',torch.stack(grids));self.register_buffer('boxes',torch.tensor(boxes))
    def forward(self,x):
        b=len(x);v=len(self.grid);views=F.grid_sample(x[:,None].expand(-1,v,-1,-1,-1).reshape(b*v,3,64,128),self.grid[None].expand(b,-1,-1,-1,-1).reshape(b*v,24,48,2),align_corners=True);tokens=self.encoder(views).mean((2,3)).reshape(b,v,48);weights=self.score(torch.cat([tokens,self.boxes[None].expand(b,-1,-1)],2)).softmax(1);context=(tokens*weights).sum(1);spatial=self.encoder(x)+self.context(context)[:,:,None,None];return self.presence(context),F.interpolate(self.location(spatial),size=(32,64),mode='bilinear',align_corners=False)
def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(431);start=time.time();fonts=json.loads((RELEASE/'assets/fonts.json').read_text())[:9]
    for f in fonts:assert digest(RELEASE/f['path'])==f['sha256']
    write(OUT/'fonts.json',fonts);write(OUT/'protocol.json',dict(tier=1,real_families=['cryptbb','city','courier','darkbay','outlined_text_captcha'],font_families=[f['family'] for f in fonts],train_fonts=[f['family'] for f in fonts[:6]],selection_font=fonts[6]['family'],heldout_fonts=[f['family'] for f in fonts[7:]],features=FEATURES,grids_columns_rows=GRIDS,views=63,training='synthetic only; no real labels',milestone='presence and visible localization; density/geometry metadata saved but prediction heads deferred',steps=1500,batch=12,selection='minimum selection BCE+dice composite every100steps',threshold=.5,limitations='original procedural generator only; holdout curves use quadraticBezier vs sine; not robust V2 acceptance',tiers2and3='deferred'))
    shutil.copy2(__file__,OUT/'script_snapshot.py');data={}
    for part,base,count,bank,recipe in [('train',30000000,2400,fonts[:6],0),('selection',31000000,420,fonts[6:7],0),('test',32000000,420,fonts[:6],0),('heldout',33000000,420,fonts[7:],1)]:
        images=[];masks=[];labels=[];records=[];full_masks=[];visible_masks=[]
        for i in range(count):
            im,mask,y,full,record=generate(bank,base+i,recipe);full_masks.append(np.uint8(full*255));visible_masks.append(np.uint8(mask*255));images.append(im);masks.append(cv2.resize(mask.transpose(1,2,0),(64,32),interpolation=cv2.INTER_AREA).transpose(2,0,1));labels.append(y);records.append(record)
            if i<12 and part=='train':Image.fromarray(im).save(OUT/f'train_{i:02d}.png')
        data[part]=(torch.tensor(np.stack(images).transpose(0,3,1,2)/255,device='cuda',dtype=torch.float32),torch.tensor(np.stack(masks),device='cuda'),torch.tensor(np.stack(labels),device='cuda'));write(OUT/(part+'_records.json'),records);np.savez_compressed(OUT/(part+'_supervision.npz'),images=np.stack(images),full_masks=np.stack(full_masks),visible_masks=np.stack(visible_masks),labels=np.stack(labels));print('BUILT',part,count,flush=True)
    def lossfn(a,z,y,m):
        prob=z.sigmoid();dice=1-(2*(prob*m).sum((2,3))+1)/((prob+m).sum((2,3))+1);return 3*F.binary_cross_entropy_with_logits(a,y,pos_weight=presence_weight)+F.binary_cross_entropy_with_logits(z,m,pos_weight=mask_weight)+dice.mean()
    presence_weight=((1-data['train'][2]).sum(0)/data['train'][2].sum(0).clamp_min(1)).detach();mask_weight=((1-data['train'][1]).sum((0,2,3))/data['train'][1].sum((0,2,3)).clamp_min(1)).clamp(1,20)[None,:,None,None].detach();write(OUT/'loss_weights.json',dict(presence=presence_weight.cpu().tolist(),mask=mask_weight.flatten().cpu().tolist()));results={};models={}
    for name,multi in [('whole_image',False),('single_grid','single_grid'),('multiview',True)]:
        torch.manual_seed(431);model=Multi(multi).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);rng=np.random.default_rng(431);best=1e9;history=[]
        def evaluate(part):
            x,m,y=data[part];aa=[];zz=[];model.eval()
            with torch.no_grad():
                for i in range(0,len(x),12):a,z=model(x[i:i+12]);aa.append(a);zz.append(z)
            return torch.cat(aa),torch.cat(zz),m,y
        for step in range(1,1501):
            model.train();x,m,y=data['train'];ids=rng.integers(len(x),size=12);a,z=model(x[ids]);loss=lossfn(a,z,y[ids],m[ids]);opt.zero_grad();loss.backward();opt.step()
            if step%100==0:
                a,z,m,y=evaluate('selection');v=float(lossfn(a,z,y,m));history.append(dict(step=step,train=float(loss.detach()),selection=v));write(OUT/(name+'_training.json'),history)
                if v<best:best=v;torch.save(dict(state_dict=model.state_dict(),step=step),OUT/(name+'.pt'))
                print(name,step,v,flush=True)
        saved=torch.load(OUT/(name+'.pt'),weights_only=True);model.load_state_dict(saved['state_dict']);models[name]=model;results[name]={'selected_step':saved['step']}
        for part in ('test','heldout'):
            a,z,m,y=evaluate(part);p=a.sigmoid()>=.5;t=y.bool();tp=(p&t).sum(0);fp=(p&~t).sum(0);fn=(~p&t).sum(0);f1=2*tp/(2*tp+fp+fn).clamp_min(1);pred=z.sigmoid()>=.5;truth=m>=.25;intersection=(pred&truth).sum((0,2,3));union=(pred|truth).sum((0,2,3));results[name][part]=dict(f1=f1.cpu().tolist(),false_positive_rate=(fp/(~t).sum(0).clamp_min(1)).cpu().tolist(),visible_mask_iou=(intersection/union.clamp_min(1)).cpu().tolist())
        write(OUT/'results.json',results)
    # Real images only after synthetic checkpoint selection; no real fitting.
    from PIL import ImageDraw
    model=models['multiview'];panel=Image.new('RGB',(1120,700),'white');draw=ImageDraw.Draw(panel);realresults={}
    for row,family in enumerate(['cryptbb','city','courier','darkbay','outlined_text_captcha']):
        profile=json.loads((RELEASE/'profiles'/f'{family}.json').read_text());items=[]
        for col,e in enumerate(profile['entries'][:4]):
            im=Image.open(e['reference_path_provenance']).convert('RGB').resize((128,64));x=torch.tensor(np.asarray(im).transpose(2,0,1)[None]/255,dtype=torch.float32,device='cuda')
            with torch.no_grad():a,z=model(x);prob=a.sigmoid()[0].cpu().numpy();maps=z.sigmoid()[0].cpu().numpy()
            items.append(dict(reference=e['reference_path_provenance'],probabilities=prob.tolist()));heat=cv2.resize(maps[3],(128,64));colored=cv2.cvtColor(cv2.applyColorMap(np.uint8(heat*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB);overlay=Image.blend(im,Image.fromarray(colored),.45);panel.paste(im,(10+280*col,20+140*row));panel.paste(overlay,(140+280*col,20+140*row))
            draw.text((10+280*col,90+140*row),' '.join(f'{p:.2f}' for p in prob),fill='black')
        draw.text((10,140*row+5),family+' | line curve dot rectangle grain scores; no real accuracy',fill='black');realresults[family]=items
    panel.save(OUT/'real_predictions.png');write(OUT/'real_inference.json',realresults);write(OUT/'status.json',dict(state='complete',seconds=time.time()-start,trained=True,tier=1,gate_pass=False,renderer_integrated=False));print('COMPLETE',flush=True)
if __name__=='__main__':main()
