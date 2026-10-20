"""Frozen DINO/CV versus synthetic density supervision. No real labels."""
import os
os.environ['OMP_NUM_THREADS']='4'
import sys,json,time,hashlib,shutil
from pathlib import Path
import numpy as np
import cv2,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw
from scipy.signal import find_peaks
from count_density_generator import synthetic,encode,ROOT
from train_character_count import CountCNN

OUT=Path(__file__).parent/'count_heatmaps_01'
def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Density(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(2,24,5,padding=2),nn.ReLU(),nn.Conv2d(24,32,3,2,1),nn.ReLU(),nn.Conv2d(32,48,3,2,1),nn.ReLU(),nn.Conv2d(48,48,3,padding=1),nn.ReLU(),nn.Conv2d(48,1,1));nn.init.constant_(self.net[-1].bias,-5)
    def forward(self,x):return F.softplus(self.net(x))[:,0]
def target(masks):
    yy,xx=np.mgrid[:16,:40];out=np.zeros((16,40),np.float32)
    for m in masks:
        y,x=np.nonzero(m);g=np.exp(-((xx-x.mean()*40/224)**2+(yy-y.mean()*16/72)**2)/(2*1.1**2));out+=g/g.sum()
    assert abs(out.sum()-len(masks))<1e-4
    return out
def cvmap(im):
    a=np.asarray(im.convert('L').resize((224,72)));a=255-a if np.median(a)<128 else a
    return (cv2.threshold(a,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]/255).astype('float32')
def peaks(a,dist,prom):
    p=a.sum(0).astype('float32');p=cv2.GaussianBlur(p[None],(3,1),0)[0];p=(p-p.min())/(np.ptp(p)+1e-8)
    return find_peaks(np.pad(p,1),distance=dist,prominence=prom)[0]-1
def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(731);start=time.time()
    shutil.copy2(__file__,OUT/'script_snapshot.py');shutil.copy2(Path(__file__).parent/'count_density_generator.py',OUT/'generator_snapshot.py')
    cache=Path.home()/'.cache/torch/hub';repo=cache/'facebookresearch_dino_main';weights=cache/'checkpoints/dino_deitsmall8_pretrain.pth'
    write('protocol.json',dict(real_training=False,selection='synthetic only; DINO head including mean, peak distance and prominence chosen by MAE; density checkpoint selection MAE',dino_weights_sha256=sha(weights),dino_source_sha256=sha(repo/'vision_transformer.py'),source='https://github.com/facebookresearch/dino',license='Apache-2.0; cached source commit unknown, exact source hashes recorded',density='unit mass Gaussian per synthetic glyph mask centroid; CNN MSE map plus count MSE; not MESA reproduction',limits='same generator as previous count pilot; count-layout shortcut remains; no real accuracy labels; automation paused'))
    shutil.copy2(repo/'vision_transformer.py',OUT/'dino_vision_transformer.py');shutil.copy2(repo/'utils.py',OUT/'dino_utils.py');shutil.copy2(repo/'LICENSE',OUT/'DINO_LICENSE')
    sys.path.insert(0,str(repo));import vision_transformer as vit
    dino=vit.vit_small(patch_size=8).cuda().eval();dino.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    for p in dino.parameters():p.requires_grad=False
    def attention(images):
        outputs=[]
        for i in range(0,len(images),32):
            x=np.stack([np.asarray(im.convert('RGB').resize((224,72)),dtype='float32').transpose(2,0,1)/255 for im in images[i:i+32]])
            x=torch.tensor(x,device='cuda');x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
            with torch.no_grad():a=dino.get_last_selfattention(x)[:,:,0,1:].reshape(-1,6,9,28).cpu().numpy()
            outputs.extend(np.concatenate([a,a.mean(1,keepdims=True)],1))
        return np.stack(outputs)
    fonts=json.loads((ROOT/'runs/style_phase3_font_05/fonts.json').read_text());data={}
    for part,base,num,bank in [('train',11000000,192,fonts[:10]),('selection',12000000,24,fonts[:10]),('test',13000000,48,fonts[:10]),('heldout_fonts',14000000,48,fonts[10:])]:
        ims=[];ys=[];ts=[]
        for n in range(2,9):
            for i in range(num):
                im,text,masks=synthetic(bank,base+n*10000+i,n);ims.append(im);ys.append(n);ts.append(target(masks))
        data[part]=dict(images=ims,y=np.array(ys),x=torch.tensor(np.stack([encode(im) for im in ims]),device='cuda'),target=torch.tensor(np.stack(ts),device='cuda'))
        print('Built',part,len(ims),flush=True)
    model=Density().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001);rng=np.random.default_rng(731);best=1e9;history=[]
    def density(x):
        model.eval()
        with torch.no_grad():return torch.cat([model(x[i:i+64]) for i in range(0,len(x),64)])
    for step in range(1,1001):
        model.train();d=data['train'];ids=rng.integers(len(d['y']),size=48);pred=model(d['x'][ids]);t=d['target'][ids];loss=100*F.mse_loss(pred,t)+.1*F.mse_loss(pred.sum((1,2)),t.sum((1,2)));opt.zero_grad();loss.backward();opt.step()
        if step%100==0:
            v=density(data['selection']['x']).sum((1,2)).cpu().numpy();mae=float(np.abs(v-data['selection']['y']).mean());history.append(dict(step=step,selection_mae=mae,train_loss=float(loss.detach())));write('training.json',history)
            if mae<best:best=mae;torch.save(dict(state_dict=model.state_dict(),step=step),OUT/'density.pt')
            print('density',step,mae,flush=True)
    saved=torch.load(OUT/'density.pt',weights_only=True);model.load_state_dict(saved['state_dict']);model.eval()
    d=data['selection'];a=attention(d['images']);cm=[cvmap(im) for im in d['images']]
    def choose(maps,heads,distances):
        options=[]
        for head in heads:
            for distance in distances:
                for prominence in (.1,.2,.3,.4):
                    guesses=np.array([len(peaks(m[head] if head>=0 else m,distance,prominence)) for m in maps]);options.append((float(abs(guesses-d['y']).mean()),head,distance,prominence))
        return min(options)
    ds=choose(a,range(7),(2,3,4));cs=choose(cm,[-1],(8,12,16,20));write('selected.json',dict(dino=ds,cv=cs,density_step=saved['step']))
    direct=CountCNN().cuda().eval();direct.load_state_dict(torch.load(ROOT/'runs/style_character_count_01/count_model.pt',weights_only=True)['state_dict'])
    metrics={}
    for part in ('test','heldout_fonts'):
        d=data[part];a=attention(d['images']);dm=density(d['x']).cpu().numpy()
        with torch.no_grad():old=direct(d['x']).argmax(1).cpu().numpy()+2
        predictions={'cv':np.array([len(peaks(cvmap(im),cs[2],cs[3])) for im in d['images']]),'dino':np.array([len(peaks(m[ds[1]],ds[2],ds[3])) for m in a]),'density':dm.sum((1,2)),'previous_count_cnn':old};metrics[part]={k:dict(exact=float((np.rint(v)==d['y']).mean()),mae=float(abs(v-d['y']).mean())) for k,v in predictions.items()}
    write('metrics.json',metrics);print(json.dumps(metrics),flush=True)
    previous=json.loads((ROOT/'runs/style_character_count_01/real_inference.json').read_text());results={};panels=[]
    for family,record in previous.items():
        rows=record['predictions'];ims=[Image.open(r['path']).convert('RGB') for r in rows];a=attention(ims);x=torch.tensor(np.stack([encode(im) for im in ims]),device='cuda');dm=density(x).cpu().numpy();res=[]
        panel=Image.new('RGB',(1000,360),'white');draw=ImageDraw.Draw(panel);draw.text((8,5),family+' | real predictions, NOT verified counts',fill='black')
        for i,(im,m,dmap,row) in enumerate(zip(ims,a,dm,rows)):
            cmap=cvmap(im);dmapatt=m[ds[1]];cp=peaks(cmap,cs[2],cs[3]);dp=peaks(dmapatt,ds[2],ds[3]);cnt=float(dmap.sum());res.append(dict(path=row['path'],cv=len(cp),dino=len(dp),density=cnt,previous_count_cnn=row['count']))
            if i<2:
                for col,(title,mp) in enumerate([('Input',None),(f'CV peaks: {len(cp)}',cmap),(f'DINO peaks: {len(dp)}',dmapatt),(f'Density sum: {cnt:.2f}',dmap)]):
                    xx=col*250;yy=25+i*160;draw.text((xx+5,yy),title,fill='black');thumb=im.resize((240,90))
                    if mp is not None:
                        z=cv2.resize(mp,(240,90));z=(z-z.min())/(np.ptp(z)+1e-8);heat=cv2.cvtColor(cv2.applyColorMap(np.uint8(z*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB);thumb=Image.blend(thumb,Image.fromarray(heat),.55)
                    panel.paste(thumb,(xx+5,yy+20))
                    if col in (1,2):
                        pp=cp if col==1 else dp;ww=224 if col==1 else 28
                        for px in pp:
                            xpos=xx+5+int(px*240/ww);draw.line((xpos,yy+112,xpos,yy+125),fill='black',width=2)
        panel.save(OUT/(family+'.png'));panels.append(panel);results[family]=res
    for k in range(2):
        board=Image.new('RGB',(1000,1440),'white')
        for i,p in enumerate(panels[k*4:k*4+4]):board.paste(p,(0,i*360))
        board.save(OUT/f'comparison_{k+1}.png')
    write('real_inference.json',results);write('status.json',dict(state='complete',elapsed_seconds=time.time()-start,real_accuracy_measured=False,selected_density_step=saved['step'],density_sha256=sha(OUT/'density.pt'),pipeline_promoted=False));print('COMPLETE',flush=True)
if __name__=='__main__':main()
