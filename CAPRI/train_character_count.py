"""Synthetic-only count pilot: no OCR, real labels, VLM, or character-type head."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json,time,copy
from pathlib import Path
import numpy as np
import cv2,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont,ImageFilter
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');sys.path.insert(0,str(ROOT))
from feature_bank.core import read,write,new_run,load,digest
from style_pipeline.font_match import encode,font_at
from style_pipeline.measure import extract_style

def synthetic(fonts,seed,n):
    rng=np.random.default_rng(seed);w=224;h=72;font=font_at(fonts[int(rng.integers(len(fonts)))]['path'],48)
    text=''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789'),n))
    masks=[];canvas=np.zeros((h,w),np.uint8);span=float(rng.uniform(.65,.88));height=int(rng.integers(25,51));centers=np.linspace(w*(1-span)/2,w*(1+span)/2,n)
    for i,char in enumerate(text):
        box=font.getbbox(char);g=Image.new('L',(box[2]-box[0]+8,box[3]-box[1]+8));ImageDraw.Draw(g).text((4-box[0],4-box[1]),char,font=font,fill=255);g=g.crop(g.getbbox())
        gh=int(np.clip(height+rng.integers(-5,6),20,55));gw=max(5,min(int(g.width*gh/g.height*rng.uniform(.7,1.15)),int(w/(n+1)*1.05)))
        g=g.resize((gw,gh),Image.Resampling.BILINEAR)
        if rng.random()<.2:
            a=np.pad(np.asarray(g),2);g=Image.fromarray(cv2.dilate(a,np.ones((3,3),np.uint8))-a)
        g=g.rotate(float(rng.uniform(-20,20)),Image.Resampling.BILINEAR,expand=True)
        scale=min(1.,(h-4)/g.height,(w-4)/g.width)
        if scale<1:g=g.resize((max(1,int(g.width*scale)),max(1,int(g.height*scale))),Image.Resampling.BILINEAR)
        x=int(np.clip(centers[i]-g.width/2+rng.uniform(-2,2),0,w-g.width));y=int(np.clip((h-g.height)/2+rng.uniform(-6,6),0,h-g.height))
        assert x>=0 and y>=0 and x+g.width<=w and y+g.height<=h
        a=np.zeros_like(canvas);a[y:y+g.height,x:x+g.width]=np.asarray(g);masks.append(a);canvas=np.maximum(canvas,a)
    bg=float(rng.uniform(150,255));fg=float(rng.uniform(0,90));ramp=np.linspace(-rng.uniform(0,25),rng.uniform(0,25),w)[None,:];alpha=canvas/255
    a=(bg+ramp)*(1-alpha)+fg*alpha+rng.normal(0,rng.uniform(0,10),(h,w));im=Image.fromarray(np.clip(a,0,255).astype('uint8'));d=ImageDraw.Draw(im)
    for _ in range(int(rng.integers(5))):d.line((0,int(rng.integers(h)),w-1,int(rng.integers(h))),fill=int(rng.integers(80,200)),width=int(rng.integers(1,3)))
    for _ in range(int(rng.integers(250))):d.point((int(rng.integers(w)),int(rng.integers(h))),fill=int(rng.integers(80,220)))
    if rng.random()<.3:im=im.filter(ImageFilter.GaussianBlur(float(rng.uniform(.2,.7))))
    if rng.random()<.3:im=Image.fromarray(255-np.asarray(im))
    return im,text

class CountCNN(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(2,24,5,2,2),nn.ReLU(),nn.Conv2d(24,48,3,2,1),nn.ReLU(),nn.Conv2d(48,96,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,16)),nn.Flatten(),nn.Linear(6144,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,7))
    def forward(self,x):return self.net(x)

def main():
    out=new_run(Path(__file__).resolve().parent/'character_count_02');start=time.time();torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(2026)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    fonts=read(ROOT/'runs/style_phase3_font_05/fonts.json')
    for f in fonts:assert digest(f['path'])==f['sha256']
    write(out/'protocol.json',dict(task='count only; supported2to8',real_training=False,fonts_train=[f['family'] for f in fonts[:10]],fonts_heldout=[f['family'] for f in fonts[10:]],steps=1200,checkpoint_selection='minimum synthetic selection cross entropy every100steps',confidence='temperature and optional90percent precision threshold on separate synthetic calibration',real_claim='unverified predictions only, no ground truth labels',automation='remains paused'))
    (out/'script_snapshot.py').write_bytes(Path(__file__).read_bytes());arrays={};membership={};texts=set()
    preview=Image.new('RGB',(896,504),'white')
    for part,base,count,bank in [('train',2000000,256,fonts[:10]),('selection',3000000,64,fonts[:10]),('calibration',4000000,64,fonts[:10]),('test',5000000,64,fonts[:10]),('heldout_fonts',6000000,64,fonts[10:])]:
        xx=[];yy=[];rows=[]
        for n in range(2,9):
            for i in range(count):
                seed=base+n*10000+i;im,text=synthetic(bank,seed,n)
                # Content overlaps (especially length2) are recorded rather than
                # mislabeled as globally disjoint strings; seeds are disjoint.
                rows.append(dict(seed=seed,count=n,text=text));xx.append(encode(im));yy.append(n-2)
                if part=='train' and i<4:preview.paste(im,(224*i,72*(n-2)))
        arrays[part]=(torch.tensor(np.stack(xx),device='cuda'),torch.tensor(yy,device='cuda'));membership[part]=rows;print('Built',part,len(xx),flush=True)
    preview.save(out/'synthetic_examples.png');write(out/'membership.json',membership)
    model=CountCNN().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.0001);rng=np.random.default_rng(992);best=float('inf');history=[]
    def logits(part):
        model.eval();x,y=arrays[part]
        with torch.no_grad():return torch.cat([model(x[i:i+128]) for i in range(0,len(x),128)]),y
    for step in range(1,1201):
        model.train();x,y=arrays['train'];ids=rng.integers(len(x),size=64);loss=F.cross_entropy(model(x[ids]),y[ids]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%100==0:
            a,y=logits('selection');v=float(F.cross_entropy(a,y));acc=float((a.argmax(1)==y).float().mean());history.append(dict(step=step,train_loss=float(loss.detach()),selection_loss=v,selection_accuracy=acc));write(out/'training.json',history)
            if v<best:best=v;torch.save(dict(state_dict=model.state_dict(),step=step,counts=list(range(2,9))),out/'count_model.pt')
            print(step,acc,flush=True)
    saved=torch.load(out/'count_model.pt',weights_only=True);model.load_state_dict(saved['state_dict']);model.eval();a,y=logits('calibration');temps=np.linspace(.5,3,51);temp=float(min(temps,key=lambda t:float(F.cross_entropy(a/t,y))));p=(a/temp).softmax(1);conf,guess=p.max(1);eligible=[]
    for t in np.linspace(.5,.99,100):
        mask=conf>=t
        if int(mask.sum())>=30 and float((guess[mask]==y[mask]).float().mean())>=.9:eligible.append(float(t))
    threshold=min(eligible) if eligible else None;metrics={}
    for part in ('test','heldout_fonts'):
        a,y=logits(part);p=(a/temp).softmax(1);conf,guess=p.max(1);matrix=np.zeros((7,7),int)
        for truth,pred in zip(y.cpu().tolist(),guess.cpu().tolist()):matrix[truth,pred]+=1
        mask=conf>=threshold if threshold is not None else torch.zeros_like(conf,dtype=torch.bool)
        metrics[part]=dict(accuracy=float((guess==y).float().mean()),mae=float((guess-y).abs().float().mean()),confusion=matrix.tolist(),accepted=int(mask.sum()),accepted_accuracy=float((guess[mask]==y[mask]).float().mean()) if mask.any() else None)
    write(out/'test_results.json',dict(selected_step=saved['step'],temperature=temp,confidence_threshold=threshold,metrics=metrics));frozen=digest(out/'count_model.pt')
    families={}
    for name in ('cryptbb','city','courier','darkbay','outlined_text_captcha'):families[name]=read(ROOT/'runs/style_phase1_05'/f'{name}_measurements.json')['fit'][:32]
    families['roncuevas']=read(ROOT/'runs/style_phase1_unseen_01/split.json')['fit'][:32]
    for name,folder in [('pymlex','style_phase3_new_family_01'),('sloth','style_phase3_new_family_02')]:families[name]=[r for r in read(ROOT/'runs'/folder/'predictions.json') if r['partition']=='train']
    results={};gallery=Image.new('RGB',(1000,1120),'white');draw=ImageDraw.Draw(gallery)
    for j,(name,rows) in enumerate(families.items()):
        ims=[load(r['path']) for r in rows];x=torch.tensor(np.stack([encode(im) for im in ims]),device='cuda')
        with torch.no_grad():p=(model(x)/temp).softmax(1).cpu().numpy()
        predictions=[]
        for i,(im,prob) in enumerate(zip(ims,p)):
            guess=int(prob.argmax()+2);confidence=float(prob.max());cv=extract_style(im)
            predictions.append(dict(path=rows[i]['path'],count=guess,confidence=confidence,accepted=threshold is not None and confidence>=threshold,probabilities=prob.tolist(),cv_glyph_proposals=cv['glyph_count_estimate']))
            if i<4:
                thumb=im.copy();thumb.thumbnail((235,85));gallery.paste(thumb,(10+250*i,140*j+25));draw.text((10+250*i,140*j+115),f'count {guess}, score {confidence:.2f}',fill='black')
        draw.text((10,j*140+5),name+' | unverified real predictions',fill='black');results[name]=dict(mean_probability=p.mean(0).tolist(),modal_prediction=int(np.bincount(p.argmax(1),minlength=7).argmax()+2),predictions=predictions)
    gallery.save(out/'real_predictions.png');write(out/'real_inference.json',results);assert digest(out/'count_model.pt')==frozen
    write(out/'status.json',dict(state='TRAINED_TESTED_INFERRED',elapsed_seconds=time.time()-start,model_sha256=frozen,real_accuracy_measured=False,generation_integration=False));print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
