"""Synthetic-only count classification versus class-agnostic set prediction."""
import os
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,time,hashlib,copy,random
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent
FONTROOT=ROOT.parent/'CAPRI_typography_v3'
torch.set_num_threads(4)
W,H=256,80
def save(name,value):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def sample(fonts,seed,n,shift=False):
 rng=np.random.default_rng(seed);palette=rng.integers(30,226,3);im=Image.new('RGB',(W,H),tuple(palette));d=ImageDraw.Draw(im)
 for _ in range(int(rng.integers(0,14))):
  x,y=rng.integers(W),rng.integers(H);d.rectangle((int(x),int(y),int(min(W,x+rng.integers(8,70))),int(min(H,y+rng.integers(4,60)))),fill=tuple(np.clip(palette+rng.integers(-35,36,3),0,255)))
 for _ in range(int(rng.integers(0,7))):d.line((0,int(rng.integers(H)),W,int(rng.integers(H))),fill=tuple(rng.integers(30,226,3)),width=int(rng.integers(1,3)))
 for _ in range(int(rng.integers(0,140))):d.point((int(rng.integers(W)),int(rng.integers(H))),fill=tuple(rng.integers(256,size=3)))
 span=float(rng.uniform(130,220));left=float(rng.uniform(15,W-span-15))
 if n<=1:xs=np.array([rng.uniform(30,W-30)])[:n]
 elif shift:
  # Held-out nonuniform gap allocation at the same count-independent span.
  gaps=rng.uniform(.5,1.5,n-1);xs=left+np.r_[0,np.cumsum(gaps)/gaps.sum()*span]
 else:xs=np.linspace(left,left+span,n)+rng.uniform(-2,2,n)
 boxes=[];masks=[];text='';widths=[]
 for x in xs:
  row=fonts[int(rng.integers(len(fonts)))];font=ImageFont.truetype(str(FONTROOT/row['path']),44);c=str(rng.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')));text+=c
  bb=font.getbbox(c);g=Image.new('L',(bb[2]-bb[0]+8,bb[3]-bb[1]+8));ImageDraw.Draw(g).text((4-bb[0],4-bb[1]),c,font=font,fill=255);g=g.crop(g.getbbox())
  # Width/height sampled independently of n, never width divided by count.
  gw=int(rng.integers(9,21));gh=int(rng.integers(24,49));widths.append(gw);g=g.resize((gw,gh),Image.Resampling.BILINEAR)
  if rng.random()<.13:g=g.filter(ImageFilter.FIND_EDGES)
  g=g.rotate(float(rng.uniform(-18,18)),Image.Resampling.BILINEAR,expand=True)
  xx=int(np.clip(round(x-g.width/2),1,W-g.width-1));yy=int(rng.integers(1,H-g.height))
  mask=Image.new('L',(W,H));mask.paste(g,(xx,yy));arr=np.array(im);support=np.array(mask)>100
  if not support.any():raise ValueError('empty glyph')
  mean=arr[support].mean();color=rng.integers(0,45,3) if mean>128 else rng.integers(211,256,3)
  im.paste(tuple(color),mask);masks.append(np.array(mask));boxes.append([(xx+g.width/2)/W,(yy+g.height/2)/H,g.width/W,g.height/H])
 for i,m in enumerate(masks):
  covered=np.maximum.reduce(masks[i+1:]) if i+1<len(masks) else np.zeros_like(m)
  if ((m>100)&(covered<100)).sum()/max(1,(m>100).sum())<.55:raise ValueError('occluded glyph')
 if rng.random()<.2:im=im.filter(ImageFilter.GaussianBlur(float(rng.uniform(.1,.5))))
 return np.array(im),np.array(boxes,dtype='float32').reshape(-1,4),dict(n=n,text=text,span=span,mean_requested_width=float(np.mean(widths)) if widths else None,seed=seed)
def dataset(fonts,count,seed,shift=False):
 images=[];boxes=[];meta=[]
 for i in range(count):
  n=i%11
  for retry in range(100):
   try:a,b,m=sample(fonts,seed+i*100+retry,n,shift);break
   except ValueError:continue
  else:raise RuntimeError('generation failed')
  images.append(a);boxes.append(torch.tensor(b));meta.append(m)
 return torch.from_numpy(np.stack(images)).permute(0,3,1,2).float()/255,boxes,meta
class Net(nn.Module):
 def __init__(self,instance=False):
  super().__init__();self.instance=instance;self.enc=nn.Sequential(nn.Conv2d(3,24,5,2,2),nn.GroupNorm(4,24),nn.SiLU(),nn.Conv2d(24,48,3,2,1),nn.GroupNorm(6,48),nn.SiLU(),nn.Conv2d(48,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU());self.count=nn.Sequential(nn.AdaptiveAvgPool2d((2,8)),nn.Flatten(),nn.Linear(1024,128),nn.SiLU(),nn.Dropout(.15),nn.Linear(128,11))
  if instance:
   self.pos=nn.Parameter(torch.randn(1,320,64)*.02);self.queries=nn.Parameter(torch.randn(1,12,64)*.02);self.dec=nn.TransformerDecoder(nn.TransformerDecoderLayer(64,4,128,.1,batch_first=True),2);self.obj=nn.Linear(64,1);self.box=nn.Sequential(nn.Linear(64,64),nn.SiLU(),nn.Linear(64,4))
 def forward(self,x):
  z=self.enc(x);count=self.count(z)
  if not self.instance:return count,None,None
  memory=z.flatten(2).transpose(1,2)+self.pos;q=self.dec(self.queries.expand(x.shape[0],-1,-1),memory)
  return count,self.obj(q).squeeze(-1),self.box(q).sigmoid()
def loss(outputs,y,targets,device):
 count,obj,box=outputs;l=F.cross_entropy(count,y)
 if obj is None:return l
 truth=torch.zeros_like(obj);reg=box.sum()*0
 for i,t in enumerate(targets):
  if not len(t):continue
  t=t.to(device);cost=5*torch.cdist(box[i],t,p=1)-obj[i].sigmoid()[:,None];a,b=linear_sum_assignment(cost.detach().cpu().numpy());truth[i,a]=1;reg=reg+F.l1_loss(box[i,a],t[b])
 return l+F.binary_cross_entropy_with_logits(obj,truth)+5*reg/len(y)
@torch.no_grad()
def predict(net,x,device):
 net.eval();ps=[];os=[]
 for i in range(0,len(x),64):
  c,o,b=net(x[i:i+64].to(device));ps.append(c.softmax(-1).cpu().numpy())
  if o is not None:os.append(o.sigmoid().cpu().numpy())
 return np.concatenate(ps),np.concatenate(os) if os else None
def metrics(pred,truth):
 return dict(exact=float(np.mean(pred==truth)),mae=float(np.abs(pred-truth).mean()),per_count={str(n):float(np.mean(pred[truth==n]==n)) for n in np.unique(truth)},n=len(truth))
def main():
 started=time.time();device='cuda' if torch.cuda.is_available() else 'cpu';fonts=json.loads((FONTROOT/'fonts.json').read_text());order=np.random.default_rng(510).permutation(len(fonts));trainfonts=[fonts[i] for i in order[:25]];heldfonts=[fonts[i] for i in order[25:]]
 protocol=dict(training='synthetic only, no target references or labels',models=['count_cnn','instance_transformer'],seeds=[731,732,733],train=3300,selection=440,calibration=440,test_each=440,steps=1200,batch=32,selection_every=100,counts=list(range(11)),primary_counts=list(range(2,9)),train_fonts=[f['family'] for f in trainfonts],held_fonts=[f['family'] for f in heldfonts],checkpoint_selection='minimum count-head MAE on synthetic selection; exact as tie breaker',instance_threshold='minimum calibration count MAE over 0.1..0.9',device=device,real_fit_only=True,real_accuracy='not measured without independent count labels',limitations='bounded pilot, both heads use learned count-context; test real predictions not ground truth')
 save('protocol.json',protocol);save('status.json',dict(state='building_synthetic',time=time.time()))
 data={}
 for name,n,seed,ff,shift in [('train',3300,11000000,trainfonts,False),('selection',440,21000000,trainfonts,False),('calibration',440,31000000,trainfonts,False),('test',440,41000000,trainfonts,False),('font_test',440,51000000,heldfonts,False),('layout_test',440,61000000,trainfonts,True)]:
  data[name]=dataset(ff,n,seed,shift);save(name+'_metadata.json',data[name][2]);save('status.json',dict(state='building_synthetic',completed=name,time=time.time()))
 # Parent pixel hashes explicitly checked across partitions.
 hashes={};integrity={}
 for name,(x,b,meta) in data.items():
  hs=[hashlib.sha256(im.numpy().tobytes()).hexdigest() for im in x]
  assert len(hs)==len(set(hs));assert not(set(hs)&set(hashes));hashes.update({h:name for h in hs});integrity[name]=hs
 save('pixel_hashes.json',integrity)
 canvas=Image.new('RGB',(W*4,110*6),'white');d=ImageDraw.Draw(canvas)
 for i in range(24):
  im=Image.fromarray((data['train'][0][i].permute(1,2,0).numpy()*255).round().astype('uint8'));xx=i%4*W;yy=i//4*110;canvas.paste(im,(xx,yy+20));d.text((xx,yy),'count '+str(data['train'][2][i]['n']),fill='black')
 canvas.save(ROOT/'synthetic_preview.png');results={}
 for kind in protocol['models']:
  for seed in protocol['seeds']:
   torch.manual_seed(seed);rng=np.random.default_rng(seed);net=Net(kind=='instance_transformer').to(device);opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.02);best=None;history=[]
   x,boxes,meta=data['train'];truth=np.array([m['n'] for m in meta]);sy=np.array([m['n'] for m in data['selection'][2]])
   for step in range(1,protocol['steps']+1):
    net.train();ids=rng.integers(len(x),size=32);y=torch.tensor(truth[ids],device=device);opt.zero_grad();l=loss(net(x[ids].to(device)),y,[boxes[i] for i in ids],device);l.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
    if step%100==0:
     pp,_=predict(net,data['selection'][0],device);m=metrics(pp.argmax(1),sy);key=(m['mae'],-m['exact'])
     if best is None or key<best[0]:best=(key,copy.deepcopy(net.state_dict()),step)
     history.append(dict(step=step,loss=float(l),selection=m));save(f'{kind}_{seed}_history.json',history);save('status.json',dict(state='training',model=kind,seed=seed,step=step,steps=1200,selection=m,elapsed=time.time()-started));print(kind,seed,step,round(m['exact'],3),flush=True)
   net.load_state_dict(best[1]);torch.save(best[1],ROOT/f'{kind}_{seed}.pt');cp,co=predict(net,data['calibration'][0],device);cy=np.array([m['n'] for m in data['calibration'][2]]);threshold=None
   if co is not None:threshold=min(np.linspace(.1,.9,81),key=lambda t:np.abs((co>=t).sum(1)-cy).mean())
   row=dict(selected_step=best[2],object_threshold=float(threshold) if threshold is not None else None,tests={})
   for name in ['test','font_test','layout_test']:
    pp,oo=predict(net,data[name][0],device);yy=np.array([m['n'] for m in data[name][2]]);row['tests'][name]=dict(count_head=metrics(pp.argmax(1),yy));np.savez_compressed(ROOT/f'{kind}_{seed}_{name}.npz',probabilities=pp,object_probabilities=oo if oo is not None else np.zeros((0,)),truth=yy)
    if oo is not None:row['tests'][name]['instances']=metrics((oo>=threshold).sum(1),yy)
   results[f'{kind}_{seed}']=row;save('results.json',results)
 # Real inference restricted to unique cached Phase1 fit-reference paths; no
 # filenames decoded, no labels read, no real-driven checkpoint selection.
 real={}
 for family in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
  profile=json.loads((ROOT.parent/'CAPRI_v2/profiles/conformer'/f'{family}.json').read_text());paths=list(dict.fromkeys(e['reference_path_provenance'] for e in profile['entries']))[:64];ims=[];valid=[]
  for path in paths:
   pp=Path(path)
   if 'eval' in [p.lower() for p in pp.parts]:raise ValueError('Evaluation directory in fit manifest')
   im=Image.open(pp).convert('RGB').resize((W,H));ims.append(np.array(im));valid.append(path)
  xx=torch.from_numpy(np.stack(ims)).permute(0,3,1,2).float()/255;familyrows=[]
  for kind in protocol['models']:
   for seed in protocol['seeds']:
    net=Net(kind=='instance_transformer').to(device);net.load_state_dict(torch.load(ROOT/f'{kind}_{seed}.pt',weights_only=True,map_location=device));pp,oo=predict(net,xx,device);threshold=results[f'{kind}_{seed}']['object_threshold'];familyrows.append(dict(model=kind,seed=seed,count_probabilities=pp.tolist(),instance_counts=(oo>=threshold).sum(1).tolist() if oo is not None else None))
  real[family]=dict(paths=valid,models=familyrows,ground_truth_used=False)
 save('real_inference.json',real);save('status.json',dict(state='complete',elapsed=time.time()-started,real_accuracy_verified=False));print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:save('failure.json',dict(error=repr(e),time=time.time()));raise

