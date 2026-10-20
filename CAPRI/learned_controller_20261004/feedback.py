import json,sys,colorsys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from scipy import ndimage
import scheme as S
from network import Encoder,Controller
from feature_bank.iter2.model import Iter2Model
from feature_bank.iter2.data import MARGIN
class Models:
 def __init__(self):
  torch.set_num_threads(4);self.enc=Encoder().cuda().eval();self.enc.load_state_dict(torch.load(S.ROOT/'training/encoder.pt',weights_only=False)['model'])
  self.policy=Controller().cuda().eval();self.policy.load_state_dict(torch.load(S.ROOT/'training/controller.pt',weights_only=False)['model'])
  self.hybrid=Iter2Model().cuda().eval();self.hybrid.load_state_dict(torch.load('E:/pranav/Final sem/sourcefree_synth/runs/iter2_baseline_15_20261003_04/baseline_training/epoch_15/ema.pt',weights_only=True))
  self.hook=self.hybrid.txt_pool.register_forward_hook(lambda m,a,o:setattr(self,'queries',o))
  self.enc.requires_grad_(False);self.hybrid.requires_grad_(False);self.policy.requires_grad_(False)
 @torch.inference_mode()
 def extract(self,images):
  result={k:[] for k in ['style','font','color','layout','content','cont','cat','mask']}
  for start in range(0,len(images),32):
   pixels=np.stack([np.asarray(Image.fromarray(im).resize((256,64),Image.Resampling.BILINEAR)) for im in images[start:start+32]])
   x=torch.from_numpy(pixels.transpose(0,3,1,2).copy()).cuda().float()/255
   with torch.autocast('cuda',dtype=torch.bfloat16):o=self.enc(x)
   xx=F.interpolate(x,size=(96,384),mode='bilinear',align_corners=False);pad=((384+MARGIN+7)//8)*8;xx=F.pad(xx,(0,pad-384,0,0),mode='replicate')
   with torch.autocast('cuda',dtype=torch.bfloat16):self.hybrid(xx,torch.full((len(xx),),48,device='cuda',dtype=torch.long))
   fq=F.normalize(self.queries[:,-7:].float(),dim=-1).flatten(1)/np.sqrt(7)
   probs=[v.float().softmax(1).cpu().numpy() for v in o['cat']]
   masks=o['mask'].float().sigmoid().cpu().numpy()[:,0]
   result['style'].append(o['z'].float().cpu().numpy());result['font'].append(fq.cpu().numpy());result['cont'].append(o['cont'].float().cpu().numpy())
   result['cat'].append(np.concatenate(probs,1));result['content'].append(np.concatenate(probs[1:3],1));result['mask'].append(masks)
   color=[];layout=[]
   for im,mask in zip(pixels/255,masks):
    gray=im@np.array([.299,.587,.114]);sat=im.max(2)-im.min(2);features=[]
    for region in [np.ones(mask.shape,bool),mask>.5,mask<.15]:
     values=im[region] if region.sum()>8 else im.reshape(-1,3)
     features.extend(np.quantile(values,[.1,.5,.9],axis=0).ravel())
     ss=values.max(1)-values.min(1);features.extend(np.quantile(ss,[.1,.5,.9]))
    features.extend([(gray>.94).mean(),(gray<.15).mean(),sat.mean()]);color.append(features)
    mm=(mask>.5).astype(float);gx=np.abs(np.diff(gray,axis=1));gy=np.abs(np.diff(gray,axis=0))
    ll,n=ndimage.label(mm);boxes=ndimage.find_objects(ll);hh=[];ww=[]
    for sl in boxes:
     if sl is not None and sl[0].stop-sl[0].start>=8 and sl[1].stop-sl[1].start>=2:
      hh.append((sl[0].stop-sl[0].start)/64);ww.append((sl[1].stop-sl[1].start)/256)
    layout.append(np.r_[mm.mean(),mm.mean(0).reshape(16,16).mean(1),mm.mean(1).reshape(8,8).mean(1),np.quantile(hh,[.25,.5,.75]) if hh else [0,0,0],np.quantile(ww,[.25,.5,.75]) if ww else [0,0,0],gx.mean(),gy.mean()])
   result['color'].append(np.array(color));result['layout'].append(np.array(layout))
  return {k:np.concatenate(v) for k,v in result.items()}
 @torch.inference_mode()
 def propose(self,target,current,p):
  c,k=S.encode(p);o=self.policy(torch.tensor(target['style'].mean(0)[None],device='cuda'),torch.tensor(current['style'].mean(0)[None],device='cuda'),torch.tensor(c[None],device='cuda'),torch.tensor(k[None],device='cuda'))
  return S.decode(o['cont'][0].cpu().numpy(),np.array([v[0].argmax().item() for v in o['cat']]))
def energy(x,y):
 from scipy.spatial.distance import cdist
 return float(max(0,2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean())/np.sqrt(x.shape[1]))
CHANNELS={'style':.2,'font':.2,'color':.25,'layout':.2,'content':.15}
class Objective:
 def __init__(self,real,initial):
  self.scales={k:max(.005 if k in ('style','font') else .015,energy(initial[k],real[k])) for k in CHANNELS}
 def components(self,f,real):return {k:energy(f[k],real[k]) for k in CHANNELS}
 def score(self,f,real,rows):
  comp=self.components(f,real);appearance=sum(CHANNELS[k]*comp[k]/self.scales[k] for k in CHANNELS)
  complete=float(np.mean([r['complete'] for r in rows]));usable=float(np.mean([r['usable'] for r in rows]));strict=float(np.mean([r['strict'] for r in rows]))
  # Readability is a rejection penalty, never a reward for adding arbitrary effects.
  return dict(components=comp,appearance=appearance,complete=complete,usable=usable,strict=strict,objective=appearance+4*(1-complete)+2*(1-usable))
def palette_proposal(p,images,features):
 p=p.copy();background=[];foreground=[];bgall=[];fgall=[]
 for image,mask in zip(images,features['mask']):
  im=np.asarray(Image.fromarray(image).resize((256,64)),float)/255
  bg=im[mask<.15];fg=im[mask>.65]
  if len(bg)>20:
   # Modal background cluster avoids averaging black text and white background to gray.
   quant=(bg*15).astype(int);keys=quant[:,0]*256+quant[:,1]*16+quant[:,2];idx=np.bincount(keys,minlength=4096).argmax();v=np.median(bg[keys==idx],axis=0);background.append(colorsys.rgb_to_hsv(*v));bgall.append(v)
  if len(fg)>10:
   v=np.median(fg,axis=0);foreground.append(colorsys.rgb_to_hsv(*v));fgall.append(v)
 for prefix,values in [('bg',background),('fg',foreground)]:
  if not values:continue
  vals=np.array(values);p[prefix+'_h']=float(np.arctan2(np.sin(vals[:,0]*2*np.pi).mean(),np.cos(vals[:,0]*2*np.pi).mean())/(2*np.pi)%1)
  p[prefix+'_s']=float(np.median(vals[:,1]));p[prefix+'_v']=float(np.median(vals[:,2]));p[prefix+'_hspread']=float(np.clip(2*np.std(vals[:,0]),0,1))
 for name,(lo,hi) in S.CONT.items():p[name]=float(np.clip(p[name],lo,hi))
 return p
