"""Paper-based Meta-Sim2 adaptation: masked grammar GRU + KDE REINFORCE.

Not official Meta-Sim2 code. No target strings, masks, boxes or hybrid predictions.
"""
from pathlib import Path
import sys, copy
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'controller_mvp_v7_dino'))
from setup7 import U,WORK,FAMILIES,random_recipes
import render7 as R
from run import load_images,save_rows
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from PIL import Image

FIELDS=dict(length=list(range(2,9)),charset=['digits','upper','lower','upper_digits','lower_digits','mixed','letters'],
 font=[],layout=['line','curve','two_layer'],size_absolute=[.4,.55,.7,.85],
 width_absolute=[.65,.85,1.,1.2],spacing_absolute=[-.15,0.,.15,.3],
 rotation_absolute=[0.,10.,25.,40.],jitter_absolute=[0.,.08,.16],
 weight_absolute=[0.,.5,1.],bold_absolute=[0,1],slant_absolute=[-.25,0.,.25],
 size_jitter_absolute=[0.,.15,.3],decoration=[(0,0),(1,0),(2,0),(0,1),(1,1)],
 background=list(range(8)),color_mode=['single','alternating','random'],ink=list(range(8)))

def grammar(fonts):
    fields=copy.deepcopy(FIELDS);fields['font']=[x['key'] for x in fonts]
    effects=list(R.C.R.R.SAMPLED)
    return dict(fields=fields,effects=effects,max_effects=3)

class Policy(nn.Module):
    def __init__(self,g):
        super().__init__();self.g=g;self.names=list(g['fields']);self.sizes=[len(x) for x in g['fields'].values()]+[1+len(g['effects'])*2]*g['max_effects']
        self.offsets=np.cumsum([0]+self.sizes).tolist();self.embedding=nn.Embedding(sum(self.sizes)+1,48)
        self.gru=nn.GRUCell(48,96);self.heads=nn.ModuleList([nn.Linear(96,n) for n in self.sizes])
        self.start=nn.Parameter(torch.zeros(96))
    def forward(self,n,teacher=None):
        device=self.start.device;h=self.start[None].expand(n,-1);inp=self.embedding.weight[-1][None].expand(n,-1)
        logs=[];entropies=[];actions=[];alive=torch.ones(n,dtype=torch.bool,device=device);used=torch.zeros(n,len(self.g['effects']),dtype=torch.bool,device=device)
        for i,head in enumerate(self.heads):
            h=self.gru(inp,h);logits=head(h);mask=torch.ones_like(logits,dtype=torch.bool)
            if i>=len(self.names):
                mask[:,1:]=(~used).repeat_interleave(2,dim=1)&alive[:,None]
                logits=logits.masked_fill(~mask,-1e9)
            dist=torch.distributions.Categorical(logits=logits)
            a=dist.sample() if teacher is None else teacher[:,i]
            logs.append(dist.log_prob(a));entropies.append(dist.entropy());actions.append(a)
            if i>=len(self.names):
                active=alive&(a>0)
                used=used.clone();used[torch.arange(n,device=device)[active],((a[active]-1)//2)]=True
                alive=alive&(a>0)
            inp=self.embedding(a+self.offsets[i])
        return torch.stack(actions,1),torch.stack(logs,1).sum(1),torch.stack(entropies,1).sum(1)

def prior(g,n,rng):
    a=[]
    for k,v in g['fields'].items():
        p=None
        if k=='layout':p=[.85,.1,.05]
        if k=='decoration':p=[.65,.12,.08,.1,.05]
        a.append(rng.choice(len(v),n,p=p))
    alive=np.ones(n,bool);used=[set() for _ in range(n)]
    for _ in range(g['max_effects']):
        x=np.zeros(n,dtype=int)
        for j in range(n):
            if alive[j] and rng.random()<.5:
                options=[k for k in range(len(g['effects'])) if k not in used[j]]
                e=int(rng.choice(options));used[j].add(e);x[j]=1+2*e+int(rng.integers(2))
            else:alive[j]=False
        a.append(x)
    return torch.tensor(np.stack(a,1),dtype=torch.long)

def pixel_palette(images):
    # Fit pixels only; no text localization or annotation. Most frequent colors first.
    arr=np.concatenate([np.asarray(Image.fromarray(x).resize((128,32))) for x in images],axis=0)
    q=Image.fromarray(arr).quantize(colors=8);pal=q.getpalette();counts=sorted(q.getcolors(),reverse=True)
    colors=[pal[3*i:3*i+3] for _,i in counts]
    while len(colors)<8:colors.append(colors[-1])
    return colors

def decode(g,action,colors):
    p=R.expand({});p['effects']={k:0 for k in g['effects']}
    for i,(name,vals) in enumerate(g['fields'].items()):
        value=vals[int(action[i])]
        if name=='decoration':p['glyph_outline'],p['frame_width']=value
        elif name=='background':p[name]=colors[value]
        elif name=='ink':
            p['text_palette']=[colors[value]] if p['color_mode']=='single' else [colors[value],*colors[1:]]
            p['effect_palette']=colors[1:]
        else:p[name]=value
    for a in action[len(g['fields']):]:
        if not int(a):break
        p['effects'][g['effects'][(int(a)-1)//2]]=[2,5][(int(a)-1)%2]
    return p

class Features:
    def __init__(self):
        torch.set_num_threads(4);self.device='cuda' if torch.cuda.is_available() else 'cpu'
        m=resnet18(weights=None);m.load_state_dict(torch.load(ROOT/'resnet18-f37072fd.pth',weights_only=True,map_location='cpu'))
        self.net=nn.Sequential(*list(m.children())[:-2]).to(self.device).eval().requires_grad_(False)
    @torch.inference_mode()
    def extract(self,images):
        out=[]
        for i in range(0,len(images),24):
            x=torch.tensor(np.stack([np.asarray(Image.fromarray(im).resize((256,64))) for im in images[i:i+24]]).transpose(0,3,1,2).copy(),device=self.device).float()/255
            z=self.net((x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None])
            # Spatial ResNet features retain layout; pixels retain color and fine strokes.
            a=F.adaptive_avg_pool2d(z,(1,4)).flatten(1)
            b=F.adaptive_avg_pool2d(x,(8,32)).flatten(1)
            edge=(x[:,:,:,1:]-x[:,:,:,:-1]).abs()
            c=F.adaptive_avg_pool2d(edge,(8,32)).flatten(1)
            out.append(torch.cat([a,b,c],1).cpu().numpy())
        return np.concatenate(out)

class Density:
    def __init__(self,target,calibration):
        both=np.concatenate([target,calibration]);self.mean=both.mean(0);self.std=np.maximum(both.std(0),.03)
        self.weights=np.concatenate([np.full(2048,np.sqrt(.5/2048)),np.full(768,np.sqrt(.3/768)),np.full(768,np.sqrt(.2/768))])
    def transform(self,x):return ((x-self.mean)/self.std*self.weights).astype(np.float32)
    @staticmethod
    def cost(q,p):
        # Gaussian KDE H=dI after weighted standardization expressed in unit bandwidth.
        # Common Gaussian constants cancel. Includes each synthetic sample itself, as paper.
        q=torch.as_tensor(q);p=torch.as_tensor(p)
        logq=torch.logsumexp(-torch.cdist(q,q).square()/2,dim=1)-np.log(len(q))
        logp=torch.logsumexp(-torch.cdist(q,p).square()/2,dim=1)-np.log(len(p))
        return (logq-logp).numpy()

def render_batch(pool,recipes,controls):
    rows=list(pool.map(R.safe_render,list(zip(recipes,controls))))
    for row,r in zip(rows,recipes):
        if 'image' not in row:
            w,h=r['size'];row.update(image=np.full((h,w,3),255,np.uint8),mask=np.zeros((h,w),np.uint8),seed=r['seed'])
    return rows

def penalty(rows):return np.asarray([2*(not r['complete'])+1*(not r['usable']) for r in rows],np.float32)
