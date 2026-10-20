from pathlib import Path
import numpy as np,torch
from controls import targets,decode,LOW,HIGH
from models import Joint
from corpus import ROOT,fonts,render
source=(ROOT/'feature_multilabel_v1/corpus.py').read_text()
source=source.replace('active=flags.copy()','active=flags.copy();observed={0:[],1:[],2:[]}')
source=source.replace('step=int(rr.integers(9,27))','step=int(rr.integers(9,27));observed[4]=step')
source=source.replace('if k==0:\n    pts=', 'if k==0:\n    observed[0].append(width)\n    pts=')
source=source.replace('amp=rr.uniform(9,24)','amp=rr.uniform(9,24);observed[1].append((width,amp))')
source=source.replace('rad=int(rr.integers(1,4));d.ellipse','rad=int(rr.integers(1,4));observed[2].append(rad);d.ellipse')
source=source.replace("if active[12]:im=im.filter(ImageFilter.GaussianBlur(float(rng(12).uniform(.8,1.5))))","if active[12]:\n  radius=float(rng(12).uniform(.8,1.5));observed[12]=radius;im=im.filter(ImageFilter.GaussianBlur(radius))")
source=source.replace('return np.array(im)','return np.array(im),observed');scope={};exec(source,scope)
bank=fonts()[:6];flags=np.ones(15,bool);flags[9]=False
for part in ['train','locked_recipe']:
 for seed in range(201000000,201000010):
  im,o=scope['render'](seed,flags,bank,part);assert np.array_equal(im,render(seed,flags,bank,part))
  expected=[len(o[0]),np.mean(o[0]),len(o[1]),np.mean([x[0] for x in o[1]]),np.mean([x[1] for x in o[1]]),len(o[2]),np.mean(o[2]),o[4],o[12]]
  assert np.allclose(LOW+targets(seed,part)*(HIGH-LOW),expected,atol=1e-5)
torch.set_num_threads(2)
for kind in ['gridded','conformer']:
 m=Joint(kind).cuda();x=torch.rand(2,3,96,192,device='cuda');a,b=m(x);assert a.shape==(2,15) and b.shape==(2,9);(a.square().mean()+b.square().mean()).backward();assert all(h.weight.grad.abs().sum()>0 for h in m.params)
 if kind=='conformer':assert all(v.weight.grad.abs().sum()>0 for v in m.to_local)
 print(kind,'parameters',sum(p.numel() for p in m.parameters()))
print('PASS exact rendered-control targets (20 replay cases), unchanged pixels, both joint output/gradient paths, bidirectional feedback gradients')
