import torch
from torch import nn
from torch.nn import functional as F
import scheme as S
def block(a,b,stride=1):return nn.Sequential(nn.Conv2d(a,b,3,stride,1,bias=False),nn.BatchNorm2d(b),nn.SiLU(),nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.SiLU())
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.b0=block(3,24,2);self.b1=block(24,48,2);self.b2=block(48,96,2);self.b3=block(96,128,2)
  self.fc=nn.Sequential(nn.Linear(128*2*8,256),nn.SiLU(),nn.Linear(256,128))
  self.cont=nn.Linear(128,len(S.CN));self.cat=nn.ModuleList([nn.Linear(128,n) for n in S.CAT.values()])
  self.seg=nn.Sequential(nn.Conv2d(96+24,48,3,padding=1),nn.SiLU(),nn.Conv2d(48,1,1))
 def forward(self,x):
  a=self.b0(x);b=self.b1(a);c=self.b2(b);d=self.b3(c)
  z=self.fc(F.adaptive_avg_pool2d(d,(2,8)).flatten(1))
  mask=self.seg(torch.cat([a,F.interpolate(c,size=a.shape[-2:],mode='bilinear',align_corners=False)],1))
  return dict(z=F.normalize(z,dim=1),cont=self.cont(z).sigmoid(),cat=[h(z) for h in self.cat],mask=F.interpolate(mask,size=x.shape[-2:],mode='bilinear',align_corners=False))
class Controller(nn.Module):
 def __init__(self):
  super().__init__();d=256+len(S.CN)+sum(S.CAT.values())
  self.body=nn.Sequential(nn.Linear(d,384),nn.SiLU(),nn.Linear(384,256),nn.SiLU())
  self.delta=nn.Linear(256,len(S.CN));self.cat=nn.ModuleList([nn.Linear(256,n) for n in S.CAT.values()])
 def forward(self,target,current,c,k):
  one=[F.one_hot(k[:,i],n).float() for i,n in enumerate(S.CAT.values())]
  h=self.body(torch.cat([target,current,c,*one],1))
  return dict(cont=(c+.65*self.delta(h).tanh()).clamp(0,1),cat=[head(h)+old for head,old in zip(self.cat,one)])
