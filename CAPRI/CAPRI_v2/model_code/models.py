import torch
from torch import nn
from torch.nn import functional as F
from controls import OWNERS
class Joint(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  self.stem=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU())
  self.position=nn.Parameter(torch.randn(1,30,96)*.02);self.input=nn.Linear(64,96)
  self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(96,4,384,.2,activation='gelu',batch_first=True,norm_first=True) for _ in range(3)])
  if kind=='gridded':
   self.pseudo=nn.Parameter(torch.randn(1,1,96)*.02);self.encode=nn.MultiheadAttention(96,4,.1,batch_first=True)
  else:
   self.locals=nn.ModuleList([nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU()) for _ in range(3)])
   self.to_local=nn.ModuleList([nn.Conv2d(96,64,1) for _ in range(3)]);self.to_global=nn.ModuleList([nn.Linear(64,96) for _ in range(3)]);self.gates=nn.Parameter(torch.full((3,),.1))
  self.norm=nn.LayerNorm(96);self.attn=nn.Linear(96,15);self.heads=nn.ModuleList([nn.Linear(96,1) for _ in range(15)]);self.params=nn.ModuleList([nn.Linear(96,1) for _ in OWNERS])
 def forward(self,x):
  local=self.stem(x);b=local.shape[0]
  if self.kind=='gridded':
   # 24x48 native map -> padded24x50, thirty neighbourhoods of8x5.
   cells=F.pad(local,(0,2,0,0),mode='replicate').reshape(b,64,3,8,10,5).permute(0,2,4,3,5,1).reshape(b*30,40,64)
   encoded=self.input(cells);q=self.pseudo.expand(b*30,-1,-1);tokens=self.encode(q,encoded,encoded,need_weights=False)[0].reshape(b,30,96)+self.position
   for block in self.blocks:tokens=block(tokens)
  else:
   tokens=self.input(F.adaptive_avg_pool2d(local,(3,10)).flatten(2).transpose(1,2))+self.position
   for i,block in enumerate(self.blocks):
    local=local+self.locals[i](local)
    tokens=block(tokens+self.to_global[i](F.adaptive_avg_pool2d(local,(3,10)).flatten(2).transpose(1,2)))
    feedback=self.to_local[i](tokens.transpose(1,2).reshape(b,96,3,10));local=local+self.gates[i].tanh()*F.interpolate(feedback,size=local.shape[-2:],mode='bilinear',align_corners=False)
   tokens=tokens+self.input(F.adaptive_avg_pool2d(local,(3,10)).flatten(2).transpose(1,2))
  tokens=self.norm(tokens);pooled=torch.einsum('btf,btd->bfd',self.attn(tokens).softmax(1),tokens)
  return torch.cat([h(pooled[:,i]) for i,h in enumerate(self.heads)],1),torch.cat([h(pooled[:,f]) for h,f in zip(self.params,OWNERS)],1).sigmoid()
