import torch
from torch import nn
from torch.nn import functional as F

def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())

class MaskNet(nn.Module):
    """Small full-resolution U-Net. No language model or recognition head."""
    def __init__(self):
        super().__init__()
        self.a=block(3,16);self.b=block(16,32);self.c=block(32,64);self.d=block(64,96)
        self.u=block(160,64);self.v=block(96,32);self.w=block(48,16);self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(F.avg_pool2d(c,2))
        def cat(x,y):return torch.cat([F.interpolate(x,size=y.shape[-2:],mode='bilinear',align_corners=False),y],1)
        return self.out(self.w(cat(self.v(cat(self.u(cat(d,c)),b)),a)))

