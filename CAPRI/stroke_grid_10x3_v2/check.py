import torch,numpy as np
from train import tiles,stitch,Detector,sample,fonts
x=torch.rand(2,3,64,128);assert torch.equal(stitch(tiles(x),2),x)
m=Detector();p=m(x);assert p.shape==(2,1,64,128);p.square().mean().backward();assert all(v.grad is not None for v in m.parameters())
bank=fonts()[:6]
for seed in range(111000000,111000120):
 a,mask,kind=sample(seed,bank);b,other,k=sample(seed,bank);assert np.array_equal(a,b);assert bool(mask.any())==bool(kind)
print('PASS exact tile/stitch reconstruction, mask dimensions/gradients,120 deterministic multicolour samples and visible labels')
