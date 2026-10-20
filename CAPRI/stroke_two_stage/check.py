import numpy as np,torch
from train import sample,fonts,Detector,Shape
bank=fonts()[:6]
for seed in range(101000000,101000012):
 a,m,l=sample(seed,bank);b,n,k=sample(seed,bank)
 assert np.array_equal(a,b) and np.array_equal(m,n) and l==k
 assert bool(m.any())==bool(l)
x=torch.rand(3,3,64,128);net=Detector();p=net(x);assert p.shape==(3,1,64,128)
p.square().mean().backward();assert any(v.grad is not None and v.grad.abs().sum()>0 for v in net.parameters())
s=Shape();q=s(torch.rand(3,1,64,128));assert q.shape==(3,2)
print('PASS deterministic samples, target presence, detector dimensions/gradient, shape output dimensions')
