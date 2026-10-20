import numpy as np,torch
from corpus import sample,fonts,recipe,FEATURES
from train import Model,GRIDS
torch.set_num_threads(2);bank=fonts()[:6]
for seed in range(201000000,201000015):
 a,y,v,r=sample(seed,'train',bank);b,z,q,s=sample(seed,'train',bank)
 assert np.array_equal(a,b) and np.array_equal(y,z) and np.array_equal(v,q)
 assert len(y)==15 and ((v==0)|(v==1)).all()
train={tuple(recipe(201000000+i,'train')) for i in range(500)};held={tuple(recipe(206000000+i,'locked_combinations')) for i in range(500)};assert not train&held
m=Model();x=torch.rand(2,3,96,192);p=m(x);assert p.shape==(2,15);assert len(m.coords)==50
labels=torch.zeros_like(p);labels[:,[0,1,2]]=1;torch.nn.functional.binary_cross_entropy_with_logits(p,labels).backward();assert all(h.weight.grad.abs().sum()>0 for h in m.heads)
print('PASS deterministic images/labels, uncertain flags, combination disjointness, multihot loss and all15 head gradients; parameters',sum(p.numel() for p in m.parameters()))
