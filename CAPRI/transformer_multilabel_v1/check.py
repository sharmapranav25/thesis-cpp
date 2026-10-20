import torch
from train import Model
torch.set_num_threads(2);torch.manual_seed(731);m=Model().cuda();x=torch.rand(2,3,96,192,device='cuda');y=torch.zeros(2,15,device='cuda');y[:,[0,1,2]]=1
z=m(x);assert z.shape==(2,15);torch.nn.functional.binary_cross_entropy_with_logits(z,y).backward()
assert all(h.weight.grad.abs().sum()>0 for h in m.heads)
assert m.patch.weight.grad.abs().sum()>0 and m.position.grad.abs().sum()>0
assert not torch.equal(m.encoder.layers[0].self_attn.in_proj_weight,m.encoder.layers[1].self_attn.in_proj_weight)
m.eval()
with torch.no_grad():assert torch.allclose(m(x),m(x))
print('PASS15head multihot gradients, patch/position gradients, independent layer initialization, deterministic evaluation; parameters',sum(p.numel() for p in m.parameters()))
