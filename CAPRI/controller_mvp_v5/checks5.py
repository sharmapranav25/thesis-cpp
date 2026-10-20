from setup5 import ROOT,U
from dino5 import DinoComparator,Objective,WEIGHTS
import numpy as np
import torch

def main():
    d=DinoComparator();rng=np.random.default_rng(51);images=[rng.integers(0,256,(64,256,3),dtype=np.uint8) for _ in range(2)]
    features=d.extract(images);_,x=d.tensor(images)
    with torch.inference_mode():
        official=d.model.get_last_selfattention(x)[:,:,0,1:];official=official/official.sum(-1,keepdim=True)
        assert torch.allclose(official,d.last_attention/d.last_attention.sum(-1,keepdim=True),atol=1e-6)
        lo=torch.quantile(official,.1,dim=-1,keepdim=True);hi=torch.quantile(official,.9,dim=-1,keepdim=True)
        expected=((official-lo)/(hi-lo).clamp_min(1e-8)).clamp(0,1).reshape(-1,6,8,32).cpu().numpy()
        assert np.allclose(expected,features['heads'],atol=2e-5)
    assert all(np.isfinite(v).all() for v in features.values())
    rows=[dict(complete=True,usable=True)]*2;o=Objective(features,features);assert o.score(features,features,rows)['objective']<1e-6
    perm=d.extract(images[::-1]);assert all(np.allclose(features[k][::-1],perm[k],atol=2e-5) for k in features)
    assert not any(p.requires_grad for p in d.model.parameters())
    U.dump(ROOT/'checks.json',dict(passed=True,official_attention_equivalence=True,identical_distribution_zero=True,order_invariance=True,frozen=True,parameters=sum(p.numel() for p in d.model.parameters())))
    print('DINO attention and objective checks passed')

if __name__=='__main__':main()
