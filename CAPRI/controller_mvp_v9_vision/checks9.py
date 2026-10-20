"""Compatibility checks only. Random DINOv3 weights are never evaluated as a model."""
from features9 import *

def main():
    missing_guard=False
    try:DINOv3(ROOT/'weights/does_not_exist.pth')
    except FileNotFoundError:missing_guard=True
    assert missing_guard
    torch.manual_seed(99001);model=DINOv3(architecture_only=True)
    image=np.random.default_rng(99001).integers(0,256,(64,256,3),dtype=np.uint8)
    result=model.extract([image]);again=model.extract([image])
    assert result['embedding'].shape==(1,6144)
    assert all(np.isfinite(v).all() for v in result.values())
    assert all(np.array_equal(v,again[k]) for k,v in result.items())
    assert all(not p.requires_grad for p in model.model.parameters())
    x=np.random.default_rng(3).normal(size=(10,8));assert energy(x,x)<1e-8 and energy(x,x+3)>0
    U.dump(ROOT/'checks.json',dict(passed=True,missing_checkpoint_guard=True,architecture_forward_shape=[1,6144],finite_features=True,
        deterministic_cached_features=True,backbone_frozen=True,energy_identity_and_shift=True,
        limitation='DINOv3 architecture compatibility only. Untrained weights; no accuracy, localization or controller quality evaluation.'))
    print('DINOv3 API/shape/finiteness, frozen backbone, missing-weight guard and distance checks passed.')

if __name__=='__main__':main()
