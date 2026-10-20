from pathlib import Path
import json
import torch
from run_v2_tier1_04 import Specialists,local_targets,GRIDS
torch.set_num_threads(4)
m=torch.zeros(2,5,64,128);m[1]=1
t=local_targets(m)
assert t.shape==(2,63,5) and t[0].sum()==0 and t[1].sum()==315
# Only a dot in bottom-right: no top-left cell of the10x3 grid may claim it.
m.zero_();m[0,2,60:63,123:127]=1;t=local_targets(m)
assert t[0,0,2]==1 and t[0,-1,2]==1 and t[0,-30,2]==0
model=Specialists(True)
assert len(model.heads)==len(GRIDS)==8
assert len({p.data_ptr() for h in model.heads for p in h.parameters()})==32
# Direct isolated supervision updates only the chosen head, not other head parameters.
model.heads[3](torch.randn(2,48)).sum().backward()
assert all(p.grad is not None for p in model.heads[3].parameters())
assert all(p.grad is None for i,h in enumerate(model.heads) if i!=3 for p in h.parameters())
out=Path(__file__).parent/'v2_tier1_04'
report=dict(native_crop_labels_checked=True,grids=GRIDS,crops=63,independent_parameter_heads=8,isolated_head_gradient_check=True)
(out/'architecture_audit.json').write_text(json.dumps(report,indent=2));print(report)
