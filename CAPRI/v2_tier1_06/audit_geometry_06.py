from pathlib import Path
import json
import numpy as np,torch,cv2
from run_v2_tier1_06 import geometry_targets,GeometryModel,BOUNDS,EDGES
torch.set_num_threads(4)
m=np.zeros((4,5,64,128),np.float32)
m[0,0,15,2:125]=1;m[1,0,2:62,60]=1
# Two separate parallel segments: direction agrees but no visible connection.
m[2,0,10,1:10]=1;m[2,0,10,15:23]=1
g,l=geometry_targets(m)
assert g.shape==(4,63,7) and l.shape==(4,len(EDGES))
assert np.isfinite(g).all() and np.isfinite(l).all()
assert np.allclose(g[0,0,3:5],[1,0]) and np.allclose(g[1,0,3:5],[-1,0])
assert g[3,:,0].sum()==0 and l[3].sum()==0
# Finest grid begins at33; its first two cells are neighboring.
edge=EDGES.index((33,34));assert l[0,edge]==1 and l[2,edge]==0
torch.manual_seed(431);a=GeometryModel(False).eval();torch.manual_seed(431);b=GeometryModel(True).eval()
assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
out,z=b(torch.rand(2,3,64,128));assert out.shape==(2,5) and z.shape==(2,5,64,128)
out.sum().backward();assert b.message.weight.grad.abs().sum()>0
report=dict(native_horizontal_vertical_angles=True,empty_strokes_handled=True,disconnected_parallel_segments_not_linked=True,adjacent_continuous_line_linked=True,models_identically_initialized=True,message_path_receives_global_gradient=True,edges=len(EDGES))
path=Path(__file__).parent/'v2_tier1_06/geometry_audit.json';path.write_text(json.dumps(report,indent=2));print(report)
