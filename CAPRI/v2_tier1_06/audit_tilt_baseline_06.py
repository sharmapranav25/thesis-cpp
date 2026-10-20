from pathlib import Path
import json,numpy as np,torch
import run_v2_tier1_06 as v6
HERE=Path(__file__).parent;OUT=HERE/'v2_tier1_06';torch.set_num_threads(4)
fonts=json.loads((OUT/'fonts.json').read_text());model=v6.GeometryModel(True).cuda().eval();model.load_state_dict(torch.load(OUT/'geometry_with_neighbors.pt',weights_only=True)['state_dict']);results={}
for part,seed,bank,recipe in [('fresh_same',71000000,fonts[:6],0),('fresh_font',72000000,fonts[7:],0),('fresh_recipe',73000000,fonts[:6],1)]:
    items=[v6.base.generate(bank,seed+i,recipe) for i in range(420)];g,_=v6.geometry_targets(np.stack([r[1] for r in items]));g=torch.tensor(g)
    x=torch.tensor(np.stack([r[0] for r in items]).transpose(0,3,1,2)/255,dtype=torch.float32);p=[]
    with torch.no_grad():
        for i in range(0,len(x),12):model(x[i:i+12].cuda());p.append(model.last_geometry.cpu())
    p=torch.cat(p);valid=(g[:,:,0]>0)&(g[:,:,5]>=.6)
    baseline=.5*torch.acos(g[:,:,3].clamp(-1,1))*180/np.pi
    err=.5*torch.acos((p[:,:,3:5]*g[:,:,3:5]).sum(2).clamp(-1,1))*180/np.pi
    tilt=.5*torch.atan2(p[:,:,4],p[:,:,3])*180/np.pi
    results[part]=dict(horizontal_baseline_mae=float(baseline[valid].mean()),model_mae=float(err[valid].mean()),predictions_within10degrees_horizontal=float((tilt[valid].abs()<10).float().mean()),target_within10degrees_horizontal=float((baseline[valid]<10).float().mean()),valid_crops=int(valid.sum()))
v6.base.write(OUT/'tilt_baseline_audit.json',results);print(results)
