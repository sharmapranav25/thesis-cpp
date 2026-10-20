"""Separate synthetic curve-threshold diagnostic, not a deployment change."""
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score
import run_v2_tier1_04 as v4
HERE=Path(__file__).parent;OUT=HERE/'v2_tier1_05';torch.set_num_threads(4)
assert (OUT/'status.json').exists(),'Training must finish first'
fonts=json.loads((OUT/'fonts.json').read_text());data={}
for part,seed,bank,recipe in [('calibration',60000000,fonts[:6],0),('fresh_same',61000000,fonts[:6],0),('fresh_font',62000000,fonts[7:],0),('fresh_recipe',63000000,fonts[:6],1)]:
    items=[v4.base.generate(bank,seed+i,recipe) for i in range(420)]
    data[part]=(torch.tensor(np.stack([i[0] for i in items]).transpose(0,3,1,2)/255,dtype=torch.float32),np.stack([i[2] for i in items])[:,1].astype(bool))
def scores(p,y,t):
    pred=p>=t;tp=np.sum(pred&y);fp=np.sum(pred&~y);fn=np.sum(~pred&y)
    return dict(f1=float(2*tp/max(1,2*tp+fp+fn)),fpr=float(fp/max(1,(~y).sum())),recall=float(tp/max(1,y.sum())),precision=float(tp/max(1,pred.sum())))
report={'protocol':'Curves only; maximize calibration F1 on420new synthetic training-font images60m; tie largerthreshold; no real fitting, no heldout font calibration. Fixed threshold and calibration reported separately. Reused procedural generator, exploratory diagnostic; not a deployment promotion.'}
for name in ('original03','original04','no_local_aux','coarse_curve_aux'):
    model=(v4.base.Multi(True) if name=='original03' else v4.Specialists(True)).cuda().eval()
    path=HERE/'v2_tier1_03/multiview.pt' if name=='original03' else HERE/'v2_tier1_04/grid_specialists.pt' if name=='original04' else OUT/(name+'.pt')
    model.load_state_dict(torch.load(path,weights_only=True)['state_dict']);p={}
    with torch.no_grad():
        for part,(x,y) in data.items():p[part]=np.concatenate([model(x[i:i+12].cuda())[0].sigmoid()[:,1].cpu().numpy() for i in range(0,len(x),12)])
    threshold=max((scores(p['calibration'],data['calibration'][1],t)['f1'],float(t)) for t in np.arange(.05,1,.01))[1]
    r={'threshold':threshold}
    for part in ('fresh_same','fresh_font','fresh_recipe'):r[part]={'fixed':scores(p[part],data[part][1],.5),'calibrated':scores(p[part],data[part][1],threshold),'auc':roc_auc_score(data[part][1],p[part])}
    report[name]=r;v4.base.write(OUT/'curve_calibration.json',report);print(name,threshold,r['fresh_same'],flush=True);del model
