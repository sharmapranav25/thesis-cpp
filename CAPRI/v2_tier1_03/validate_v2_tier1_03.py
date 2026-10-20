"""Fresh synthetic calibration/evaluation after architectural development; no real fitting."""
from pathlib import Path
import importlib.util,json
import numpy as np
import torch
from torch.nn import functional as F
HERE=Path(__file__).parent;OUT=HERE/'v2_tier1_03'
def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v2=module('run_v2_tier1_02');v3=module('run_v2_tier1_03');torch.set_num_threads(4)
fonts=json.loads((OUT/'fonts.json').read_text());data={}
for part,seed,bank,recipe in [('calibration',41000000,fonts[6:7],0),('fresh_same',42000000,fonts[:6],0),('fresh_font',43000000,fonts[7:],0),('fresh_recipe',44000000,fonts[:6],1)]:
    items=[v3.generate(bank,seed+i,recipe) for i in range(420)]
    data[part]=(torch.tensor(np.stack([i[0] for i in items]).transpose(0,3,1,2)/255,dtype=torch.float32),np.stack([i[2] for i in items]).astype(bool),torch.tensor(np.stack([i[1] for i in items])))
    v3.write(OUT/(part+'_records.json'),[i[4] for i in items])
report={'protocol':{'samples_per_partition':420,'seeds':'41m calibration;42m fresh same;43m font only;44m Bezier only','threshold_selection':'maximize calibration F1 subject to empirical calibration FPR<=0.10; grid0.05..0.99. This is not a population guarantee.','mask_comparison':'both models downsample probabilities to32x64, full targets area-resized32x64 then>=0.25; same as02 evaluation','limitations':'same procedural generator; fresh seeds do not establish independent generator transfer; no real accuracy'}}
for revision,mod in [('02',v2),('03',v3)]:
    for name,multi in [('whole_image',False),('single_grid','single_grid'),('multiview',True)]:
        model=mod.Multi(multi).cuda().eval();saved=torch.load(HERE/f'v2_tier1_{revision}'/(name+'.pt'),weights_only=True);model.load_state_dict(saved['state_dict']);preds={};ious={}
        for part,(x,y,m) in data.items():
            probs=[];inter=torch.zeros(5);union=torch.zeros(5)
            with torch.no_grad():
                for i in range(0,len(x),12):
                    a,z=model(x[i:i+12].cuda());probs.append(a.sigmoid().cpu().numpy());p=F.interpolate(z.sigmoid(),size=(32,64),mode='area').cpu()>=.5;t=F.interpolate(m[i:i+12],size=(32,64),mode='area')>=.25;inter+=(p&t).sum((0,2,3));union+=(p|t).sum((0,2,3))
            preds[part]=np.concatenate(probs);ious[part]=(inter/union.clamp_min(1)).tolist()
        thresholds=[];yc=data['calibration'][1]
        for k in range(5):
            choices=[]
            for t in np.arange(.05,1,.01):
                p=preds['calibration'][:,k]>=t;y=yc[:,k];tp=(p&y).sum();fp=(p&~y).sum();fn=(~p&y).sum();fpr=fp/max(1,(~y).sum());f1=2*tp/max(1,2*tp+fp+fn)
                if fpr<=.10:choices.append((f1,float(t)))
            thresholds.append(max(choices)[1] if choices else 1.)
        result={'thresholds':thresholds,'selected_step':saved['step']}
        for part in ('fresh_same','fresh_font','fresh_recipe'):
            result[part]={'common32x64_iou':ious[part]}
            for rule,th in [('fixed',.5),('calibrated',np.array(thresholds))]:
                p=preds[part]>=th;y=data[part][1];tp=(p&y).sum(0);fp=(p&~y).sum(0);fn=(~p&y).sum(0)
                result[part][rule]={'f1':(2*tp/np.maximum(1,2*tp+fp+fn)).tolist(),'precision':(tp/np.maximum(1,tp+fp)).tolist(),'recall':(tp/np.maximum(1,tp+fn)).tolist(),'fpr':(fp/np.maximum(1,(~y).sum(0))).tolist()}
        report[revision+'_'+name]=result;v3.write(OUT/'fresh_validation.json',report);print(revision,name,result['fresh_same'],flush=True)
        del model
print('FRESH VALIDATION COMPLETE',flush=True)
