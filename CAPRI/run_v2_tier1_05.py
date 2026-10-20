"""Matched shared versus grid-specialist heads, with local mask supervision.

Run beside run_v2_tier1_03.py. No real samples enter training or selection.
"""
import json,time,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import run_v2_tier1_03 as base

HERE=Path(__file__).parent;OUT=HERE/'v2_tier1_05'
GRIDS=base.GRIDS;FEATURES=base.FEATURES

class Specialists(base.Multi):
    def __init__(self,separate=True):
        super().__init__(True)
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(48,32),nn.ReLU(),nn.Linear(32,5)) for _ in range(8 if separate else 1)])
        self.separate=separate
        self.evidence=nn.Sequential(nn.Linear(57,48),nn.ReLU())
        self.local_score=nn.Linear(48,1)
        self.grid_score=nn.Sequential(nn.Linear(50,24),nn.Tanh(),nn.Linear(24,1))
        self.register_buffer('scales',torch.tensor(GRIDS,dtype=torch.float32)/10)
        self.last_local=None

    def forward(self,x):
        b=len(x);v=len(self.grid)
        views=F.grid_sample(x[:,None].expand(-1,v,-1,-1,-1).reshape(b*v,3,64,128),self.grid[None].expand(b,-1,-1,-1,-1).reshape(b*v,24,48,2),align_corners=True)
        tokens=self.encoder(views).mean((2,3)).reshape(b,v,48)
        local=[];summaries=[];offset=0
        for g,(cols,rows) in enumerate(GRIDS):
            n=cols*rows;t=tokens[:,offset:offset+n]
            logits=self.heads[g if self.separate else 0](t);local.append(logits)
            evidence=self.evidence(torch.cat([t,self.boxes[None,offset:offset+n].expand(b,-1,-1),logits.sigmoid()],2))
            weights=self.local_score(evidence).softmax(1)
            summaries.append((weights*evidence).sum(1));offset+=n
        self.last_local=torch.cat(local,1)
        summaries=torch.stack(summaries,1)
        weights=self.grid_score(torch.cat([summaries,self.scales[None].expand(b,-1,-1)],2)).softmax(1)
        self.last_grid_weights=weights
        context=(summaries*weights).sum(1)
        spatial=self.encoder(x)+self.context(context)[:,:,None,None]
        detail=self.detail(x);mid=F.interpolate(self.mid(detail),size=x.shape[-2:],mode='bilinear',align_corners=False)
        coarse=F.interpolate(spatial,size=x.shape[-2:],mode='bilinear',align_corners=False)
        z=self.decode(torch.cat([detail,mid,coarse],1));prob=z.sigmoid().flatten(2)
        summary=torch.cat([prob.mean(2),prob.topk(32,dim=2).values.mean(2)],1)
        return self.presence(torch.cat([context,summary],1)),z

def local_targets(m):
    targets=[]
    for cols,rows in GRIDS:
        for j in range(rows):
            for i in range(cols):
                # Native pixel centers inside each grid_sample crop extent.
                x0=int(np.ceil(i/cols*127));x1=int(np.floor((i+1)/cols*127))+1
                y0=int(np.ceil(j/rows*63));y1=int(np.floor((j+1)/rows*63))+1
                targets.append((m[:,:,y0:y1,x0:x1].sum((2,3))>=3).float())
    return torch.stack(targets,1)

def metrics(p,t):
    p=p>=.5;t=t.bool();tp=(p&t).sum(0);fp=(p&~t).sum(0);fn=(~p&t).sum(0)
    return dict(f1=(2*tp/(2*tp+fp+fn).clamp_min(1)).tolist(),precision=(tp/(tp+fp).clamp_min(1)).tolist(),recall=(tp/(tp+fn).clamp_min(1)).tolist(),fpr=(fp/(~t).sum(0).clamp_min(1)).tolist())

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);start=time.time()
    for f in (__file__,base.__file__):shutil.copy2(f,OUT/Path(f).name)
    fonts=json.loads((HERE/'v2_tier1_03/fonts.json').read_text());base.write(OUT/'fonts.json',fonts)
    protocol=dict(training='synthetic only',grids=GRIDS,views=63,heads=8,shared_backbone=True,auxiliary='crop presence from visible native-mask mass>=3; average loss separately per grid',fusion='within-grid coordinate-aware evidence pooling, then learned grid pooling; local feature probabilities included',comparators=['03 original shared aggregation frozen','04 shared local head, same fusion/loss/steps','04 eight grid heads'],steps=1800,batch=12,selection='same global BCE+dice criterion as03; local auxiliary excluded from checkpoint ranking',threshold=.5,training_seed=431,fresh_seeds=[61000000,62000000,63000000],interventions=['all auxiliary weight0','only 5x3/10x3 curve auxiliary terms zero; no reweighting of remaining terms'],scope='5 real cohorts;9 fonts;no rendering promotion',limitations='one training seed, different head parameter counts, same procedural generator; raw probabilities uncalibrated')
    base.write(OUT/'protocol.json',protocol);data={}
    for part in ('train','selection'):
        d=np.load(HERE/'v2_tier1_03'/(part+'_supervision.npz'))
        x=torch.tensor(d['images'].transpose(0,3,1,2)/255,dtype=torch.float32,device='cuda')
        m=torch.tensor(d['visible_masks']/255,dtype=torch.float32,device='cuda');y=torch.tensor(d['labels'],device='cuda')
        data[part]=(x,m,y,local_targets(m));print('LOADED',part,len(x),flush=True)
    pw=((1-data['train'][2]).sum(0)/data['train'][2].sum(0).clamp_min(1))
    mw=((1-data['train'][1]).sum((0,2,3))/data['train'][1].sum((0,2,3)).clamp_min(1)).clamp(1,20)[None,:,None,None]
    local_weights=[];offset=0
    for cols,rows in GRIDS:
        n=cols*rows;t=data['train'][3][:,offset:offset+n];local_weights.append(((1-t).sum((0,1))/t.sum((0,1)).clamp_min(1)).clamp(1,20));offset+=n
    def global_loss(a,z,y,m):
        prob=z.sigmoid();dice=1-(2*(prob*m).sum((2,3))+1)/((prob+m).sum((2,3))+1)
        return 3*F.binary_cross_entropy_with_logits(a,y,pos_weight=pw)+F.binary_cross_entropy_with_logits(z,m,pos_weight=mw)+dice.mean()
    audit={'all_clean_targets_zero':bool((data['train'][3][data['train'][2].sum(1)==0]==0).all()),'grid1_matches_global':bool(torch.equal(data['train'][3][:,0],data['train'][2]))}
    assert all(audit.values());base.write(OUT/'target_audit.json',audit)
    for name,separate in [('no_local_aux',True),('coarse_curve_aux',True)]:
        torch.manual_seed(431);model=Specialists(separate).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);rng=np.random.default_rng(431);best=1e9;history=[]
        for step in range(1,1801):
            model.train();x,m,y,t=data['train'];ids=rng.integers(len(x),size=12);a,z=model(x[ids]);loss=global_loss(a,z,y[ids],m[ids]);aux=[];offset=0
            for g,(cols,rows) in enumerate(GRIDS):
                n=cols*rows;terms=F.binary_cross_entropy_with_logits(model.last_local[:,offset:offset+n],t[ids,offset:offset+n],pos_weight=local_weights[g],reduction='none')
                if name=='coarse_curve_aux' and g>=6:
                    keep=torch.ones(5,device='cuda');keep[1]=0;terms=terms*keep
                aux.append(terms.mean());offset+=n
            auxiliary=torch.stack(aux).mean();loss=loss+(0 if name=='no_local_aux' else 1)*auxiliary;opt.zero_grad();loss.backward()
            if step==1:
                assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in h.parameters()) for h in model.heads)
            opt.step()
            if step%100==0:
                model.eval();x,m,y,t=data['selection'];total=0.
                with torch.no_grad():
                    for i in range(0,len(x),12):a,z=model(x[i:i+12]);total+=float(global_loss(a,z,y[i:i+12],m[i:i+12]))*len(a)
                score=total/len(x);history.append(dict(step=step,selection=score,train=float(loss.detach()),local_aux=float(auxiliary.detach())))
                base.write(OUT/(name+'_training.json'),history)
                if score<best:best=score;torch.save(dict(state_dict=model.state_dict(),step=step),OUT/(name+'.pt'))
                print(name,step,score,flush=True)
        del model,opt
    del data
    evaluate(fonts)
    base.write(OUT/'status.json',dict(state='complete',seconds=time.time()-start,gate_pass=False,renderer_integrated=False));print('COMPLETE',flush=True)

def evaluate(fonts):
    data={};seedsets=set()
    for part,seed,bank,recipe in [('fresh_same',61000000,fonts[:6],0),('fresh_font',62000000,fonts[7:],0),('fresh_recipe',63000000,fonts[:6],1)]:
        items=[base.generate(bank,seed+i,recipe) for i in range(420)]
        x=torch.tensor(np.stack([a[0] for a in items]).transpose(0,3,1,2)/255,dtype=torch.float32)
        m=torch.tensor(np.stack([a[1] for a in items]));y=torch.tensor(np.stack([a[2] for a in items]));data[part]=(x,m,y,local_targets(m))
        for item in items:assert item[4]['seed'] not in seedsets;seedsets.add(item[4]['seed'])
        base.write(OUT/(part+'_records.json'),[a[4] for a in items])
    results={}
    for name,separate in [('original03',None),('original04',True),('no_local_aux',True),('coarse_curve_aux',True)]:
        model=(base.Multi(True) if separate is None else Specialists(separate)).cuda().eval()
        path=HERE/'v2_tier1_03/multiview.pt' if separate is None else HERE/'v2_tier1_04/grid_specialists.pt' if name=='original04' else OUT/(name+'.pt')
        saved=torch.load(path,weights_only=True);model.load_state_dict(saved['state_dict']);result={'selected_step':saved['step'],'parameters':sum(p.numel() for p in model.parameters())}
        for part,(x,m,y,t) in data.items():
            aa=[];ll=[];inter=torch.zeros(5);union=torch.zeros(5)
            with torch.no_grad():
                for i in range(0,len(x),12):
                    a,z=model(x[i:i+12].cuda());aa.append(a.sigmoid().cpu());p=z.sigmoid().cpu()>=.5;truth=m[i:i+12]>=.25;inter+=(p&truth).sum((0,2,3));union+=(p|truth).sum((0,2,3))
                    if separate is not None:ll.append(model.last_local.sigmoid().cpu())
            result[part]=metrics(torch.cat(aa),y);result[part]['native_iou']=(inter/union.clamp_min(1)).tolist()
            if ll:
                local=torch.cat(ll);offset=0;detail={}
                for cols,rows in GRIDS:
                    n=cols*rows;detail[f'{cols}x{rows}']=metrics(local[:,offset:offset+n].flatten(0,1),t[:,offset:offset+n].flatten(0,1));offset+=n
                result[part]['local_heads']=detail
        results[name]=result;base.write(OUT/'results.json',results);print('EVAL',name,result['fresh_same']['f1'],flush=True)
        if name=='coarse_curve_aux':
            real={}
            with torch.no_grad():
                for family in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
                    profile=json.loads((base.RELEASE/'profiles'/f'{family}.json').read_text());rows=[]
                    for entry in profile['entries'][:4]:
                        im=Image.open(entry['reference_path_provenance']).convert('RGB').resize((128,64));x=torch.tensor(np.asarray(im).transpose(2,0,1)[None]/255,dtype=torch.float32,device='cuda');a,z=model(x)
                        rows.append(dict(reference=entry['reference_path_provenance'],probabilities=a.sigmoid()[0].cpu().tolist(),local_probabilities=model.last_local.sigmoid()[0].cpu().tolist(),grid_weights=model.last_grid_weights[0,:,0].cpu().tolist(),existing_cv_profile=entry))
                    real[family]=rows
            base.write(OUT/'real_inference.json',real)
        del model

if __name__=='__main__':main()
