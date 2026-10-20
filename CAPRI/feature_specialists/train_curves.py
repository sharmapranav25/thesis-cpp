"""Finite independent curve-specialist search; resumable completed candidates."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,hashlib,json,time,sys,shutil
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from corpus import prepare,ROOT

HERE=Path(__file__).parent
GRIDS=[(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(5,3),(10,3)]

class Classifier(nn.Module):
    def __init__(self,grids,use_pop):
        super().__init__();self.grids=grids;self.use_pop=use_pop
        self.encoder=nn.Sequential(nn.Conv2d(4 if use_pop else 3,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv2d(16,24,3,2,1),nn.GroupNorm(4,24),nn.SiLU(),nn.Conv2d(24,32,3,2,1),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,48,3,2,1),nn.GroupNorm(4,48),nn.SiLU(),nn.AdaptiveAvgPool2d((4,8)),nn.Flatten(),nn.Linear(48*4*8,96),nn.SiLU())
        self.position=nn.Sequential(nn.Linear(100,96),nn.SiLU());self.score=nn.Linear(96,1);self.output=nn.Linear(96,1)
        sample=[];boxes=[]
        for cols,rows in grids:
            for j in range(rows):
                for i in range(cols):
                    y,x=torch.meshgrid(torch.linspace(2*j/rows-1,2*(j+1)/rows-1,48),torch.linspace(2*i/cols-1,2*(i+1)/cols-1,96),indexing='ij');sample.append(torch.stack([x,y],-1));boxes.append([i/cols,j/rows,1/cols,1/rows])
        self.register_buffer('grid',torch.stack(sample));self.register_buffer('boxes',torch.tensor(boxes))
    def forward(self,x):
        x=x if self.use_pop else x[:,:3];b,c,h,w=x.shape;v=len(self.grid)
        crops=F.grid_sample(x[:,None].expand(-1,v,-1,-1,-1).reshape(b*v,c,h,w),self.grid[None].expand(b,-1,-1,-1,-1).reshape(b*v,48,96,2),align_corners=True)
        tokens=self.encoder(crops).reshape(b,v,96);tokens=self.position(torch.cat([tokens,self.boxes[None].expand(b,-1,-1)],2));offset=0;pools=[]
        for cols,rows in self.grids:
            n=cols*rows;t=tokens[:,offset:offset+n];pools.append((self.score(t).softmax(1)*t).sum(1));offset+=n
        return self.output(torch.stack(pools,1).mean(1)).squeeze(1)

def stats(p,y,threshold):
    a=p>=threshold;y=y.astype(bool);tp=int((a&y).sum());fp=int((a&~y).sum());fn=int((~a&y).sum());tn=int((~a&~y).sum());n=fp+tn;q=fp/max(n,1);z=1.96
    upper=(q+z*z/(2*n)+z*np.sqrt(q*(1-q)/n+z*z/(4*n*n)))/(1+z*z/n) if n else 1.
    return dict(f1=2*tp/max(1,2*tp+fp+fn),recall=tp/max(1,tp+fn),precision=tp/max(1,tp+fp),fpr=q,fpr_wilson95_upper=upper,tp=tp,fp=fp,tn=tn,fn=fn,ap=float(average_precision_score(y,p)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--monitor',action='store_true');args=parser.parse_args()
    out=HERE/'runs/curves_v1';out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    import cv2;cv2.setNumThreads(1)
    # OS-managed exclusive byte lock releases on crash; no stale PID auto-deletion.
    import msvcrt
    lock=(out/'job.lock').open('a+b');lock.seek(0)
    if lock.read(1)==b'':lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Another curve-specialist job owns this run')
    def write(name,value):
        target=out/name;temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf8');temporary.replace(target)
    def event(stage,**values):
        row=dict(time=time.time(),pid=os.getpid(),stage=stage,**values);write('status.json',row)
        with (out/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if args.monitor:
            sys.path.insert(0,str(ROOT));from style_pipeline.monitor import update
            update('curve_specialist_'+stage,json.dumps(values),running=stage not in ('complete','failed','prepared'),run=str(out))
    try:
        # Hash all execution sources; refuse a silent change underneath an existing run.
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'PLAN.md',HERE/'corpus.py',Path(__file__)]}
        protocol=dict(source_hashes=hashes,feature='curved_interference',grids=GRIDS,screen_epochs=12,confirm_epochs=24,batch=16,seeds=[431,432,433],selection='AP on synthetic selection; patience4/min6epochs',runtime=dict(torch=torch.__version__,cuda=torch.cuda.is_available()),gate=dict(f1=.9,recall=.9,fpr=.05))
        if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())['source_hashes']==hashes,'Source changed: create a new run version'
        else:write('protocol.json',protocol)
        event('preparing');prepare(out,event)
        if args.prepare_only:event('prepared',training_started=False);return
        assert torch.cuda.is_available(),'CUDA required for scheduled search'
        cache={}
        def data(part):
            if part not in cache:
                d=np.load(out/(part+'.npz'));x=np.concatenate([d['images'],d['pop'][:,:,:,None]],3).transpose(0,3,1,2).copy();cache[part]=(torch.tensor(x,device='cuda',dtype=torch.uint8),d['labels'])
            return cache[part]
        def predict(model,part):
            x,y=data(part);p=[];model.eval()
            with torch.no_grad():
                for i in range(0,len(x),16):p.extend(model(x[i:i+16].float()/255).sigmoid().cpu().numpy())
            return np.asarray(p),y
        def train(name,grids,use_pop,seed,epochs):
            grids=[list(g) for g in grids]
            metrics=out/(name+'.json');checkpoint=out/(name+'.pt')
            if metrics.exists():return json.loads(metrics.read_text())
            if checkpoint.exists():event('restart_incomplete_candidate',candidate=name,reason='no completed candidate record; deterministic retrain')
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Classifier(grids,use_pop).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01);best=-1;stale=0;history=[];start=time.time();x,y=data('train');labels=torch.tensor(y,device='cuda');y_counts=np.bincount(y.astype(int));assert y_counts[0]==y_counts[1]
            for epoch in range(1,epochs+1):
                model.train();order=rng.permutation(len(x));total=0.
                for i in range(0,len(x),16):
                    ids=order[i:i+16];a=model(x[ids].float()/255);loss=F.binary_cross_entropy_with_logits(a,labels[ids]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ids)
                    if i==0 or (i//16+1)%250==0:event('batch_progress',candidate=name,epoch=epoch,batch=i//16+1,total_batches=(len(x)+15)//16,loss=float(loss.detach()))
                p,t=predict(model,'selection');ap=float(average_precision_score(t,p));history.append(dict(epoch=epoch,loss=total/len(x),selection_ap=ap));write(name+'_history.json',history)
                if ap>best+1e-4:
                    best=ap;stale=0;torch.save(dict(state_dict=model.state_dict(),grids=grids,use_pop=use_pop,epoch=epoch,seed=seed),checkpoint)
                else:stale+=1
                event('training',candidate=name,epoch=epoch,max_epochs=epochs,selection_ap=ap,best_ap=best,elapsed_seconds=time.time()-start)
                if epoch>=6 and stale>=4:break
            result=dict(name=name,grids=grids,use_pop=use_pop,seed=seed,selection_ap=best,seconds=time.time()-start,epochs_run=len(history),parameters=sum(p.numel() for p in model.parameters()));write(name+'.json',result);del model,opt;return result
        screen=[]
        for grid in GRIDS:
            for use_pop in (False,True):screen.append(train(f'screen_{grid[0]}x{grid[1]}_'+('pop' if use_pop else 'raw'),[grid],use_pop,431,12))
        ranked=sorted(screen,key=lambda r:r['selection_ap'],reverse=True);first=ranked[0];second=next(r for r in ranked if r['grids']!=first['grids']);combination=first['grids']+second['grids']
        combo=train('screen_combination',combination,first['use_pop'],431,12);winner=max(screen+[combo],key=lambda r:r['selection_ap']);write('view_selection.json',dict(candidates=screen+[combo],winner=winner,locked_tests_used=False));event('configuration_frozen',winner=winner)
        confirmed=[]
        for seed in (431,432,433):confirmed.append(train(f'confirm_{seed}',winner['grids'],winner['use_pop'],seed,24))
        evaluations={};passed=True
        for candidate in confirmed:
            name=candidate['name'];saved=torch.load(out/(name+'.pt'),weights_only=True);model=Classifier(saved['grids'],saved['use_pop']).cuda();model.load_state_dict(saved['state_dict']);p,y=predict(model,'calibration');choices=[]
            for threshold in np.arange(.01,1,.01):
                s=stats(p,y,threshold)
                if s['fpr']<=.05:choices.append((s['f1'],float(threshold)))
            if not choices:evaluations[name]={'calibration_failed':True};passed=False;continue
            threshold=max(choices)[1];r={'threshold':threshold,'calibration':stats(p,y,threshold)}
            for part in ('locked_same','locked_font','locked_shape','locked_raster'):
                event('locked_test_consumed',candidate=name,partition=part,warning='now development evidence for any subsequent revision')
                p,y=predict(model,part);s=stats(p,y,threshold);r[part]=s;passed=passed and s['f1']>=.9 and s['recall']>=.9 and s['fpr']<=.05
                records=json.loads((out/(part+'_records.json')).read_text());r[part]['angle_bins']={}
                for angle in range(12):
                    ids=np.repeat([a['angle_bin']==angle for a in records],2);r[part]['angle_bins'][str(angle)]=stats(p[ids],y[ids],threshold)
                r[part]['contrast_bins']={}
                for label,lo,hi in [('low',0,40),('medium',40,100),('high',100,1000)]:
                    ids=np.repeat([lo<=a['contrast']<hi for a in records],2)
                    if ids.any():r[part]['contrast_bins'][label]=stats(p[ids],y[ids],threshold)
                del cache[part]
            evaluations[name]=r;write('locked_results.json',evaluations);del model
        write('gate.json',dict(synthetic_gate_pass=bool(passed),production_promoted=False,real_accuracy_validated=False))
        report='# Independent curve specialist: completed finite search\n\nWinner (selection only): '+json.dumps(winner)+'\n\nSynthetic gate: '+str(bool(passed))+'. Production unchanged. No real accuracy or renderer-parameter claim.\n\n| Seed | Partition | F1 | Recall | FPR |\n|---|---|---:|---:|---:|\n'
        for name,r in evaluations.items():
            for part,s in r.items():
                if part.startswith('locked_'):report+=f"| {name} | {part} | {s['f1']:.3f} | {s['recall']:.3f} | {s['fpr']:.3f} |\n"
        (out/'RESULTS.md').write_text(report,encoding='utf8')
        if args.monitor:
            (ROOT/'docs/V2_CURVE_SPECIALIST_RESULTS.md').write_text(report,encoding='utf8')
            with (ROOT/'docs/STYLE_DECISIONS.md').open('a',encoding='utf8') as f:f.write('\n\nD084 outcome: independent curve search completed; synthetic gate='+str(bool(passed))+'. See V2_CURVE_SPECIALIST_RESULTS.md. No production promotion.\n')
            with (ROOT/'docs/STYLE_HANDOFF.md').open('a',encoding='utf8') as f:f.write('\n\nLATEST: independent curve-specialist finite search complete. Read V2_CURVE_SPECIALIST_RESULTS.md and feature_specialists_v2/runs/curves_v1/gate.json. No worker remaining; other feature specialists queued. Automation paused.\n')
        event('complete',synthetic_gate_pass=bool(passed),production_promoted=False)
    except BaseException as exc:
        event('failed',error=repr(exc));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':main()
