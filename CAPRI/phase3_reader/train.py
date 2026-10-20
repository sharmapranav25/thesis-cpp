"""Finite matched training; synthetic selection/calibration only; freeze before inference."""
import argparse, math, random, sys, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from models import Reader

def batch(cache,indices,device,layout=False):
    x=torch.from_numpy(np.asarray(cache[indices]).copy()).permute(0,3,1,2).to(device).float()/127.5-1
    if layout:
        # Fixed independently specified layout stress; never used for selection.
        small=F.interpolate(x,size=(32,116),mode='bilinear',align_corners=False)
        x=F.pad(small,(38,38),value=1.)
    return x

@torch.no_grad()
def infer(model,cache,indices,device,layout=False):
    model.eval();pred=[];confidence=[];cs=[];gs=[]
    for start in range(0,len(indices),128):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
            p,c,a=model.predict(batch(cache,indices[start:start+128],device,layout))
        pred.extend(p);confidence.extend(c)
        if a:cs.extend(a['count_logits']);gs.extend(a['group_logits'])
    return dict(predictions=pred,confidence=confidence,count_logits=np.asarray(cs),group_logits=np.asarray(gs))

def calibrated(result,calibration):
    if not len(result['count_logits']):return None,None
    cp=torch.tensor(result['count_logits']/calibration['count_temperature']).softmax(-1).numpy()
    gp=torch.tensor(result['group_logits']/np.asarray(calibration['group_temperatures'])).sigmoid().numpy()
    return cp,gp

def score(result,labels,cal):
    cp,gp=calibrated(result,cal)
    return metrics(labels,result['predictions'],None if cp is None else cp.argmax(-1)+2,None if gp is None else gp>=.5)

def calibrate(result,labels):
    out=dict(method='Synthetic heldout calibration only; temperature grid 0.5..5; no transfer guarantee',count_temperature=1.,group_temperatures=[1.]*3,confidence_bins=[])
    if len(result['count_logits']):
        counts=torch.tensor([len(t)-2 for t in labels]);gs=torch.tensor([groups(t) for t in labels],dtype=torch.float32);grid=np.geomspace(.5,5,41)
        cl=torch.tensor(result['count_logits']);gl=torch.tensor(result['group_logits'])
        out['count_temperature']=float(min(grid,key=lambda t:float(F.cross_entropy(cl/t,counts))))
        out['group_temperatures']=[float(min(grid,key=lambda t:float(F.binary_cross_entropy_with_logits(gl[:,j]/t,gs[:,j])))) for j in range(3)]
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        ix=[i for i,v in enumerate(result['confidence']) if lo<=v<=hi and (lo==0 or v>lo)]
        out['confidence_bins'].append(dict(lower=float(lo),upper=float(hi),samples=len(ix),synthetic_exact_rate=float(np.mean([labels[i]==result['predictions'][i] for i in ix])) if ix else None))
    return out

def run(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    torch.backends.cudnn.benchmark=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    a.out.mkdir(parents=True,exist_ok=False)
    audit=read(a.data/'audit.json');assert audit['passed']
    rows=[json.loads(l) for l in (a.data/'partitions.jsonl').open()]
    assert sha(a.data/'partitions.jsonl')==audit['partitions_sha256']
    ix={p:np.array([r['index'] for r in rows if r['partition']==p]) for p in ['train','selection','calibration','test','font_test']}
    cache=np.load(a.data/'synthetic.npy',mmap_mode='r'); model=Reader(a.model,a.aux,a.seed).to(device)
    cfg=dict(model=a.model,auxiliary=a.aux,seed=a.seed,epochs=a.epochs,batch=128,lr=.0003,weight_decay=.01,image_size=[32,192],max_decode_length=12,training_lengths=list(range(2,9)),count_head_support=list(range(2,9)),auxiliary_loss_weights=[.25,.25],optimizer='AdamW',selection='lowest synthetic selection CER, then highest exact accuracy',precision='AMP bfloat16 without gradient scaling; float32 CTC',pretrained=False,parameters=sum(p.numel() for p in model.parameters()),partition_sha256=audit['partitions_sha256'],source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},architecture='CRNN upstream default; SVTR tiny local/global; PARSeq 192d six encoder layers, six permutations; adapted scale, not paper reproduction',real_labels_used=False)
    write(a.out/'protocol.json',cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01); scaler=torch.amp.GradScaler('cuda',enabled=False)
    steps=math.ceil(len(ix['train'])/128);total_steps=steps*a.epochs;step=0;best=(float('inf'),float('inf'));history=[];start_time=time.time()
    for epoch in range(a.epochs):
        order=np.random.default_rng(a.seed+epoch).permutation(ix['train']);model.train();losses=[]
        for start in range(0,len(order),128):
            indices=order[start:start+128];labels=[rows[i]['label'] for i in indices];x=batch(cache,indices,device)
            progress=step/max(1,total_steps-1);factor=(progress/.1 if progress<.1 else .5*(1+math.cos(math.pi*(progress-.1)/.9)))
            for pg in optimizer.param_groups:pg['lr']=.0003*max(.01,factor)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):loss,recognition=model.loss(x,labels)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);scaler.step(optimizer);scaler.update();step+=1;losses.append(float(recognition))
            if step%50==0:
                status=dict(state='training',model=a.model,auxiliary=a.aux,pid=os.getpid(),epoch=epoch+1,epochs=a.epochs,step=step,total_steps=total_steps,recognition_loss=float(np.mean(losses[-50:])),elapsed_seconds=time.time()-start_time)
                write(a.out/'status.json',status);print(json.dumps(status),flush=True)
        result=infer(model,cache,ix['selection'],device);m=score(result,[rows[i]['label'] for i in ix['selection']],dict(count_temperature=1.,group_temperatures=[1.,1.,1.]))
        key=(m['cer'],-m['exact_string_accuracy']);entry=dict(epoch=epoch+1,steps=step,recognition_loss=float(np.mean(losses)),selection=m,elapsed_seconds=time.time()-start_time);history.append(entry);write(a.out/'history.json',history);print(json.dumps(entry),flush=True)
        if key<best:
            best=key;torch.save(dict(state_dict=model.state_dict(),protocol=cfg,epoch=epoch+1,steps=step),a.out/'best.pt')
    model.load_state_dict(torch.load(a.out/'best.pt',map_location=device,weights_only=False)['state_dict']);model.eval();model.requires_grad_(False);model.feature=None
    checkpoint_hash=sha(a.out/'best.pt');cal_result=infer(model,cache,ix['calibration'],device);cal=calibrate(cal_result,[rows[i]['label'] for i in ix['calibration']]);cal['checkpoint_sha256']=checkpoint_hash;write(a.out/'calibration.json',cal)
    write(a.out/'calibration_metrics.json',score(cal_result,[rows[i]['label'] for i in ix['calibration']],cal))
    tests={}
    for role in ['test','font_test','layout_test']:
        ids=ix['test' if role=='layout_test' else role];r=infer(model,cache,ids,device,layout=role=='layout_test');tests[role]=score(r,[rows[i]['label'] for i in ids],cal)
        with (a.out/(role+'_predictions.jsonl')).open('w') as f:
            for j,i in enumerate(ids):f.write(json.dumps(dict(index=int(i),label=rows[i]['label'],prediction=r['predictions'][j],confidence=r['confidence'][j]))+'\n')
    write(a.out/'synthetic_results.json',tests)
    real=read(a.data/'real_locators.json')
    for part in ['fit','validation']:
        rc=np.load(a.data/(part+'.npy'),mmap_mode='r');result=infer(model,rc,np.arange(len(rc)),device);cp,gp=calibrated(result,cal)
        # IDs only in predictions: filename strings never enter reader or aggregation.
        with (a.out/(part+'_predictions.jsonl')).open('w') as f:
            for i,pred in enumerate(result['predictions']):
                row=dict(id=real[part][i]['id'],pixel_sha256=real[part][i]['pixel_sha256'],prediction=pred,decoded_length=len(pred),decoded_groups=groups(pred),confidence=result['confidence'][i],checkpoint_sha256=checkpoint_hash)
                if cp is not None:row.update(count_probabilities=cp[i].tolist(),group_probabilities=gp[i].tolist(),head_count=int(cp[i].argmax()+2))
                f.write(json.dumps(row)+'\n')
        write(a.out/(part+'_summary.json'),dict(samples=len(real[part]),prediction_lengths=dict(__import__('collections').Counter(map(len,result['predictions']))),real_accuracy=None,ground_truth_status='No independently verified transcription annotations available; filenames are not labels',scheme_use=part=='fit'))
    assert sha(a.out/'best.pt')==checkpoint_hash
    write(a.out/'status.json',dict(state='complete',model=a.model,auxiliary=a.aux,pid=os.getpid(),checkpoint_sha256=checkpoint_hash,epochs=a.epochs,steps=step,elapsed_seconds=time.time()-start_time,synthetic=tests,real_accuracy=None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['crnn','svtr','parseq'],required=True);p.add_argument('--aux',action='store_true');p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=10);p.add_argument('--seed',type=int,default=731);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():write(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));write(a.out/'status.json',dict(state='failed',error=repr(e)))
        raise
