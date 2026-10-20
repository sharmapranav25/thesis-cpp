from setup4 import ROOT,U,ARMS,CHARSET
import json,time,math
import numpy as np
import torch
from torch.nn import functional as F
from solver4 import Solver,decode,metrics

@torch.inference_mode()
def infer(model,array,device):
    model.eval();pred=[]
    for i in range(0,len(array),128):
        x=torch.from_numpy(np.asarray(array[i:i+128]).copy()).permute(0,3,1,2).to(device).float()/255
        pred.extend(decode(model(x)))
    return pred

def main():
    data=ROOT/'corpus';dest=ROOT/'solvers';dest.mkdir(exist_ok=False);assert U.read(data/'audit.json')['passed']
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';torch.backends.cudnn.benchmark=False
    rows=U.read(data/'train_labels.json');labels=[r['label'] for r in rows];val=np.load(data/'validation_common.npy',mmap_mode='r');vlabels=[r['label'] for r in U.read(data/'validation_labels.json')]
    token=[torch.tensor([CHARSET.index(c)+1 for c in t],dtype=torch.long) for t in labels];n=len(labels);epochs=20;batch=128;frozen={};started=time.time()
    U.dump(dest/'protocol.json',dict(arms=ARMS,seeds=[43103,43109],epochs=epochs,batch=batch,optimizer='AdamW lr=.001 cosine decay, weight_decay=.0001',initialization='Identical random starting weights within each seed; no pretraining',
        selection='Lowest common synthetic validation CER, then highest exact accuracy; all models get all 20 epochs.',real_labels_used=False,vlm=False,corpus_audit_sha256=U.sha(data/'audit.json'),source_sha256={p.name:U.sha(p) for p in [ROOT/'train4.py',ROOT/'solver4.py']}))
    for seed in [43103,43109]:
        torch.manual_seed(seed);initial=Solver().state_dict();initial={k:v.clone() for k,v in initial.items()};torch.save(initial,dest/f'initial_{seed}.pt')
        for arm in ARMS:
            out=dest/f'{arm}_{seed}';out.mkdir();torch.manual_seed(seed);model=Solver().to(device);model.load_state_dict(initial)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);array=np.load(data/('train_'+arm+'.npy'),mmap_mode='r');history=[];best=(float('inf'),float('inf'));step=0
            for epoch in range(epochs):
                model.train();losses=[];order=np.random.default_rng(seed+epoch).permutation(n)
                for pos in range(0,n,batch):
                    ids=order[pos:pos+batch];x=torch.from_numpy(np.asarray(array[ids]).copy()).permute(0,3,1,2).to(device).float()/255
                    targets=torch.cat([token[i] for i in ids]).to(device);lengths=torch.tensor([len(labels[i]) for i in ids],dtype=torch.long)
                    logits=model(x);loss=F.ctc_loss(logits.log_softmax(-1).transpose(0,1),targets,torch.full_like(lengths,logits.shape[1]),lengths,zero_infinity=False)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite CTC loss')
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()));step+=1
                for pg in opt.param_groups:pg['lr']=.001*(.05+.95*(1+math.cos(math.pi*(epoch+1)/epochs))/2)
                pred=infer(model,val,device);score=metrics(vlabels,pred);row=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation=score,steps=step,seconds=time.time()-started);history.append(row)
                if (score['cer'],-score['exact'])<best:
                    best=(score['cer'],-score['exact']);torch.save(model.state_dict(),out/'best.pt');U.dump(out/'selected.json',row)
                U.dump(out/'history.json',history);U.dump(ROOT/'training_status.json',dict(arm=arm,seed=seed,**row));print(json.dumps(dict(arm=arm,seed=seed,**row)),flush=True)
            frozen[f'{arm}_{seed}']=dict(checkpoint=str(out/'best.pt'),sha256=U.sha(out/'best.pt'),initial_sha256=U.sha(dest/f'initial_{seed}.pt'),steps=step,selection=U.read(out/'selected.json'))
    U.dump(dest/'frozen_models.json',frozen);U.dump(ROOT/'training_status.json',dict(stage='complete',seconds=time.time()-started,models=len(frozen)))

if __name__=='__main__':main()
