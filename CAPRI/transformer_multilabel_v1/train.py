import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import json,time,hashlib
from pathlib import Path
import numpy as np,torch,cv2
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from corpus import sample,fonts,FEATURES,ROOT
HERE=Path(__file__).parent;OUT=HERE/'run';GRIDS=[(24,12)]
SPLITS={'train':(201000000,8000),'selection':(202000000,1200),'calibration':(203000000,1000),'locked_same':(304000000,800),'locked_font':(305000000,800),'locked_combinations':(306000000,800),'locked_recipe':(307000000,800)}
def write(name,value):
 path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def event(stage,**kw):
 row=dict(time=time.time(),pid=os.getpid(),stage=stage,**kw);write('status.json',row)
 monitor=ROOT/'monitor/transformer_multilabel_status.json';tmp=monitor.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(monitor)
 with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
class Model(nn.Module):
 def __init__(self):
  super().__init__();dim=96
  self.patch=nn.Linear(3*8*8,dim)
  self.position=nn.Parameter(torch.zeros(1,12*24,dim))
  self.queries=nn.Parameter(torch.zeros(1,len(FEATURES),dim))
  layer=nn.TransformerEncoderLayer(d_model=dim,nhead=4,dim_feedforward=384,dropout=.2,activation='gelu',batch_first=True,norm_first=True)
  self.encoder=nn.TransformerEncoder(layer,num_layers=3,norm=nn.LayerNorm(dim),enable_nested_tensor=False)
  # PyTorch clones layer values; initialize each layer independently.
  for block in self.encoder.layers:
   for p in block.parameters():
    if p.ndim>1:nn.init.xavier_uniform_(p)
  self.dropout=nn.Dropout(.2);self.heads=nn.ModuleList([nn.Linear(dim,1) for _ in FEATURES])
  nn.init.trunc_normal_(self.position,std=.02);nn.init.trunc_normal_(self.queries,std=.02)
 def forward(self,x):
  b,c,h,w=x.shape
  assert (h,w)==(96,192)
  patches=x.reshape(b,c,12,8,24,8).permute(0,2,4,1,3,5).reshape(b,288,192)
  tokens=self.dropout(self.patch(patches)+self.position)
  encoded=self.encoder(torch.cat([self.queries.expand(b,-1,-1),tokens],1))
  return torch.cat([head(encoded[:,i]) for i,head in enumerate(self.heads)],1)

def ap(y,p,valid):
 result=[]
 for k in range(len(FEATURES)):
  ids=valid[:,k]>0;result.append(float(average_precision_score(y[ids,k],p[ids,k])))
 return result
def metrics(y,p,valid,thresholds):
 result={}
 for k,f in enumerate(FEATURES):
  ids=valid[:,k]>0;t=y[ids,k].astype(bool);a=p[ids,k]>=thresholds[k];tp=int((t&a).sum());fp=int((~t&a).sum());fn=int((t&~a).sum());tn=int((~t&~a).sum())
  result[f]=dict(ap=float(average_precision_score(t,p[ids,k])),f1=2*tp/max(1,2*tp+fp+fn),recall=tp/max(1,tp+fn),fpr=fp/max(1,fp+tn),tp=tp,fp=fp,fn=fn,tn=tn)
 return result
def main():
 import msvcrt
 OUT.mkdir(exist_ok=True);lock=(OUT/'job.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 assert not (OUT/'results.json').exists(),'Completed run preserved'
 torch.set_num_threads(4);cv2.setNumThreads(1);assert torch.cuda.is_available()
 hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['corpus.py','train.py','PLAN.md']}
 protocol=dict(hashes=hashes,features=FEATURES,splits=SPLITS,views=GRIDS,architecture='pure patch transformer:8x8patches,96dim,3layers,4heads,15learned feature tokens',feature_pop=False,seeds=[731,732,733],epochs=40,min_epochs=8,patience=6,criterion='selection macroAP',batch=32,thresholds='synthetic calibration exact score breakpoints maxF1 underFPR.10',gates=dict(f1=.85,recall=.85,fpr=.1),real_fitting=False,production_promoted=False)
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())['hashes']==hashes
 write('protocol.json',protocol);bank=fonts();audit={};seen=set()
 for part,(start,n) in SPLITS.items():
  path=OUT/(part+'.npz')
  if not path.exists():
   chosen=bank[7:] if part=='locked_font' else bank[:6];images=[];ys=[];vs=[];records=[]
   for i in range(n):
    im,y,v,r=sample(start+i,part,chosen);images.append(im);ys.append(y);vs.append(v);records.append(r)
    if (i+1)%200==0:event('corpus',partition=part,done=i+1,total=n)
   np.savez_compressed(path,images=np.stack(images),labels=np.stack(ys),valid=np.stack(vs));write(part+'_records.json',records)
  with np.load(path) as d:
   counts={f:dict(positive=int(((d['labels'][:,k]>0)&(d['valid'][:,k]>0)).sum()),negative=int(((d['labels'][:,k]==0)&(d['valid'][:,k]>0)).sum()),ignored=int((d['valid'][:,k]==0).sum())) for k,f in enumerate(FEATURES)}
   assert all(r['positive']>=35 and r['negative']>=35 for r in counts.values()),'insufficient class support'
   for im in d['images']:
    digest=hashlib.sha256(im.tobytes()).hexdigest();assert digest not in seen,'duplicate image';seen.add(digest)
   audit[part]=counts
 write('corpus_audit.json',dict(counts=audit,duplicate_images=0,passed=True));event('corpus_verified',images=len(seen))
 cache={}
 def data(part):
  if part not in cache:
   d=np.load(OUT/(part+'.npz'));cache[part]=(torch.tensor(d['images'].transpose(0,3,1,2).copy(),device='cuda',dtype=torch.uint8),d['labels'],d['valid'])
  return cache[part]
 def predict(model,part,limit=None):
  x,y,v=data(part);n=len(x) if limit is None else limit;model.eval();ps=[]
  with torch.inference_mode():
   for i in range(0,n,32):ps.append(model(x[i:min(i+32,n)].float()/255).sigmoid().cpu().numpy())
  return y[:n],np.concatenate(ps),v[:n]
 x,y,v=data('train');yt=torch.tensor(y,device='cuda');vt=torch.tensor(v,device='cuda');confirm=[]
 # Each seed independently selects its checkpoint before any locked evaluation.
 for seed in protocol['seeds']:
  torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Model().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.02);best=-1;stale=0;history=[]
  for epoch in range(40):
   model.train();total=0;order=rng.permutation(len(x))
   for i in range(0,len(x),32):
    ids=order[i:i+32];a=x[ids].float()/255
    if rng.random()<.5:a=a.flip(3)
    logits=model(a);loss=(F.binary_cross_entropy_with_logits(logits,yt[ids],reduction='none')*vt[ids]).sum()/vt[ids].sum().clamp_min(1);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step();total+=float(loss.detach())*len(ids)
   t,p,valid=predict(model,'selection');scores=ap(t,p,valid);score=float(np.mean(scores));history.append(dict(epoch=epoch+1,loss=total/len(x),macro_ap=score,per_feature_ap=dict(zip(FEATURES,scores))))
   write(f'history_{seed}.json',history);event('training',seed=seed,epoch=epoch+1,loss=total/len(x),selection_macro_ap=score)
   if score>best+.001:best=score;stale=0;torch.save(model.state_dict(),OUT/f'model_{seed}.pt')
   else:stale+=1
   if epoch>=7 and stale>=6:break
  model.load_state_dict(torch.load(OUT/f'model_{seed}.pt',weights_only=True));t,p,valid=predict(model,'train',1200);train_ap=ap(t,p,valid);confirm.append(dict(seed=seed,selection_macro_ap=best,train_subset_macro_ap=float(np.mean(train_ap)),epochs=len(history)))
  del model,opt
 write('confirmation.json',confirm);event('models_frozen',confirmation=confirm);results={}
 for seed in protocol['seeds']:
  model=Model().cuda();model.load_state_dict(torch.load(OUT/f'model_{seed}.pt',weights_only=True));y,p,v=predict(model,'calibration');thresholds=[]
  for k in range(len(FEATURES)):
   ids=v[:,k]>0;t=y[ids,k].astype(bool);scores=p[ids,k];choices=[]
   for th in np.unique(np.r_[scores,1.000001]):
    a=scores>=th;tp=(a&t).sum();fp=(a&~t).sum();fn=(~a&t).sum();fpr=fp/max(1,(~t).sum())
    if fpr<=.1:choices.append((2*tp/max(1,2*tp+fp+fn),-float(th)))
   thresholds.append(-max(choices)[1])
  per={'thresholds':dict(zip(FEATURES,thresholds))}
  for part in ['locked_same','locked_font','locked_combinations','locked_recipe']:
   event('evaluation',seed=seed,partition=part);per[part]=metrics(*predict(model,part),thresholds);del cache[part]
  results[str(seed)]=per;write('evaluation_partial.json',results);del model
 # Frozen CNN replay on exactly the fresh test images, with original calibration thresholds.
 import importlib.util
 spec=importlib.util.spec_from_file_location('cnn_comparison',ROOT/'feature_multilabel_v1/train.py');cnn=importlib.util.module_from_spec(spec);spec.loader.exec_module(cnn)
 old=json.loads((ROOT/'feature_multilabel_v1/run/results.json').read_text());baseline={}
 for seed in protocol['seeds']:
  net=cnn.Model().cuda();net.load_state_dict(torch.load(ROOT/f'feature_multilabel_v1/run/model_{seed}.pt',weights_only=True));threshold=[old['results'][str(seed)]['thresholds'][f] for f in FEATURES];baseline[str(seed)]={}
  for part in ['locked_same','locked_font','locked_combinations','locked_recipe']:
   baseline[str(seed)][part]=metrics(*predict(net,part),threshold);del cache[part]
  del net
 write('cnn_fresh_comparison.json',baseline)
 passed={f:all(results[str(s)][part][f]['f1']>=.85 and results[str(s)][part][f]['recall']>=.85 and results[str(s)][part][f]['fpr']<=.1 for s in protocol['seeds'] for part in ['locked_same','locked_font','locked_combinations','locked_recipe']) for f in FEATURES}
 write('results.json',dict(per_feature_gate=passed,results=results,production_promoted=False,real_accuracy_verified=False))
 report='# Transformer multi-label v1 results\n\nSynthetic-only; production unchanged; real accuracy unverified.\n\n| Feature | All-seed/all-stratum gate |\n|---|---|\n'+''.join(f'| {f} | {v} |\n' for f,v in passed.items())+'\nSee run/results.json for per-feature confusion counts and thresholds.\n'
 (OUT/'RESULTS.md').write_text(report);(ROOT/'docs/V2_TRANSFORMER_MULTILABEL_RESULTS.md').write_text(report)
 for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
  with (ROOT/'docs'/name).open('a',encoding='utf8') as stream:stream.write('\n\nTransformer multi-label v1 finite run complete. See V2_TRANSFORMER_MULTILABEL_RESULTS.md; passed features='+str([f for f,v in passed.items() if v])+'. No production promotion.\n')
 event('complete',per_feature_gate=passed,production_promoted=False)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  OUT.mkdir(exist_ok=True);event('failed',error=repr(exc));raise
