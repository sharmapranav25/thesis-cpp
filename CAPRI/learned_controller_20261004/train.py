import os
os.environ['OMP_NUM_THREADS']='4'
import time,json,copy
import numpy as np
import torch
from torch.nn import functional as F
import scheme as S
from network import Encoder,Controller
torch.set_num_threads(4);torch.manual_seed(719);np.random.seed(719)
torch.backends.cudnn.benchmark=True
ROOT=S.ROOT;DATA=ROOT/'corpus';OUT=ROOT/'training'
def cat_loss(logits,target):return sum(F.cross_entropy(v.float(),target[:,i]) for i,v in enumerate(logits))/len(logits)
def main():
 OUT.mkdir(exist_ok=False);start=time.time()
 ims=np.load(DATA/'images.npy',mmap_mode='r');masks=np.load(DATA/'masks.npy',mmap_mode='r');lab=np.load(DATA/'labels.npz');split=np.load(DATA/'split.npz')
 c=torch.from_numpy(lab['cont']).cuda();k=torch.from_numpy(lab['cat']).cuda();net=Encoder().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
 def batch(ids):
  x=torch.from_numpy(np.array(ims[ids])).reshape(-1,64,256,3).permute(0,3,1,2).cuda().float()/255
  m=torch.from_numpy(np.array(masks[ids])).reshape(-1,1,64,256).cuda().float()/255
  return x,m,c[ids].repeat_interleave(2,0),k[ids].repeat_interleave(2,0)
 best=1e9;history=[];rng=np.random.default_rng(719)
 for epoch in range(1,25):
  net.train();losses=[]
  order=rng.permutation(split['train'])
  for start_i in range(0,len(order),32):
   ids=order[start_i:start_i+32];x,m,tc,tk=batch(ids);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    out=net(x);reg=F.mse_loss(out['cont'].float(),tc);cl=cat_loss(out['cat'],tk)
    seg=F.binary_cross_entropy_with_logits(out['mask'].float(),m,pos_weight=torch.tensor(3.,device='cuda'))
    prob=out['mask'].float().sigmoid();dice=1-(2*(prob*m).sum((1,2,3))+1)/(prob.sum((1,2,3))+m.sum((1,2,3))+1)
    z=out['z'].float();similar=z@z.T/.15;similar.fill_diagonal_(-1e4);positive=torch.arange(len(z),device='cuda')^1
    contrast=F.cross_entropy(similar,positive)
    loss=8*reg+cl+seg+.5*dice.mean()+.35*contrast
   loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(float(loss.detach()))
  net.eval();mae=[];acc=[];ious=[];zs=[];closs=[]
  with torch.inference_mode():
   for st in range(0,len(split['val']),32):
    ids=split['val'][st:st+32];x,m,tc,tk=batch(ids)
    with torch.autocast('cuda',dtype=torch.bfloat16):o=net(x)
    mae.append(float(abs(o['cont'].float()-tc).mean()));acc.append([float((p.argmax(1)==tk[:,j]).float().mean()) for j,p in enumerate(o['cat'])])
    pred=o['mask'].sigmoid()>.5;truth=m>.5;ious.append(float(((pred&truth).sum((1,2,3))/((pred|truth).sum((1,2,3))+1)).float().mean()))
    zs.append(o['z'].float().cpu().numpy().reshape(-1,2,128));closs.append(float(cat_loss(o['cat'],tk)))
  z=np.concatenate(zs);sim=z[:,0]@z[:,1].T;retrieval=float(np.mean(sim.argmax(1)==np.arange(len(z))))
  val=8*np.mean(mae)+np.mean(closs)+(1-np.mean(ious))
  record=dict(epoch=epoch,loss=float(np.mean(losses)),val_cont_mae=float(np.mean(mae)),val_categories=dict(zip(S.KN,np.mean(acc,0).tolist())),val_mask_iou=float(np.mean(ious)),val_scheme_retrieval=retrieval,seconds=round(time.time()-start,1))
  history.append(record);S.dump(OUT/'encoder_history.json',history);print(json.dumps(record),flush=True)
  if val<best:
   best=val;torch.save(dict(model=net.state_dict(),metrics=record),OUT/'encoder.pt')
  if epoch in (12,20):
   for group in opt.param_groups:group['lr']*=.35
 net.load_state_dict(torch.load(OUT/'encoder.pt',weights_only=False)['model']);net.eval();features=[]
 with torch.inference_mode():
  for st in range(0,len(ims),32):
   x,_,_,_=batch(np.arange(st,min(st+32,len(ims))))
   with torch.autocast('cuda',dtype=torch.bfloat16):o=net(x)
   features.append(o['z'].float().reshape(-1,2,128).mean(1).cpu().numpy())
 features=np.concatenate(features);np.save(OUT/'scheme_embeddings.npy',features);z=torch.from_numpy(features).cuda()
 policy=Controller().cuda();opt=torch.optim.AdamW(policy.parameters(),lr=.001,weight_decay=.0001);hist=[];best=1e9
 train=split['train'];valid=split['val'];vi=torch.from_numpy(valid).cuda();other=vi^1
 for epoch in range(1,81):
  policy.train();ls=[]
  for _ in range(24):
   ids=rng.choice(train,512);goal=ids^1
   random=rng.random(len(ids))<.15;goal[random]=rng.choice(train,int(random.sum()))
   same=rng.random(len(ids))<.15;goal[same]=ids[same]
   ii=torch.from_numpy(ids).cuda();gg=torch.from_numpy(goal).cuda();o=policy(z[gg],z[ii],c[ii],k[ii]);loss=15*F.mse_loss(o['cont'],c[gg])+cat_loss(o['cat'],k[gg]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();ls.append(float(loss.detach()))
  if epoch%5==0:
   policy.eval()
   with torch.inference_mode():
    o=policy(z[other],z[vi],c[vi],k[vi]);vl=float(15*F.mse_loss(o['cont'],c[other])+cat_loss(o['cat'],k[other]));mae=float(abs(o['cont']-c[other]).mean());baseline=float(abs(c[vi]-c[other]).mean());accur=[float((v.argmax(1)==k[other,j]).float().mean()) for j,v in enumerate(o['cat'])]
   rec=dict(epoch=epoch,loss=float(np.mean(ls)),val_loss=vl,correction_mae=mae,uncorrected_mae=baseline,categories=dict(zip(S.KN,accur)),seconds=round(time.time()-start,1));hist.append(rec);S.dump(OUT/'controller_history.json',hist);print(json.dumps(rec),flush=True)
   if vl<best:best=vl;torch.save(dict(model=policy.state_dict(),metrics=rec),OUT/'controller.pt')
 S.dump(OUT/'complete.json',dict(seconds=time.time()-start,encoder_parameters=sum(p.numel() for p in net.parameters()),controller_parameters=sum(p.numel() for p in policy.parameters()),training_images=len(split['train'])*2,validation_images=len(split['val'])*2))
if __name__=='__main__':main()
