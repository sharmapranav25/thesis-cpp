"""Finite two-stage learnability experiment; synthetic only, no release promotion."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import json,sys,time,hashlib,math
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0,str(ROOT/'feature_specialists_v2'))
from corpus import fonts,RELEASE
HERE=Path(__file__).parent;OUT=HERE/'run';OUT.mkdir(exist_ok=True)
torch.set_num_threads(2);cv2.setNumThreads(1)

def event(stage,**kw):
 row=dict(time=time.time(),pid=os.getpid(),stage=stage,**kw)
 temp=OUT/'status.tmp';temp.write_text(json.dumps(row,indent=2));temp.replace(OUT/'status.json')
 with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)

def sample(seed,bank,with_text=True):
 rng=np.random.default_rng(seed);kind=seed%3 # none, straight, curve
 bg=rng.integers(0,256,3)
 arr=np.broadcast_to(bg,(64,128,3)).copy().astype(float)
 if seed%2:arr+=np.linspace(-65,65,128)[None,:,None]
 base=Image.fromarray(np.uint8(np.clip(arr,0,255)));d=ImageDraw.Draw(base)
 if seed%5==0:
  for _ in range(5):
   xx,yy=rng.integers([0,0],[110,48]);d.rectangle((int(xx),int(yy),int(xx+15),int(yy+16)),fill=tuple(rng.integers(0,256,3)))
 if with_text:
  font=ImageFont.truetype(str(RELEASE/bank[int(rng.integers(len(bank)))]['path']),int(rng.integers(20,34)))
  text=''.join(rng.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'),int(rng.integers(2,9))))
  d.text((int(rng.integers(0,10)),int(rng.integers(2,22))),text,font=font,fill=tuple(rng.integers(0,256,3)))
 mask=Image.new('L',(128,64));ink=tuple(rng.integers(0,256,3))
 if kind:
  while np.linalg.norm(np.array(ink)-bg)<80:ink=tuple(rng.integers(0,256,3))
  before=np.asarray(base).copy()
  t=np.linspace(-1,1,100);y=np.zeros_like(t) if kind==1 else rng.choice([-1,1])*rng.uniform(.35,.7)*(1-t*t)
  xy=np.stack([t,y-y.mean()],1);a=rng.uniform(0,np.pi);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
  xy=xy@rot.T;xy*=min(46/max(np.ptp(xy[:,0]),.1),40/max(np.ptp(xy[:,1]),.1),rng.uniform(25,40));xy+=rng.uniform([35,24],[93,40])
  points=[tuple(p) for p in np.rint(xy).astype(int)];width=int(rng.integers(1,4))
  ImageDraw.Draw(mask).line(points,fill=255,width=width);d.line(points,fill=ink,width=width)
 target=np.asarray(mask).copy()
 if kind:
  delta=np.max(np.abs(np.asarray(base).astype(float)-before.astype(float)),axis=2)
  target[delta<12]=0
  assert np.count_nonzero(target)>=12,'Target must remain visible'
 return np.asarray(base),target,kind

def tiles(x):
 b,c,h,w=x.shape
 assert (h,w)==(64,128)
 x=F.pad(x,(0,2,0,2),mode='replicate')
 return x.reshape(b,c,3,22,10,13).permute(0,2,4,1,3,5).reshape(b*30,c,22,13)

def stitch(x,b):
 c=x.shape[1]
 return x.reshape(b,3,10,c,22,13).permute(0,3,1,4,2,5).reshape(b,c,66,130)[:,:,:64,:128]

class Detector(nn.Module):
 def __init__(self):
  super().__init__()
  self.net=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,1,1))
 def forward(self,x):
  return stitch(self.net(tiles(x)),len(x))

class Shape(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(1,16,5,padding=2),nn.ReLU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,stride=2,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,8)),nn.Flatten(),nn.Linear(1024,64),nn.ReLU(),nn.Linear(64,2))
 def forward(self,x):return self.net(x)

def main():
 import msvcrt
 lock=(OUT/'job.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 if (OUT/'results.json').exists():raise RuntimeError('Completed experiment; create a new version')
 torch.manual_seed(941);rng=np.random.default_rng(941);bank=fonts();device='cuda';assert torch.cuda.is_available()
 protocol={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'synthetic_only':True,'splits':{'train':[111000000,2400],'selection':[112000000,600],'locked':[113000000,600]},'view':[10,3],'feature_pop':False,'detector_epochs':24,'shape_epochs':16,'threshold':.5,'gate':{'positive_iou':.7,'none_false_positive_rate':.05,'end_to_end_shape_accuracy':.85},'limitations':'10 columns x3 rows, shared raw RGB patch detector; stitched mask; no cross-tile context; one target maximum; colour/gradient/block variety; no multi-instance grouping'}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));data={}
 for part,(start,n) in protocol['splits'].items():
  chosen=bank[:6] if part=='train' else bank[:7] if part=='selection' else bank[7:9]
  for text in ([False,True] if part=='train' else [True]):
   rows=[sample(start+i,chosen,text) for i in range(n)];images=np.stack([r[0] for r in rows]);masks=np.stack([r[1] for r in rows]);labels=np.array([r[2] for r in rows]);assert all((masks[i]>0).any()==bool(labels[i]) for i in range(n))
   key=part+('_clean' if not text else '');np.savez_compressed(OUT/(key+'.npz'),images=images,masks=masks,labels=labels)
   data[key]=(torch.tensor(images.transpose(0,3,1,2),device=device).float()/255,torch.tensor(masks[:,None],device=device).float()/255,torch.tensor(labels,device=device))
   event('corpus',partition=key,images=n)
 detector=Detector().to(device);shape=Shape().to(device)
 def predict(net,x):
  net.eval()
  with torch.no_grad():return torch.cat([net(x[i:i+32]) for i in range(0,len(x),32)])
 def iou(p,m,l):
  a=p.sigmoid()>=.5;b=m>.5;inter=(a&b).sum((1,2,3));union=(a|b).sum((1,2,3));return float((inter[l>0]/union[l>0].clamp_min(1)).mean())
 opt=torch.optim.AdamW(detector.parameters(),lr=.001);best=-1
 for epoch in range(24):
  key='train_clean' if epoch<4 else 'train';x,m,l=data[key];detector.train();losses=[]
  for ids in np.array_split(rng.permutation(len(x)),75):
   p=detector(x[ids]);bce=F.binary_cross_entropy_with_logits(p,m[ids],pos_weight=torch.tensor(15.,device=device));q=p.sigmoid();dice=1-(2*(q*m[ids]).sum()+1)/(q.sum()+m[ids].sum()+1);loss=bce+dice;opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
  sx,sm,sl=data['selection'];score=iou(predict(detector,sx),sm,sl);event('detector_training',epoch=epoch+1,curriculum=key,loss=float(np.mean(losses)),selection_iou=score)
  if epoch>=4 and score>best:best=score;torch.save(detector.state_dict(),OUT/'detector.pt')
 detector.load_state_dict(torch.load(OUT/'detector.pt',weights_only=True));opt=torch.optim.AdamW(shape.parameters(),lr=.001);best=-1
 # Train shape on exact synthetic masks; predicted-mask transfer explicitly tested below.
 _,m,l=data['train'];m=m[l>0];l=l[l>0]-1
 for epoch in range(16):
  shape.train();losses=[]
  for ids in np.array_split(rng.permutation(len(m)),50):
   logits=shape(m[ids]);loss=F.cross_entropy(logits,l[ids]);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
  _,sm,sl=data['selection'];score=float((predict(shape,sm[sl>0]).argmax(1)==sl[sl>0]-1).float().mean());event('shape_training',epoch=epoch+1,loss=float(np.mean(losses)),oracle_selection_accuracy=score)
  if score>best:best=score;torch.save(shape.state_dict(),OUT/'shape.pt')
 shape.load_state_dict(torch.load(OUT/'shape.pt',weights_only=True));results={}
 for part in ['selection','locked']:
  x,m,l=data[part];p=predict(detector,x);pred=p.sigmoid()>=.5;found=pred.sum((1,2,3))>=12;shape_pred=predict(shape,pred.float()).argmax(1)+1;end=torch.where(found,shape_pred,0)
  results[part]={'positive_iou':iou(p,m,l),'none_false_positive_rate':float(found[l==0].float().mean()),'detection_recall':float(found[l>0].float().mean()),'oracle_shape_accuracy':float((predict(shape,m[l>0]).argmax(1)==l[l>0]-1).float().mean()),'end_to_end_shape_accuracy':float((end[l>0]==l[l>0]).float().mean()),'all_class_accuracy':float((end==l).float().mean())}
  if part=='locked':
   np.savez_compressed(OUT/'locked_predictions.npz',masks=pred.cpu().numpy(),classes=end.cpu().numpy())
 r=results['locked'];passed=r['positive_iou']>=.7 and r['none_false_positive_rate']<=.05 and r['end_to_end_shape_accuracy']>=.85
 results.update(diagnostic_gate_pass=passed,production_promoted=False);(OUT/'results.json').write_text(json.dumps(results,indent=2));event('complete',**results)

if __name__=='__main__':
 try:main()
 except BaseException as e:event('failed',error=repr(e));raise
