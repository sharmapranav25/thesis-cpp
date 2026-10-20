"""Simple CNN trained on existing DarkBay CAPRI images, no new synthesis."""
import os
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,csv,hashlib,time,copy
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parent;DATA=Path('E:/pranav/Final sem/synth_data/darkbay_capri_v2_100k');W,H=192,64
def save(name,row):
 p=ROOT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(row,indent=2));q.replace(p)
class Images(Dataset):
 def __init__(self,rows):self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];im=Image.open(DATA/r['filename']).convert('RGB').resize((W,H),Image.Resampling.BILINEAR)
  return torch.from_numpy(np.array(im).transpose(2,0,1).copy()).float()/255,len(r['label'])-2
class CountCNN(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,24,5,2,2),nn.BatchNorm2d(24),nn.ReLU(),nn.Conv2d(24,48,3,2,1),nn.BatchNorm2d(48),nn.ReLU(),nn.Conv2d(48,96,3,2,1),nn.BatchNorm2d(96),nn.ReLU(),nn.AdaptiveAvgPool2d((2,12)),nn.Flatten(),nn.Linear(2304,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,7))
 def forward(self,x):return self.net(x)
@torch.no_grad()
def evaluate(net,loader,device):
 net.eval();ys=[];ps=[];loss=0
 for x,y in loader:
  x=x.to(device);y=y.to(device);z=net(x);loss+=float(F.cross_entropy(z,y,reduction='sum'));ys.extend(y.cpu().tolist());ps.extend(z.softmax(1).cpu().tolist())
 ys=np.array(ys);ps=np.array(ps);pred=ps.argmax(1)
 return dict(images=len(ys),loss=loss/len(ys),exact=float(np.mean(pred==ys)),mae=float(np.abs(pred-ys).mean()),per_length={str(k+2):dict(images=int((ys==k).sum()),exact=float(np.mean(pred[ys==k]==k))) for k in range(7)}),ys,ps
def main():
 started=time.time();torch.set_num_threads(4);torch.manual_seed(731);device='cuda' if torch.cuda.is_available() else 'cpu'
 assert json.loads((DATA/'audit.json').read_text())['passed'];rows=list(csv.DictReader((DATA/'labels.csv').open()));parts={k:[] for k in ['train','selection','test']}
 for r in rows:
  assert 2<=len(r['label'])<=8
  bucket=int(hashlib.sha256(('length-cnn03:'+r['label']).encode()).hexdigest()[:8],16)%100
  parts['train' if bucket<90 else 'selection' if bucket<95 else 'test'].append(r)
 strings={k:set(r['label'] for r in v) for k,v in parts.items()};assert not(strings['train']&strings['selection'] or strings['train']&strings['test'] or strings['selection']&strings['test'])
 save('membership.json',parts);protocol=dict(source=str(DATA),source_label_sha256=hashlib.sha256((DATA/'labels.csv').read_bytes()).hexdigest(),source_images=len(rows),split={k:len(v) for k,v in parts.items()},split_method='synthetic-text hash90/5/5, no shared strings; shared fonts and family profiles',architecture='three convolution blocks + pooled MLP, 7 lengths2–8',seed=731,epochs_max=6,patience=2,batch=128,optimizer='AdamW lr0.001 weight_decay0.0001',selection='lowest synthetic selection cross entropy',real_training=False,real_reference_set='entire DarkBay pool, evaluation directory excluded',real_accuracy_verified=False,limitation='existing count-dependent generator layout retained deliberately; this tests the requested corpus, not shortcut-free counting')
 save('protocol.json',protocol);loaders={k:DataLoader(Images(v),batch_size=128,shuffle=k=='train',num_workers=4,persistent_workers=True,pin_memory=device=='cuda') for k,v in parts.items()};net=CountCNN().to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);history=[];best=float('inf');stale=0
 for epoch in range(1,7):
  net.train();loss=0;correct=0;seen=0
  for x,y in loaders['train']:
   x=x.to(device);y=y.to(device);opt.zero_grad();z=net(x);l=F.cross_entropy(z,y);l.backward();opt.step();loss+=float(l)*len(y);correct+=int((z.argmax(1)==y).sum());seen+=len(y)
   if seen%12800==0:save('status.json',dict(state='training',epoch=epoch,seen=seen,total=len(parts['train']),elapsed=time.time()-started))
  metrics,_,_=evaluate(net,loaders['selection'],device);row=dict(epoch=epoch,train_loss=loss/seen,train_exact=correct/seen,selection=metrics,elapsed=time.time()-started);history.append(row);save('history.json',history);print(json.dumps(row),flush=True)
  if metrics['loss']<best:
   best=metrics['loss'];stale=0;torch.save(net.state_dict(),ROOT/'best.pt');save('selected.json',dict(epoch=epoch,selection=metrics))
  else:stale+=1
  if stale>=2:break
 net.load_state_dict(torch.load(ROOT/'best.pt',map_location=device,weights_only=True));test,y,ps=evaluate(net,loaders['test'],device);save('test.json',test);np.savez_compressed(ROOT/'test_predictions.npz',truth=y+2,probabilities=ps)
 save('status.json',dict(state='real_inference',elapsed=time.time()-started));pool=Path('E:/pranav/cryptbb_sourcefree/data/darkbay/pool');paths=sorted(p for p in pool.rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp']);real=[];net.eval();gallery=Image.new('RGB',(W*4,105*6),'white');draw=ImageDraw.Draw(gallery)
 with torch.inference_mode():
  for i in range(0,len(paths),128):
   subset=paths[i:i+128];ims=[Image.open(p).convert('RGB').resize((W,H),Image.Resampling.BILINEAR) for p in subset];x=torch.from_numpy(np.stack([np.array(im) for im in ims])).permute(0,3,1,2).float().to(device)/255;prob=net(x).softmax(1).cpu().numpy()
   for j,(p,im,q) in enumerate(zip(subset,ims,prob)):
    n=int(q.argmax()+2);real.append(dict(path=str(p),predicted_length=n,probabilities=q.tolist()))
    if i+j<24:
     k=i+j;xx=k%4*W;yy=k//4*105;draw.text((xx+3,yy+3),f'predicted {n}; score {q.max():.2f}',fill='black');gallery.paste(im,(xx,yy+25))
 save('real_predictions.json',real);gallery.save(ROOT/'darkbay_predictions.png');pred=[r['predicted_length'] for r in real];save('real_summary.json',dict(images=len(pred),histogram={str(n):pred.count(n) for n in range(2,9)},ground_truth_used=False,accuracy_verified=False));save('status.json',dict(state='complete',elapsed=time.time()-started,test=test,real_images=len(real)));print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:save('failure.json',dict(error=repr(e)));raise
