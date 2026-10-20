import sys,json,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
base=Path('E:/pranav/Final sem/sourcefree_synth/feature_specialists_v2')
sys.path.insert(0,str(base))
from train_curves import Classifier
torch.set_num_threads(2)
run=base/'runs/curves_v1'
d=np.load(run/'train.npz');x=torch.tensor(np.concatenate([d['images'],d['pop'][...,None]],3).transpose(0,3,1,2),device='cuda',dtype=torch.uint8);y=torch.tensor(d['labels'],device='cuda')
out={}
saved=torch.load(run/'confirm_431.pt',weights_only=True);m=Classifier(saved['grids'],saved['use_pop']).cuda();m.load_state_dict(saved['state_dict']);m.eval()
with torch.no_grad():p=torch.cat([m(x[i:i+16].float()/255).sigmoid() for i in range(0,512,16)]).cpu().numpy()
out['checkpoint_train512']={'ap':float(average_precision_score(y[:512].cpu(),p)),'min':float(p.min()),'max':float(p.max()),'std':float(p.std()),'pair_positive_wins':float((p[1::2]>p[::2]).mean())}
for lr in [.0007,.0001]:
 torch.manual_seed(431);m=Classifier([[1,1]],True).cuda();opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.01);history=[]
 # Fixed balanced 16 images / 8 pairs: an intentional memorization diagnostic.
 a=x[:16].float()/255;b=y[:16]
 for step in range(401):
  logits=m(a);loss=F.binary_cross_entropy_with_logits(logits,b);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  if step%100==0:history.append({'step':step,'loss':float(loss.detach()),'accuracy':float(((logits>0)==b.bool()).float().mean()),'logit_std':float(logits.detach().std())})
 out[str(lr)]=history
 print(json.dumps(out),flush=True)
Path('curve_diagnosis.json').write_text(json.dumps(out,indent=2))
