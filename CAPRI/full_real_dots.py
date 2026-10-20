import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,json,time,hashlib,shutil
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
root=Path('E:/pranav/Final sem/sourcefree_synth');base=root/'feature_specialists_v2/dots';sys.path.insert(0,str(base))
from train_dots import Classifier
from corpus import pop
out=Path(__file__).parent/'full_real_dot_review_02';out.mkdir(exist_ok=True)
def status(**kw):
 row=dict(time=time.time(),pid=os.getpid(),**kw);tmp=out/'status.tmp';tmp.write_text(json.dumps(row,indent=2));tmp.replace(out/'status.json');print(json.dumps(row),flush=True)
def main():
 import msvcrt
 lock=(out/'job.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 assert not (out/'summary.json').exists(),'Completed run preserved'
 checkpoint=Path(__file__).parent/'real_dot_review_01/frozen_model.pt';shutil.copy2(checkpoint,out/'frozen_model.pt');torch.set_num_threads(2)
 s=torch.load(checkpoint,map_location='cpu',weights_only=True);m=Classifier(s['grids'],s['use_pop']);m.load_state_dict(s['state_dict']);m.eval()
 families=['cryptbb','city','courier','darkbay','outlined_text_captcha'];manifest={};reserved={}
 for family in families:
  records=json.loads((root/f'runs/style_phase1_05/{family}_measurements.json').read_text());folder=Path(records['fit'][0]['path']).parent
  if family=='darkbay':folder=folder.parent # include the sibling eval directory for the explicitly requested whole-dataset pass
  manifest[family]=[str(p) for p in sorted(folder.rglob('*')) if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.gif','.webp')]
  reserved[family]={str(Path(r['path'])) for r in records['validation']}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 (out/'protocol.json').write_text(json.dumps(dict(model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),view='5x3 + dot pop',threshold=.5,threshold_status='provisional; no real fitting',scope='all image files recursively in each established source image directory',counts={k:len(v) for k,v in manifest.items()},boundaries='User explicitly requests whole datasets. Previously reserved validation paths tagged in output; now distribution-exposed, never claim unseen in later development.',preprocessing='RGB bilinear resize128x64 then morphology pop',device='CPU; training GPU left available'),indent=2))
 summary={}
 for family,paths in manifest.items():
  rows=[];errors=[];status(stage='inference',family=family,done=0,total=len(paths))
  with (out/(family+'.jsonl')).open('w') as stream:
   for start in range(0,len(paths),16):
    images=[];valid=[]
    for path in paths[start:start+16]:
     try:
      with Image.open(path) as im:images.append(np.array(im.convert('RGB').resize((128,64),Image.Resampling.BILINEAR)))
      valid.append(path)
     except Exception as e:errors.append(dict(path=path,error=repr(e)))
    if images:
     a=np.stack(images);x=torch.tensor(np.concatenate([a,pop(a)[...,None]],3).transpose(0,3,1,2)).float()/255
     with torch.inference_mode():scores=m(x).sigmoid().numpy()
     for path,p in zip(valid,scores):
      row=dict(path=path,score=float(p),present=bool(p>=.5),previously_reserved_validation=path in reserved[family]);rows.append(row);stream.write(json.dumps(row)+'\n')
    if start%256==0:stream.flush();status(stage='inference',family=family,done=min(start+16,len(paths)),total=len(paths))
  scores=np.array([r['score'] for r in rows]);summary[family]=dict(discovered=len(paths),scored=len(rows),errors=errors,present_count=int((scores>=.5).sum()),present_fraction=float((scores>=.5).mean()) if len(scores) else None,score_quantiles=np.quantile(scores,[0,.1,.25,.5,.75,.9,1]).tolist() if len(scores) else [],accuracy='not established by unlabeled inference')
  ranked=sorted(rows,key=lambda r:r['score']);sheet=Image.new('RGB',(900,600),'#eeeeee');d=ImageDraw.Draw(sheet)
  for j,index in enumerate(np.linspace(0,len(ranked)-1,12).astype(int)):
   r=ranked[index];im=Image.open(r['path']).convert('RGB');im.thumbnail((290,115));x=j%3*300;y=j//3*150;sheet.paste(im,(x,y+25));d.text((x+3,y+3),f"rank {index+1}/{len(ranked)} score {r['score']:.3f}",fill='black')
  sheet.save(out/(family+'_distribution.png'));(out/'partial_summary.json').write_text(json.dumps(summary,indent=2));status(stage='family_complete',family=family,summary=summary[family])
 (out/'summary.json').write_text(json.dumps(summary,indent=2));status(stage='complete',counts={k:v['scored'] for k,v in summary.items()})
if __name__=='__main__':
 try:main()
 except BaseException as e:status(stage='failed',error=repr(e));raise
