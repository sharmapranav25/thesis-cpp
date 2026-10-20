import os
os.environ['OMP_NUM_THREADS']='1'
import json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image,ImageDraw
import scheme as S
def task(item):
 i,p=item;rows=[]
 for view in range(2):
  r=S.render((p,110000+i*17+view,(256,64),False))
  rows.append((r['image'],r['mask'],r['usable']))
 return i,rows
def main():
 S.make_fonts();out=S.ROOT/'corpus';out.mkdir(exist_ok=False);N=4096
 recipes=[]
 for i in range(N//2):
  p=S.sample(990000+i);recipes.extend([p,S.perturb(p,880000+i)])
 S.dump(out/'recipes.json',recipes)
 images=np.lib.format.open_memmap(out/'images.npy',mode='w+',dtype=np.uint8,shape=(N,2,64,256,3))
 masks=np.lib.format.open_memmap(out/'masks.npy',mode='w+',dtype=np.uint8,shape=(N,2,64,256))
 labels=[S.encode(p) for p in recipes];np.savez(out/'labels.npz',cont=np.stack([x[0] for x in labels]),cat=np.stack([x[1] for x in labels]))
 # Base/perturbed sibling pairs never cross the training/validation boundary.
 order=np.random.default_rng(7681).permutation(N//2);val=set(order[:256].tolist())
 np.savez(out/'split.npz',train=np.array([i for i in range(N) if i//2 not in val]),val=np.array([i for i in range(N) if i//2 in val]))
 flags=[];start=time.time()
 with ProcessPoolExecutor(max_workers=16) as pool:
  for i,rows in pool.map(task,enumerate(recipes),chunksize=4):
   for j,(im,mask,ok) in enumerate(rows):images[i,j]=im;masks[i,j]=mask;flags.append(ok)
   if (i+1)%128==0:
    images.flush();masks.flush();status=dict(stage='corpus',schemes=i+1,total=N,seconds=round(time.time()-start,1),usable=float(np.mean(flags)));S.dump(out/'status.json',status);print(json.dumps(status),flush=True)
 sheet=Image.new('RGB',(1024,64*8),'white')
 for i in range(32):sheet.paste(Image.fromarray(images[i,0]),((i%4)*256,(i//4)*64))
 sheet.save(out/'preview.jpg');S.dump(out/'complete.json',dict(schemes=N,images=N*2,usable=float(np.mean(flags)),seconds=time.time()-start))
if __name__=='__main__':main()
