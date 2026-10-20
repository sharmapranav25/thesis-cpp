"""Synthetic-only training; independent random recipes and held-out font families."""
from boot import ROOT,U
import json,time,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from mask_model import MaskNet
import render_control as R

def generate(item):
    seed,fontkeys=item;rng=np.random.default_rng(seed);bank=R.bank()
    ts=R.T.sample_text_spec(rng,bank,charset=str(rng.choice(['digits','upper','lower','mixed','upper_digits'])),layout='line')
    ts.update(fonts=[str(rng.choice(fontkeys))],size=float(rng.uniform(.45,.86)),rotation=float(rng.uniform(0,25)),spacing=float(rng.uniform(-.12,.35)),multicolour=bool(rng.random()<.5))
    es=R.R.sample_spec_k(rng,int(rng.integers(0,5)),levels=range(1,8))
    # Balanced difficult phenomena, independent of real target images or labels.
    group=seed%4
    if group==1:
        es['effects']=[e for e in es['effects'] if e['name'] not in ['bold','thin','hollow']]
        es['effects'].append(dict(name='hollow',level=10));ts['bold_stroke']=int(rng.integers(1,3))
    if group in [1,2]:es['effects'].append(dict(name=str(rng.choice(['lines','bands','scribbles'])),level=int(rng.integers(1,6))))
    if group==3:ts['multicolour']=True
    out=R.C.render_captcha(rng,bank,(int(rng.integers(180,301)),64),es,ts)
    if out is None:raise RuntimeError('Empty synthetic sample')
    ids=[r['id'] for r in out['records'] if r.get('head')=='text']
    mask=np.isin(out['opaque'],ids).astype(np.uint8)*255
    image=np.asarray(Image.fromarray(out['image']).resize((256,64),Image.Resampling.BILINEAR))
    mask=np.asarray(Image.fromarray(mask).resize((256,64),Image.Resampling.NEAREST))
    return image,mask,dict(seed=seed,group=group,font=ts['fonts'][0],text=ts['text'],effects=es)

@torch.inference_mode()
def evaluate(model,x,y,groups,device):
    values=[]
    for i in range(0,len(x),32):
        a=torch.tensor(x[i:i+32].transpose(0,3,1,2).copy(),device=device).float()/255
        truth=torch.tensor(y[i:i+32]>127,device=device)
        pred=model(a)[:,0].sigmoid()>.5
        values.extend(((pred&truth).sum((1,2))/(pred|truth).sum((1,2)).clamp_min(1)).cpu().tolist())
    values=np.asarray(values)
    return dict(mean_iou=float(values.mean()),groups={str(g):float(values[np.asarray(groups)==g].mean()) for g in range(4)})

def main():
    out=ROOT/'mask_training';out.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(63219);np.random.seed(63219)
    torch.backends.cudnn.benchmark=True;device='cuda' if torch.cuda.is_available() else 'cpu'
    fonts=U.read(U.WORK/'learned_controller_20261004/fonts.json');families=sorted(set(f['family'] for f in fonts));np.random.default_rng(63219).shuffle(families)
    held=set(families[:max(1,len(families)//6)]);trainfonts=[f['key'] for f in fonts if f['family'] not in held];valfonts=[f['key'] for f in fonts if f['family'] in held]
    counts=dict(train=4096,validation=512);metadata={}
    for split,n in counts.items():
        path=out/(split+'.npz')
        if not path.exists():
            offset=100000 if split=='train' else 900000
            keys=trainfonts if split=='train' else valfonts
            with ProcessPoolExecutor(max_workers=8) as pool:rows=list(pool.map(generate,[(offset+i,keys) for i in range(n)],chunksize=8))
            np.savez_compressed(path,images=np.stack([r[0] for r in rows]),masks=np.stack([r[1] for r in rows]))
            U.dump(out/(split+'_recipes.json'),[r[2] for r in rows])
        metadata[split]=U.read(out/(split+'_recipes.json'))
        print(json.dumps(dict(stage='data_ready',split=split,n=n)),flush=True)
    U.dump(out/'protocol.json',dict(seed=63219,counts=counts,trainfonts=trainfonts,validationfonts=valfonts,target_images_used=0,vlm=False,
        mask_definition='Visible original glyph ink after effects/occlusion; decorative shadows and outline expansion excluded.',epochs=20,selection='Best mean synthetic validation IoU; target pixels never used.',source_sha256={p.name:U.sha(p) for p in [ROOT/'mask_model.py',ROOT/'train_mask.py']}))
    train=np.load(out/'train.npz');val=np.load(out/'validation.npz');x=train['images'];y=train['masks'];groups=[r['group'] for r in metadata['validation']]
    model=MaskNet().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=-1;history=[];start=time.time()
    for epoch in range(20):
        model.train();losses=[]
        for idx in np.array_split(np.random.permutation(len(x)),len(x)//32):
            a=torch.tensor(x[idx].transpose(0,3,1,2).copy(),device=device).float()/255;b=torch.tensor(y[idx,None].copy(),device=device).float()/255
            # Geometry-preserving photometric augmentation: labels remain valid.
            gain=torch.rand(len(a),1,1,1,device=device)*.4+.8;a=(a*gain+torch.randn_like(a)*.015).clamp(0,1)
            logits=model(a);p=logits.sigmoid();dice=1-(2*(p*b).sum((1,2,3))+1)/(p.sum((1,2,3))+b.sum((1,2,3))+1)
            loss=F.binary_cross_entropy_with_logits(logits,b)+dice.mean();optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval();metrics=evaluate(model,val['images'],val['masks'],groups,device)
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=round(time.time()-start,1),**metrics);history.append(row);print(json.dumps(row),flush=True)
        if metrics['mean_iou']>best:
            best=metrics['mean_iou'];torch.save(dict(model=model.state_dict(),metrics=row),out/'best.pt')
        U.dump(out/'history.json',history)
    # Evaluate the old mask encoder on exactly the same synthetic validation pixels.
    from comparison import Comparator
    old=Comparator();f=old.extract(list(val['images']));p=f['mask']>.5;t=val['masks']>127
    iou=(p&t).sum((1,2))/np.maximum(1,(p|t).sum((1,2)))
    U.dump(out/'comparison.json',dict(old=dict(mean_iou=float(iou.mean()),groups={str(g):float(iou[np.asarray(groups)==g].mean()) for g in range(4)}),new=torch.load(out/'best.pt',weights_only=False)['metrics'],groups={'0':'random effects','1':'hollow plus interference','2':'crossing interference','3':'multicolor'},limitation='Synthetic validation only; not real target mask accuracy.',checkpoint_sha256=U.sha(out/'best.pt')))

if __name__=='__main__':main()
