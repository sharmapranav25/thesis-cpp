import hashlib,json,time
from pathlib import Path
import numpy as np
import torch,yaml
from PIL import Image,ImageDraw
from feature_bank.iter2.model import Iter2Model
from feature_bank.iter2 import metrics as M
from feature_bank.train.data import TOK_W,MARGIN
root=Path('E:/pranav/Final sem/sourcefree_synth/runs/iter2_baseline_15_20261003_04')
out=Path('C:/Users/pranav/Documents/ChatGPT/thesis/ctc_city_courier_01');out.mkdir(exist_ok=False)
# Use the same checkpoint as the DarkBay/CryptBB comparison.
checkpoint=root/'baseline_training/epoch_10/ema.pt'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
model=Iter2Model(**yaml.safe_load((root/'training_config.yaml').read_text())['model']).cpu().eval()
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.requires_grad_(False)
families={'City':Path('E:/pranav/City (live)-20260503T045450Z-3-001/City (live)'),
          'Courier':Path('E:/pranav/Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)')}
results=[];start=time.perf_counter()
with torch.inference_mode():
    for family,directory in families.items():
        paths=sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
        selected=np.random.default_rng(42).choice(len(paths),32,replace=False)
        for begin in range(0,32,4):
            batch=[paths[int(i)] for i in selected[begin:begin+4]]
            originals=[Image.open(p).convert('RGB') for p in batch]
            ims=[np.asarray(im.resize((round(im.width*96/im.height),96),Image.Resampling.BILINEAR),dtype=np.uint8) for im in originals]
            cols=np.asarray([(im.shape[1]+TOK_W-1)//TOK_W for im in ims],dtype=np.int64)
            wp=((max(int(cols.max())*TOK_W,max(im.shape[1] for im in ims))+MARGIN+TOK_W-1)//TOK_W)*TOK_W
            pixels=np.stack([np.pad(im,((0,0),(0,wp-im.shape[1]),(0,0)),mode='edge') for im in ims]).transpose(0,3,1,2).copy()
            readings=M.decode(model(torch.from_numpy(pixels).float()/255,torch.from_numpy(cols))['ctc'].numpy(),cols*model.ctc_frames)
            for j,(path,im,reading) in enumerate(zip(batch,originals,readings)):
                scale=2 if im.width>250 else 3
                enlarged=im.resize((im.width*scale,im.height*scale),Image.Resampling.NEAREST)
                canvas=Image.new('RGB',(max(600,enlarged.width),enlarged.height+60),'white');canvas.paste(enlarged,(0,0))
                draw=ImageDraw.Draw(canvas);draw.text((8,enlarged.height+8),f'{family} | CTC: {reading} | no verified answer label',fill='black')
                draw.text((8,enlarged.height+30),'Epoch 10 EMA; random seed 42; CPU; real image',fill='black')
                preview=out/f'{family}_{begin+j:02d}.png';canvas.save(preview)
                results.append(dict(family=family,file=str(path),prediction=reading,preview=str(preview)))
            print(f'{family}: {begin+len(batch)}/32',flush=True)
report=dict(checkpoint=str(checkpoint),epoch=10,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),seed=42,device='cpu',threads=1,seconds=time.perf_counter()-start,note='Unlabelled real inference only. No accuracy claimed. No real-data training or tuning. Same epoch 10 EMA as DarkBay/CryptBB.',results=results)
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
