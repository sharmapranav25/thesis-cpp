"""Evaluate the independent length head on real labelled images, CPU only."""
import csv,hashlib,json,re,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch,yaml
from PIL import Image,ImageDraw
from feature_bank.iter2.model import Iter2Model
from feature_bank.train.data import TOK_W,MARGIN

root=Path('E:/pranav/Final sem/sourcefree_synth/runs/iter2_baseline_15_20261003_04')
out=Path('C:/Users/pranav/Documents/ChatGPT/thesis/real_length_test_01');out.mkdir(exist_ok=False)
checkpoint_dir=sorted(p for p in (root/'baseline_training').glob('epoch_*') if (p/'scores.json').exists() and (p/'ema.pt').exists())[-1]
checkpoint=checkpoint_dir/'ema.pt'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
model=Iter2Model(**yaml.safe_load((root/'training_config.yaml').read_text())['model']).cpu().eval()
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.requires_grad_(False)
def labelled(directory):
    directory=Path(directory)
    with (directory/'labels.csv').open(newline='',encoding='utf-8-sig') as f:
        return [(directory/r['filename'],len(r['label'])) for r in csv.DictReader(f)]
city=Path('E:/john/Captcha Solver/test_set/City')
families={'DarkBay':labelled('E:/pranav/cryptbb_sourcefree/data/darkbay/eval'),
          'City':[(p,len(p.stem)) for p in sorted(city.glob('*.png')) if re.fullmatch('[0-9]{5}',p.stem)],
          'CryptBB':labelled('E:/pranav/model 1/data/eval')}
results=[];summaries={};start=time.perf_counter()
with torch.inference_mode():
    for family,rows in families.items():
        chosen=np.random.default_rng(42).choice(len(rows),32,replace=False)
        family_results=[]
        for begin in range(0,32,4):
            batch=[rows[int(i)] for i in chosen[begin:begin+4]]
            originals=[Image.open(p).convert('RGB') for p,_ in batch]
            ims=[np.asarray(im.resize((round(im.width*96/im.height),96),Image.Resampling.BILINEAR),dtype=np.uint8) for im in originals]
            cols=np.asarray([(im.shape[1]+TOK_W-1)//TOK_W for im in ims],dtype=np.int64)
            wp=((max(int(cols.max())*TOK_W,max(im.shape[1] for im in ims))+MARGIN+TOK_W-1)//TOK_W)*TOK_W
            pixels=np.stack([np.pad(im,((0,0),(0,wp-im.shape[1]),(0,0)),mode='edge') for im in ims]).transpose(0,3,1,2).copy()
            logits=model(torch.from_numpy(pixels).float()/255,torch.from_numpy(cols))['length']
            counts=(logits.argmax(-1)+2).tolist()
            for (path,actual),im,count in zip(batch,originals,counts):
                row=dict(family=family,file=str(path),actual_length=actual,predicted_length=count,correct=count==actual)
                enlarged=im.resize((im.width*2,im.height*2),Image.Resampling.NEAREST)
                canvas=Image.new('RGB',(max(600,enlarged.width),enlarged.height+55),'white');canvas.paste(enlarged,(0,0))
                draw=ImageDraw.Draw(canvas);draw.text((8,enlarged.height+8),f'{family} | Actual length: {actual} | Predicted length: {count}',fill='black')
                draw.text((8,enlarged.height+30),f'{checkpoint_dir.name} EMA; independent length head; seed 42',fill='black')
                preview=out/f'{family}_{len(family_results):02d}.png';canvas.save(preview);row['preview']=str(preview)
                family_results.append(row)
            print(f'{family}: {len(family_results)}/32',flush=True)
        summaries[family]=dict(available=len(rows),tested=32,correct=sum(r['correct'] for r in family_results),accuracy=sum(r['correct'] for r in family_results)/32,predicted_counts=dict(sorted(Counter(r['predicted_length'] for r in family_results).items())),actual_counts=dict(sorted(Counter(r['actual_length'] for r in family_results).items())))
        results.extend(family_results)
report=dict(epoch=int(checkpoint_dir.name.split('_')[1]),checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),device='cpu',threads=1,seed=42,seconds=time.perf_counter()-start,summaries=summaries,note='Length head argmax+2 only; no CTC decoding or family-specific constraints. Real answer lengths used for scoring only. Random small samples; City duplicate-suffix filename excluded.',results=results)
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
