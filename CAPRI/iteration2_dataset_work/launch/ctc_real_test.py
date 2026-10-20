"""CPU-only, reproducible scoring on real held-out CAPTCHA images."""
import csv, hashlib, json, re, time
from pathlib import Path
import numpy as np
import torch, yaml
from PIL import Image, ImageDraw
from feature_bank.iter2 import metrics as M
from feature_bank.iter2.model import Iter2Model
from feature_bank.train.data import TOK_W, MARGIN

root=Path('E:/pranav/Final sem/sourcefree_synth/runs/iter2_baseline_15_20261003_04')
out=Path('C:/Users/pranav/Documents/ChatGPT/thesis/ctc_real_test_01')
out.mkdir(exist_ok=False)
checkpoint_dir=sorted(p for p in (root/'baseline_training').glob('epoch_*') if (p/'scores.json').exists() and (p/'ema.pt').exists())[-1]
checkpoint=checkpoint_dir/'ema.pt'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
config=yaml.safe_load((root/'training_config.yaml').read_text())
model=Iter2Model(**config['model']).cpu().eval()
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
model.requires_grad_(False)
crypt=Path('E:/pranav/model 1/data/eval')
with (crypt/'labels.csv').open(newline='',encoding='utf-8-sig') as f:
    crypt_rows=[(crypt/r['filename'],r['label']) for r in csv.DictReader(f)]
dark=Path('E:/pranav/cryptbb_sourcefree/data/darkbay/eval')
with (dark/'labels.csv').open(newline='',encoding='utf-8-sig') as f:
    dark_rows=[(dark/r['filename'],r['label']) for r in csv.DictReader(f)]
families={'CryptBB':crypt_rows,'DarkBay':dark_rows}
results=[];summaries={};start=time.perf_counter()
with torch.inference_mode():
    for family,rows in families.items():
        chosen=np.random.default_rng(42).choice(len(rows),32,replace=False)
        family_results=[]
        for begin in range(0,32,4):
            batch_rows=[rows[int(i)] for i in chosen[begin:begin+4]]
            originals=[Image.open(p).convert('RGB') for p,_ in batch_rows]
            ims=[np.asarray(im.resize((round(im.width*96/im.height),96),Image.Resampling.BILINEAR),dtype=np.uint8) for im in originals]
            cols=np.asarray([(im.shape[1]+TOK_W-1)//TOK_W for im in ims],dtype=np.int64)
            wp=((max(int(cols.max())*TOK_W,max(im.shape[1] for im in ims))+MARGIN+TOK_W-1)//TOK_W)*TOK_W
            images=np.stack([np.pad(im,((0,0),(0,wp-im.shape[1]),(0,0)),mode='edge') for im in ims]).transpose(0,3,1,2).copy()
            prediction=model(torch.from_numpy(images).float()/255,torch.from_numpy(cols))
            readings=M.decode(prediction['ctc'].numpy(),cols*model.ctc_frames)
            for (path,answer),im,reading in zip(batch_rows,originals,readings):
                row=dict(family=family,file=str(path),answer=answer,prediction=reading,correct=answer==reading,character_errors=M.levenshtein(answer,reading))
                enlarged=im.resize((im.width*3,im.height*3),Image.Resampling.NEAREST)
                canvas=Image.new('RGB',(max(enlarged.width,600),enlarged.height+65),'white');canvas.paste(enlarged,(0,0))
                draw=ImageDraw.Draw(canvas)
                draw.text((8,enlarged.height+8),f'{family} | Actual: {answer} | CTC: {reading} | '+('CORRECT' if row['correct'] else 'WRONG'),fill='black')
                draw.text((8,enlarged.height+32),f'{checkpoint_dir.name} EMA; real held-out image; seed 42; CPU',fill='black')
                preview=out/f'{family}_{len(family_results):02d}.png';canvas.save(preview);row['preview']=str(preview)
                family_results.append(row)
            print(f'{family}: tested {len(family_results)}/32',flush=True)
        summaries[family]=dict(available=len(rows),tested=len(family_results),correct=sum(r['correct'] for r in family_results),exact=sum(r['correct'] for r in family_results)/len(family_results),cer=sum(r['character_errors'] for r in family_results)/sum(len(r['answer']) for r in family_results))
        results.extend(family_results)
report=dict(epoch=int(checkpoint_dir.name.split('_')[1]),checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),device='cpu',threads=1,seed=42,seconds=time.perf_counter()-start,summaries=summaries,note='Real labels used for scoring only. No training, tuning or checkpoint selection based on these results. Plain greedy CTC decoding without family constraints.',results=results)
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
