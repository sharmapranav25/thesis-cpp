"""Random, reproducible CPU-only reading check of a saved EMA checkpoint."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from feature_bank.iter2 import data as D, metrics as M
from feature_bank.iter2.model import Iter2Model

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    root=Path(args.run)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    checkpoint_dir=sorted(p for p in (root/'baseline_training').glob('epoch_*')
                          if (p/'scores.json').exists() and (p/'ema.pt').exists())[-1]
    checkpoint=checkpoint_dir/'ema.pt'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config=yaml.safe_load((root/'training_config.yaml').read_text())
    model=Iter2Model(**config['model']).cpu().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.requires_grad_(False)
    dataset=D.open_split(root,'seen_test')[0]
    selected=np.random.default_rng(42).choice(len(dataset),32,replace=False).tolist()
    results=[];start=time.perf_counter()
    with torch.inference_mode():
        for begin in range(0,len(selected),4):
            indices=selected[begin:begin+4]
            batch=D.collate([(dataset,i) for i in indices])
            prediction=model(torch.from_numpy(batch['image']).float()/255,torch.from_numpy(batch['cols']))
            readings=M.decode(prediction['ctc'].numpy(),batch['cols']*model.ctc_frames)
            for i,reading in zip(indices,readings):
                length=int(dataset.ix['length'][i])
                answer=''.join(M.CHARS[int(c)] for c in dataset.ix['chars'][i][:length])
                row=dict(index=i,answer=answer,prediction=reading,correct=answer==reading,
                         character_errors=M.levenshtein(answer,reading))
                # Save all selected images individually; show a fixed prefix.
                im=Image.fromarray(dataset.image(i))
                enlarged=im.resize((im.width*3,im.height*3),Image.Resampling.NEAREST)
                canvas=Image.new('RGB',(enlarged.width,enlarged.height+65),'white')
                canvas.paste(enlarged,(0,0));draw=ImageDraw.Draw(canvas)
                draw.text((8,enlarged.height+8),f"Actual: {answer}   CTC: {reading}   {'CORRECT' if row['correct'] else 'WRONG'}",fill='black')
                draw.text((8,enlarged.height+32),f"seen_test image {i}; {checkpoint_dir.name} EMA; CPU FP32; random seed 42",fill='black')
                name=f'{len(results):02d}_seen_test_{i:06d}.png'
                canvas.save(out/name)
                row['preview']=str(out/name)
                results.append(row)
            print(f'Tested {len(results)}/32',flush=True)
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                epoch=int(checkpoint_dir.name.split('_')[1]),split='seen_test',sample_seed=42,n=len(results),
                correct=sum(r['correct'] for r in results),exact=sum(r['correct'] for r in results)/len(results),
                cer=sum(r['character_errors'] for r in results)/sum(len(r['answer']) for r in results),
                seconds=time.perf_counter()-start,device='cpu',threads=1,
                note='Exploratory sample requested by user; not used for model choice or tuning.',results=results)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
