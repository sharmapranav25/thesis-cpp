"""Fresh CPU inference of the frozen simple hybrid on all three target families.

Only source identities are reused. No real answers, old predictions, or specialists
enter inference, training, calibration, or checkpoint selection.
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import yaml
from PIL import Image

PROJECT = Path('E:/pranav/Final sem/sourcefree_synth')
WORK = Path('C:/Users/pranav/Documents/ChatGPT/thesis')
OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT))
from feature_bank.iter2.model import Iter2Model
from feature_bank.iter2.generate import effect_heads, TEXT_HEADS, CATEGORIES, STROKES
from feature_bank.train.data import MARGIN

BASE = PROJECT/'runs/iter2_baseline_15_20261003_04'
CHECKPOINT = BASE/'baseline_training/epoch_15/ema.pt'
EXPECTED = 'ae73375ead99071d55c6394f7c4eb7c4d27c0a3285f8f15d8d84c9df43bc6b32'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def event(**data):
    data['elapsed_seconds']=round(time.perf_counter()-START,1)
    (OUT/'inference_status.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert digest(CHECKPOINT)==EXPECTED
    model=Iter2Model(**yaml.safe_load((BASE/'training_config.yaml').read_text())['model']).cpu().eval()
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    model.requires_grad_(False)
    old=WORK/'iter2_full_target_recreation_01'
    rows=[{k:r[k] for k in ['family','index','source']} for r in
          map(json.loads,(old/'predictions.jsonl').read_text().splitlines())]
    manifest=json.loads((old/'manifest.json').read_text())
    assert len(rows)==3599 and dict(Counter(r['family'] for r in rows))==manifest['counts']
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for r in rows:
        with Image.open(r['source']) as im:
            r['size']=list(im.size)
        r['width96']=min(480,round(r['size'][0]*96/r['size'][1]))
    ordered=sorted(rows,key=lambda r:r['width96'])
    effects=effect_heads()
    completed=[]
    event(stage='inference_started',total=len(rows),device='cpu',threads=4,
          checkpoint=str(CHECKPOINT),checkpoint_sha256=EXPECTED)
    for offset in range(0,len(ordered),16):
        chunk=ordered[offset:offset+16]
        widths=[r['width96'] for r in chunk]
        cols=np.array([(w+7)//8 for w in widths],np.int64)
        wp=((max(cols)*8+MARGIN+7)//8)*8
        batch=[]
        for r,w in zip(chunk,widths):
            with Image.open(r['source']) as im:
                a=np.asarray(im.convert('RGB').resize((w,96),Image.Resampling.BILINEAR))
            batch.append(np.pad(a,((0,0),(0,wp-w),(0,0)),mode='edge').transpose(2,0,1))
        with torch.inference_mode():
            p=model(torch.from_numpy(np.stack(batch)).float()/255,torch.from_numpy(cols))
        for j,r in enumerate(chunk):
            result={k:v for k,v in r.items() if k!='width96'}
            result['length']=int(p['length'][j].argmax())+2
            result['length_probabilities']=p['length'][j].softmax(-1).tolist()
            for key,block,names in [('eff','effects',effects),('txt','text_features',TEXT_HEADS)]:
                prob=p[key+'_presence'][j].sigmoid().numpy()
                ordinal=(p[key+'_ordinal'][j]>0).sum(-1).numpy()
                result[block]={name:dict(probability=float(prob[i]),present=bool(prob[i]>=.5),level=int(ordinal[i]))
                               for i,name in enumerate(names)}
            result['font']=dict(category=CATEGORIES[int(p['font_cat'][j].argmax())],
                stroke=STROKES[int(p['font_stroke'][j].argmax())],italic=bool(p['font_italic'][j]>0))
            result['font'].update({name:int((p['font_ordinal'][j,i]>0).sum())
                for i,name in enumerate(['weight','width','x_height','slant'])})
            result['font_category_probabilities']=dict(zip(CATEGORIES,p['font_cat'][j].softmax(-1).tolist()))
            completed.append(result)
        if offset%256==0:
            event(stage='inference',completed=len(completed),total=len(rows))
    completed.sort(key=lambda r:(r['family'],r['index']))
    with open(OUT/'predictions.jsonl','w',encoding='utf-8') as f:
        for r in completed:
            f.write(json.dumps(r)+'\n')
    summary={}
    for family in manifest['counts']:
        selected=[r for r in completed if r['family']==family]
        summary[family]=dict(images=len(selected),lengths=dict(Counter(r['length'] for r in selected)),
            font_categories=dict(Counter(r['font']['category'] for r in selected)),
            effects={h:round(sum(r['effects'][h]['present'] for r in selected)/len(selected),4) for h in effects},
            text_features={h:round(sum(r['text_features'][h]['present'] for r in selected)/len(selected),4) for h in TEXT_HEADS})
    assert digest(CHECKPOINT)==EXPECTED
    report=dict(state='complete',total=len(completed),counts=manifest['counts'],summary=summary,
        checkpoint=str(CHECKPOINT),checkpoint_sha256=EXPECTED,device='cpu',presence_threshold=.5,
        seconds=round(time.perf_counter()-START,1),real_answers_used=False,baseline_unchanged=True)
    (OUT/'inference_report.json').write_text(json.dumps(report,indent=2))
    event(stage='inference_complete',total=len(completed),counts=manifest['counts'])


if __name__=='__main__':
    START=time.perf_counter()
    main()
