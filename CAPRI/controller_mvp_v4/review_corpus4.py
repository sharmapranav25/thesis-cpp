"""Audit the completed matched corpus and show deterministic, unselected examples."""
from setup4 import ROOT,U,FAMILIES,ARMS
from corpus4 import compatible_font
from collections import Counter
import copy,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
    data=ROOT/'corpus';assert U.read(data/'audit.json')['passed']
    rows=U.read(data/'train_labels.json');out=ROOT/'corpus_review';out.mkdir(exist_ok=True)
    arrays={a:np.load(data/f'train_{a}.npy',mmap_mode='r') for a in ARMS}
    summary={}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for family in FAMILIES:
        selected=[r for r in rows if r['family']==family and not r['anchor']]
        configs=U.read(ROOT/'collections'/family/'archive.json');effective=[]
        for original in configs:
            p=copy.deepcopy(original);p['length']=None
            if p['font']:p['font']=compatible_font(p['font'])
            if p['font_pool']:p['font_pool']=list(dict.fromkeys(compatible_font(f) for f in p['font_pool']))
            effective.append(json.dumps(p,sort_keys=True))
        summary[family]=dict(examples=len(selected),effective_configurations=len(set(effective)),stroke_repair_counts=dict(Counter(r['stroke_repair'] for r in selected)),
            retry_counts=dict(Counter(r['attempts'] for r in selected)),identical_single_diverse=sum(r['arms']['single']['pixel_sha256']==r['arms']['diverse']['pixel_sha256'] for r in selected))
        canvas=Image.new('RGB',(1000,80+8*110),'#eef1f5');draw=ImageDraw.Draw(canvas)
        draw.text((16,10),family+' — first 8 matched training examples',fill='#172033',font=font)
        for j,arm in enumerate(ARMS):draw.text((90+j*300,46),arm,fill='#172033',font=font)
        for i,row in enumerate(selected[:8]):
            y=80+i*110;draw.text((12,y+4),row['label'],fill='#172033',font=small)
            for j,arm in enumerate(ARMS):
                im=Image.fromarray(arrays[arm][row['index']]).resize((288,72))
                canvas.paste(im,(90+j*300,y+26))
        canvas.save(out/(family+'.png'))
    U.dump(out/'audit.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
