"""All eight current sources: Phase1 directly to frozen font inference."""
import sys,json,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');sys.path.insert(0,str(ROOT))
from feature_bank.core import read,write,new_run,load,pixel_hash,digest
from style_pipeline.font_match import FontMatcher,infer
from style_pipeline.measure import extract_style
from style_pipeline.render import render

def jsonl(p):return [json.loads(x) for x in p.read_text().splitlines()]
out=new_run(Path(__file__).resolve().parent/'direct_bank_renders_01');torch.set_num_threads(4)
run=ROOT/'runs/style_phase3_font_05';fonts=read(run/'fonts.json');checkpoint=run/'font_matcher.pt';before=digest(checkpoint)
model=FontMatcher(len(fonts)).cuda();model.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict']);model.eval()
families={};variants={};old=read(ROOT/'runs/style_phase1_05/results.json')['families']
for name in old:
    folder=ROOT/'runs'/('style_frozen_'+name+('_02' if name=='darkbay' else '_01'))
    protocol=read(folder/'protocol.json')
    for f,p in protocol['models'].items():assert digest(p)==protocol['model_sha256_before'][f]
    lookup={r['pixel_sha256']:r for r in jsonl(folder/'predictions.jsonl')}
    rows=read(ROOT/'runs/style_phase1_05'/f'{name}_measurements.json')['fit'][:16]
    families[name]=[lookup[r['pixel_sha256']] for r in rows];variants[name]=old[name]['selected']['variant']
lookup={r['pixel_sha256']:r for r in jsonl(ROOT/'runs/style_frozen_inference_01/predictions.jsonl')}
families['roncuevas']=[lookup[r['pixel_sha256']] for r in read(ROOT/'runs/style_phase1_unseen_01/split.json')['fit'][:16]]
for name,folder in [('pymlex','style_phase3_new_family_01'),('sloth','style_phase3_new_family_02')]:families[name]=[r for r in read(ROOT/'runs'/folder/'predictions.json') if r['partition']=='train'][:16]
results={};summary_panels=[]
for name,rows in families.items():
    images=[load(r['path']) for r in rows]
    for im,r in zip(images,rows):assert pixel_hash(im)==r['pixel_sha256']
    p,h,_=infer(model,images);chosen=int(p.mean(0).argmax());effect='hollow' if h[:,1].mean()>=.7 else 'filled'
    panel=Image.new('RGB',(1000,430),'#f1f3f5');draw=ImageDraw.Draw(panel);records=[]
    for i,(im,r) in enumerate(zip(images[:4],rows[:4])):
        style=extract_style(im);style['classifier_features']={f:q['image_state'] for f,q in r['predictions'].items()}
        text=''.join(np.random.default_rng(180000+i).choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),5));seed=181000+i;variant=variants.get(name,0)
        base,_,br=render(style,text,fonts[0]['path'],seed,variant)
        adjusted=copy.deepcopy(style);adjusted['font_effect']=effect
        final,mask,recipe=render(adjusted,text,fonts[chosen]['path'],seed,variant)
        assert br['operations']==recipe['operations']
        final.save(out/f'{name}_{i:02d}.png');mask.save(out/f'{name}_{i:02d}_mask.png');base.save(out/f'{name}_{i:02d}_phase1.png')
        for rownum,img in enumerate([im,base,final]):
            img=img.copy();img.thumbnail((235,100));panel.paste(img,(10+i*250,30+rownum*140))
        records.append(recipe)
    for y,label in [(5,'Reference style samples (development)'),(145,'Phase 1'),(285,'Phase 1 + font adjustment (new Phase 2)')]:draw.text((10,y),name+' | '+label,fill='black')
    panel.save(out/f'{name}_comparison.png');summary_panels.append(panel)
    results[name]=dict(font=fonts[chosen]['family'],effect=effect,font_probabilities=p.mean(0).tolist(),hollow_probability=float(h[:,1].mean()),references=len(rows),renders=4,background_operations_preserved=True,recipes=records)
assert digest(checkpoint)==before
write(out/'results.json',results);write(out/'protocol.json',dict(sequence='Phase1 then frozen font inference; no former Phase2 recipe/refiner or typography gains',feature_predictions='reused hash-checked pixel-matched frozen feature inference',cv='fresh per rendered reference',font_model_sha256=before,real_training=False,scope='development preview, not held-out validation',generic_character_count=5,source_families=8))
print({k:(r['font'],r['effect']) for k,r in results.items()})
