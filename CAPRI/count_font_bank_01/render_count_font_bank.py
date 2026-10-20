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
from train_character_count import CountCNN,encode

def jsonl(p):return [json.loads(x) for x in p.read_text().splitlines()]
out=new_run(Path(__file__).resolve().parent/'count_font_bank_01');torch.set_num_threads(4)
run=ROOT/'runs/style_phase3_font_05';fonts=read(run/'fonts.json');checkpoint=run/'font_matcher.pt';before=digest(checkpoint)
model=FontMatcher(len(fonts)).cuda();model.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict']);model.eval()
count_run=ROOT/'runs/style_character_count_01'
count_checkpoint=count_run/'count_model.pt';count_hash=digest(count_checkpoint)
count_model=CountCNN().cuda().eval();count_model.load_state_dict(torch.load(count_checkpoint,weights_only=True)['state_dict'])
temperature=read(count_run/'test_results.json')['temperature']
count_references=read(count_run/'real_inference.json');count_records={}
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
    count_rows=count_references[name]['predictions'];count_images=[load(r['path']) for r in count_rows]
    with torch.no_grad(): count_probs=(count_model(torch.tensor(np.stack([encode(im) for im in count_images]),device='cuda'))/temperature).softmax(1).cpu().numpy()
    family_prob=count_probs.mean(0);cdf=family_prob.cumsum();cdf[-1]=1
    lengths=[int(np.searchsorted(cdf,q)+2) for q in (.125,.375,.625,.875)]
    count_records[name]=dict(probabilities=family_prob.tolist(),lengths=lengths,mode=int(family_prob.argmax()+2),references=[dict(path=r['path'],pixel_sha256=pixel_hash(im),probabilities=prob.tolist()) for r,im,prob in zip(count_rows,count_images,count_probs)])
    p,h,_=infer(model,images);chosen=int(p.mean(0).argmax());effect='hollow' if h[:,1].mean()>=.7 else 'filled'
    panel=Image.new('RGB',(1000,430),'#f1f3f5');draw=ImageDraw.Draw(panel);records=[]
    for i,(im,r) in enumerate(zip(images[:4],rows[:4])):
        style=extract_style(im);style['classifier_features']={f:q['image_state'] for f,q in r['predictions'].items()}
        text=''.join(np.random.default_rng(180000+i).choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),lengths[i]));seed=181000+i;variant=variants.get(name,0)
        base,_,br=render(style,text,fonts[0]['path'],seed,variant)
        adjusted=copy.deepcopy(style);adjusted['font_effect']=effect
        final,mask,recipe=render(adjusted,text,fonts[chosen]['path'],seed,variant)
        assert br['operations']==recipe['operations']
        final.save(out/f'{name}_{i:02d}.png');mask.save(out/f'{name}_{i:02d}_mask.png');base.save(out/f'{name}_{i:02d}_phase1.png')
        for rownum,img in enumerate([im,base,final]):
            img=img.copy();img.thumbnail((235,100));panel.paste(img,(10+i*250,30+rownum*140))
        assert len(text)==lengths[i]
        draw.text((10+i*250,410),f'Generated length {len(text)} | estimated',fill='black')
        records.append(dict(text=text,requested_length=len(text),recipe=recipe))
    for y,label in [(5,'Reference style samples (development)'),(145,'Phase 1'),(285,'Phase 1 + font adjustment (new Phase 2)')]:draw.text((10,y),name+' | '+label,fill='black')
    panel.save(out/f'{name}_comparison.png');summary_panels.append(panel)
    results[name]=dict(font=fonts[chosen]['family'],effect=effect,font_probabilities=p.mean(0).tolist(),hollow_probability=float(h[:,1].mean()),references=len(rows),renders=4,background_operations_preserved=True,recipes=records)
assert digest(checkpoint)==before
assert digest(count_checkpoint)==count_hash
write(out/'count_inference.json',count_records)
for k in range(2):
    board=Image.new('RGB',(1000,1720),'white')
    for i,panel in enumerate(summary_panels[k*4:k*4+4]):board.paste(panel,(0,430*i))
    board.save(out/f'bank_{k+1}.png')
write(out/'results.json',results);write(out/'protocol.json',dict(sequence='Phase1 CV + frozen features + fresh frozen count inference, then Phase2 font inference; no optional refinement',feature_predictions='reused hash-checked pixel-matched frozen feature inference',cv='fresh per rendered reference',font_model_sha256=before,real_training=False,scope='development preview, not held-out validation',count_model_sha256=count_hash,count_temperature=temperature,count_policy="family mean posterior; four fixed stratified quantiles .125/.375/.625/.875, not a calibrated real error margin or verified family length distribution",count_support=list(range(2,9)),optional_phase3=False,source_families=8))
print({k:(r['font'],r['effect']) for k,r in results.items()})
