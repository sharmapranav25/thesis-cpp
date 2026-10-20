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
from render_color_locked import render
from style_pipeline.recipes import apply_recipe,RecipeCNN

def jsonl(p):return [json.loads(x) for x in p.read_text().splitlines()]
out=new_run(Path(__file__).resolve().parent/'phase3_color_locked_02');torch.set_num_threads(4)
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
prior=read(ROOT/'runs/style_direct_bank_01/results.json');controls=read(ROOT/'runs/style_phase2_pilot_01/selected_recipes.json');recipe_model=RecipeCNN().cuda();recipe_model.load_state_dict(torch.load(ROOT/'runs/style_phase2_pilot_01/recipe_cnn.pt',weights_only=True)['state_dict']);recipe_model.eval()
results={};summary_panels=[];failures=[]
for name,rows in families.items():
    images=[load(r['path']) for r in rows]
    for im,r in zip(images,rows):assert pixel_hash(im)==r['pixel_sha256']
    chosen=next(i for i,f in enumerate(fonts) if f['family']==prior[name]['font']);effect=prior[name]['effect']
    if name in controls:parameters=np.array(controls[name],float)
    else:
        tx=torch.stack([torch.from_numpy(np.array(im.resize((192,64)),np.float32).transpose(2,0,1)/255) for im in images]).cuda()
        with torch.no_grad():parameters=recipe_model(tx).mean(0).cpu().numpy()
    parameters[:3]=0
    panel=Image.new('RGB',(1000,430),'#f1f3f5');draw=ImageDraw.Draw(panel);records=[]
    for i,(im,r) in enumerate(zip(images[:4],rows[:4])):
        style=extract_style(im);style['classifier_features']={f:q['image_state'] for f,q in r['predictions'].items()}
        text=''.join(np.random.default_rng(180000+i).choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),5));seed=181000+i;variant=variants.get(name,0)
        br=prior[name]['recipes'][i];base=load(ROOT/'runs/style_direct_bank_01'/f'{name}_{i:02d}.png')
        adjusted=apply_recipe(style,parameters);adjusted['font_effect']=effect
        for key in ('background_palette','foreground_palette','gradient_left','gradient_right'):
            adjusted[key]=copy.deepcopy(style[key])
            assert adjusted[key]==style[key]
        adjusted['_locked_operation_colors']={kind:[op['color'] for op in br['operations'] if op['kind']==kind] for kind in ('block','line','point','classifier_curve')}
        adjusted['_locked_glyph_colors']=[g['color'] for g in br['glyphs']]
        try:
            final,mask,recipe=render(adjusted,text,fonts[chosen]['path'],seed,variant)
            assert [g['color'] for g in recipe['glyphs']]==adjusted['_locked_glyph_colors']
            for op in recipe['operations']:
                assert op['color'] in adjusted['_locked_operation_colors'][op['kind']]
        except ValueError as exc:
            failures.append(dict(family=name,index=i,reason=str(exc)));final=base;mask=load(ROOT/'runs/style_direct_bank_01'/f'{name}_{i:02d}_mask.png');recipe=br
        final.save(out/f'{name}_{i:02d}.png');mask.save(out/f'{name}_{i:02d}_mask.png');base.save(out/f'{name}_{i:02d}_phase1.png')
        for rownum,img in enumerate([im,base,final]):
            img=img.copy();img.thumbnail((235,100));panel.paste(img,(10+i*250,30+rownum*140))
        records.append(recipe)
    for y,label in [(5,'Reference style samples (development)'),(145,'Phase 1 + fonts (before)'),(285,'Phase 3 with colors locked')]:draw.text((10,y),name+' | '+label,fill='black')
    panel.save(out/f'{name}_comparison.png');summary_panels.append(panel)
    results[name]=dict(font=fonts[chosen]['family'],effect=effect,parameters=parameters.tolist(),color_locked=True,references=len(rows),renders=4,background_geometry_may_change=True,recipes=records)
assert digest(checkpoint)==before
write(out/'failures.json',failures);write(out/'results.json',results);write(out/'protocol.json',dict(sequence='Phase1 then existing font selection then color-locked recipe refinement; no pixel refiner',feature_predictions='reused hash-checked pixel-matched frozen feature inference',cv='fresh per rendered reference',font_model_sha256=before,real_training=False,scope='development preview, not held-out validation',generic_character_count=5,source_families=8))
print({k:(r['font'],r['effect']) for k,r in results.items()})
