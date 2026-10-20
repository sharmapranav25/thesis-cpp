"""Matched background rollback experiment; reads historical project only."""
import sys,json,copy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');sys.path.insert(0,str(ROOT))
from feature_bank.core import read,write,new_run,load
from style_pipeline.render import render
from style_pipeline.recipes import apply_recipe

out=new_run(Path(__file__).resolve().parent/'background_regression_01')
fonts=read(ROOT/'runs/style_phase3_font_05/fonts.json');controls=read(ROOT/'runs/style_phase2_pilot_01/selected_recipes.json');old=read(ROOT/'runs/style_phase1_05/results.json')['families'];summary={}
for name in controls:
    measured=read(ROOT/'runs/style_phase1_05'/f'{name}_measurements.json')['fit'][96:128]
    folder=ROOT/'runs'/('style_frozen_'+name+('_02' if name=='darkbay' else '_01'))
    preds={r['pixel_sha256']:r for r in (json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines())}
    selected=read(ROOT/'runs/style_phase3_font_05'/name/'selection.json')['selected'];previous=old[name]['selected'];panel=Image.new('RGB',(1000,620),'white');draw=ImageDraw.Draw(panel);records=[]
    p=np.array(controls[name]);q=p.copy();q[:3]=0;q[6:]=0
    for i,row in enumerate(measured):
        style=copy.deepcopy(row['style']);style['classifier_features']={f:r['image_state'] for f,r in preds[row['pixel_sha256']]['predictions'].items()}
        seed=63000+i;text=''.join(np.random.default_rng(seed+1000).choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),5))
        images={};recipes={}
        for method,params in [('current',p),('background_rollback',q)]:
            s=apply_recipe(style,params)
            if not selected['baseline']:s.update(font_effect=selected['effect'],glyph_width_gain=selected['width'],text_height=float(np.clip(s['text_height']*selected['height'],.25,.75)))
            image,mask,recipe=render(s,text,fonts[selected['font']]['path'],seed,previous['variant']);images[method]=image;recipes[method]=recipe
            image.save(out/f'{name}_{i:03d}_{method}.png')
        baseline,_,base_recipe=render(apply_recipe(style,p),text,fonts[previous['font']]['path'],seed,previous['variant'])
        # Background is drawn before glyph processing, so this is an exact
        # operation-level test of whether font adjustment changed it.
        font_keeps_background=base_recipe['operations']==recipes['current']['operations']
        saved=load(ROOT/'runs/style_phase3_font_05'/name/f'{i:03d}.png')
        replay_exact=np.array_equal(np.asarray(saved),np.asarray(images['current']))
        counts=lambda recipe:{kind:sum(op['kind']==kind for op in recipe['operations']) for kind in ('line','point','classifier_curve','block')}
        records.append(dict(index=i,font_preserves_background_operations=font_keeps_background,current_replays_saved_exactly=replay_exact,current=counts(recipes['current']),rollback=counts(recipes['background_rollback'])))
        if i<4:
            for rownum,im in enumerate([load(row['path']),baseline,images['current'],images['background_rollback']]):
                im=im.copy();im.thumbnail((235,110));panel.paste(im,(10+i*250,30+rownum*150))
    for y,label in [(5,'Reference development images'),(155,'Phase 2 recipe + older font'),(305,'Same Phase 2 background + latest font'),(455,'Background controls rolled back + latest font')]:draw.text((10,y),name+' | '+label,fill='black')
    panel.save(out/f'{name}_comparison.png');summary[name]=dict(background_rgb_offset=(p[:3]*45).tolist(),line_count_multiplier=float(np.exp(p[6]*1.6)),line_width_multiplier=float(np.exp(p[7]*1.4)),dot_multiplier=float(np.exp(p[8]*1.6)),records=records)
write(out/'audit.json',summary)
print(json.dumps({k:{'all_font_background_equal':all(r['font_preserves_background_operations'] for r in v['records']),'all_replay_exact':all(r['current_replays_saved_exactly'] for r in v['records']),'line_multiplier':v['line_count_multiplier'],'width_multiplier':v['line_width_multiplier'],'dots':v['dot_multiplier']} for k,v in summary.items()},indent=2))
