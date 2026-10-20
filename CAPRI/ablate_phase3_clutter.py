"""Development-only causal ablation; does not replace user-selected controls."""
import sys,copy,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');sys.path.insert(0,str(ROOT))
from feature_bank.core import read,write,new_run,load,digest
from style_pipeline.measure import extract_style
from style_pipeline.recipes import apply_recipe
from render_color_locked import render

out=new_run(Path(__file__).resolve().parent/'phase3_clutter_ablation_01')
prior=read(ROOT/'runs/style_direct_bank_01/results.json')['sloth'];full=read(ROOT/'runs/style_phase3_color_locked_01/results.json')['sloth']['parameters']
rows=[r for r in read(ROOT/'runs/style_phase3_new_family_02/predictions.json') if r['partition']=='train'][:4]
font=next(f['path'] for f in read(ROOT/'runs/style_phase3_font_05/fonts.json') if f['family']==prior['font'])
variants=[('full',None),('line_count_gain_off',6),('line_width_gain_off',7),('dot_gain_off',8),('curve_gain_off',9)]
panel=Image.new('RGB',(1000,750),'white');draw=ImageDraw.Draw(panel);results=[]
for rownum,(name,index) in enumerate(variants):
    draw.text((10,rownum*150+5),name+' | diagnostic only, colors locked',fill='black')
    for i,r in enumerate(rows):
        style=extract_style(load(r['path']));style['classifier_features']={f:q['image_state'] for f,q in r['predictions'].items()}
        p=np.array(full);p[:3]=0
        if index is not None:p[index]=0
        s=apply_recipe(style,p);s['font_effect']=prior['effect'];before=prior['recipes'][i]
        for k in ('background_palette','foreground_palette','gradient_left','gradient_right'):s[k]=copy.deepcopy(style[k])
        s['_locked_glyph_colors']=[g['color'] for g in before['glyphs']]
        s['_locked_operation_colors']={k:[o['color'] for o in before['operations'] if o['kind']==k] for k in ('block','point','line','classifier_curve')}
        im,mask,recipe=render(s,before['text'],font,before['seed'],before['variant'])
        assert [g['color'] for g in recipe['glyphs']]==s['_locked_glyph_colors']
        for op in recipe['operations']:assert op['color'] in s['_locked_operation_colors'][op['kind']]
        if name=='full':assert np.array_equal(np.asarray(im),np.asarray(load(ROOT/'runs/style_phase3_color_locked_01'/f'sloth_{i:02d}.png')))
        im.save(out/f'{name}_{i:02d}.png');im.thumbnail((235,110));panel.paste(im,(10+i*250,rownum*150+30))
        lineops=[o for o in recipe['operations'] if o['kind']=='line']
        results.append(dict(variant=name,index=i,line_count=len(lineops),line_total_width=sum(o['width'] for o in lineops),dot_count=sum(o['kind']=='point' for o in recipe['operations']),curve_count=sum(o['kind']=='classifier_curve' for o in recipe['operations'])))
panel.save(out/'comparison.png');write(out/'results.json',dict(scope='exposed Sloth development, one control disabled at a time; no promotion or retraining',full_parameters=full,results=results))
print({name:{k:sum(r[k] for r in results if r['variant']==name) for k in ('line_count','line_total_width','dot_count','curve_count')} for name,_ in variants})
