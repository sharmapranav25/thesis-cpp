"""Manual corrections to saved baseline recipes, with before/after provenance."""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent
WORK=OUT.parent
sys.path.insert(0,str(WORK/'manual_family_renderer_20261004'))
import render_examples as A
from captcha_effects import render as R,text as T
from captcha_effects.captcha import legibility


def cryptbb(start):
    rng=np.random.default_rng(start['seed'])
    cv=R.new_canvas(*start['description']['size'],dict(background=[255,255,255]),rng)
    under=dict(effects=[dict(name='blocks',level=4,style=dict(palette='pastel',layout='tiled',opacity=.48,speckle=True)),
                       dict(name='dots',level=7,style=dict(palette='pastel'))])
    R.draw_under(cv,under,rng)
    colors=[(226,172,195),(153,209,206),(186,202,237),(193,218,164),(223,197,145)]
    for x in range(0,cv.w,7):
        A.line(cv,[(x,0),(x,cv.h-1)],colors[int(rng.integers(len(colors)))],.65,'bg_grid')
    for y in range(0,cv.h,7):
        A.line(cv,[(0,y),(cv.w-1,y)],colors[int(rng.integers(len(colors)))],.65,'bg_grid')
    for k in range(17):
        xs=np.linspace(-5,cv.w+5,120)
        phase=rng.uniform(0,2*math.pi)
        ys=rng.uniform(8,cv.h-8)+rng.uniform(7,17)*np.sin(xs/rng.uniform(9,25)+phase)
        A.line(cv,list(zip(xs,ys)),colors[k%len(colors)],.65,'curves')
    spec=copy.deepcopy(start['text_spec'])
    spec.update(text='G7kQm',charset='mixed',fonts=[A.font(f) for f in ['Roboto Slab','Roboto','Merriweather']],
        size=.75,size_jitter=.20,weight=.90,bold_stroke=1,x_scale=1.,rotation=26.,jitter=.17,
        spacing=.22,multicolour=True,palette='manual_glyph_palette',min_contrast=45.)
    palette=[(89,104,34),(48,83,143),(148,44,92),(20,123,101),(102,56,138)]
    txt=A.draw_glyphs(cv,spec,rng,palette)
    A.rectangle(cv,(0,0,cv.w-1,cv.h-1),(35,35,35),head='border_frame',fill=False,width=1)
    controls=dict(background=[255,255,255],grid_spacing=7,grid_palette=colors,curves=17,
                  under=under,glyph_palette=palette)
    edits=['Correct length 2 -> 5 and digits -> mixed content',
           'Replace thin italic Display font with heavy mixed serif/sans fonts',
           'Remove hollow glyph treatment',
           'Set white background and colored glyph palette',
           'Remove circles, blobs and background pattern',
           'Keep grid presence but set fine aligned pastel geometry',
           'Add missing curves, dot clutter and speckled pastel blocks',
           'Add rotation, vertical jitter and size variation']
    return cv,txt,spec,controls,edits


def darkbay(start):
    rng=np.random.default_rng(start['seed'])
    background=[127,114,137]
    cv=R.new_canvas(*start['description']['size'],dict(background=background),rng)
    for _ in range(21):
        x=float(rng.uniform(-10,188));w=float(rng.uniform(13,48))
        y=float(rng.uniform(-8,12));h=float(rng.uniform(31,63))
        delta=int(rng.choice([-37,-25,-16,10,19,29]))
        color=tuple(int(np.clip(c+delta,0,255)) for c in background)
        A.rectangle(cv,(x,y,x+w,y+h),color)
    for _ in range(8):
        x=float(rng.uniform(0,178));w=float(rng.uniform(14,37))
        y=float(rng.uniform(0,8));h=float(rng.uniform(33,49))
        delta=int(rng.choice([-29,24]))
        color=tuple(int(np.clip(c+delta,0,255)) for c in background)
        A.rectangle(cv,(x,y,x+w,y+h),color,fill=False,width=2)
    spec=copy.deepcopy(start['text_spec'])
    # Retain predicted count, non-italic style, zero slant, width and weight controls.
    spec.update(text='TOW2RJ',charset='upper_digits',fonts=[A.font('Roboto')],size=.87,size_jitter=.05,
        rotation=17.,jitter=.08,spacing=.12,multicolour=True,palette='manual_glyph_palette',min_contrast=45.)
    palette=[(250,250,248),(20,19,23),(248,248,246),(19,18,22),(246,246,244),(22,21,25)]
    txt=A.draw_glyphs(cv,spec,rng,palette)
    controls=dict(background=background,tall_rectangles=21,outlined_rectangles=8,glyph_palette=palette)
    edits=['Set muted purple-gray background and alternating black/white glyph colors',
           'Replace pattern, border and inversion effects with tall layered rectangles and frames',
           'Use a proportional sans font; retain predicted non-italic, zero-slant style',
           'Keep six characters but permit a digit alongside uppercase letters',
           'Increase glyph size, reduce excessive spacing, and add rotation/jitter']
    return cv,txt,spec,controls,edits


def city(start):
    rng=np.random.default_rng(start['seed'])
    cv=R.new_canvas(*start['description']['size'],dict(background=[255,255,255]),rng)
    bands=[(10,15,5),(22,18,4),(34,48,5),(45,37,4),(58,68,5),(71,62,4)]
    for y0,y1,width in bands:
        A.line(cv,[(-4,y0),(cv.w+4,y1)],(202,202,202),width,'bands',antialias=False)
    spec=copy.deepcopy(start['text_spec'])
    # Keep model-supplied text/count, font, weight, width, wide spacing and style.
    spec.update(size=.49,rotation=11.,jitter=.13,multicolour=False)
    txt=A.draw_glyphs(cv,spec,rng,[(43,43,43)])
    post=dict(effects=[copy.deepcopy(e) for e in start['effects_spec']['effects'] if e['name']=='pixelate'])
    R.apply_post(cv,post,rng)
    controls=dict(background=[255,255,255],bands=bands,band_color=[202,202,202],glyph_palette=[(43,43,43)],post=post)
    edits=['Set white/gray/black palette instead of random colors',
           'Keep bands but control their shallow slopes, widths and spacing',
           'Adjust glyph size and add the missing rotation/vertical jitter']
    return cv,txt,spec,controls,edits


def changed_fields(before,after):
    return {key:dict(before=before.get(key),after=after.get(key))
            for key in sorted(set(before)|set(after)) if before.get(key)!=after.get(key)}


def comparison(family):
    pictures=[Image.open(OUT/f'{family}_{stage}.png').convert('RGB') for stage in ['target','before','after']]
    labels=['TARGET REFERENCE','SIMPLE HYBRID START','AFTER MANUAL ADJUSTMENT']
    panel=480;pad=14
    scaled=[]
    for im in pictures:
        factor=min(3.,(panel-2*pad)/im.width)
        scaled.append(im.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.NEAREST))
    sheet=Image.new('RGB',(panel*3,max(im.height for im in scaled)+64),(247,247,247))
    d=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for j,(im,label) in enumerate(zip(scaled,labels)):
        d.text((j*panel+pad,12),label,fill=(35,35,35),font=font)
        sheet.paste(im,(j*panel+(panel-im.width)//2,43))
    sheet.save(OUT/f'{family}_comparison.png')


def main():
    report=dict(method='Baseline predictions -> current mapping -> inspected starting render -> manual control edits',
        limitations=['One preselected reference per family for the adjustment trial',
                     'Qualitative effort assessment, not a timed blind experiment',
                     'Target references and manual controls were already inspected in the preceding demo',
                     'No real answers used; inference results do not alter specialist training or checkpoint selection'],families={})
    for family,fn in [('CryptBB',cryptbb),('DarkBay',darkbay),('City',city)]:
        start=json.loads((OUT/f'{family}_starting_point.json').read_text())
        cv,txt,spec,controls,edits=fn(start)
        if txt is None:
            raise RuntimeError(f'{family}: adjusted rendering failed')
        im=Image.fromarray(cv.image());im.save(OUT/f'{family}_after.png')
        im.resize((im.width*3,im.height*3),Image.Resampling.NEAREST).save(OUT/f'{family}_after_preview.png')
        guard=legibility(cv,txt)
        record=dict(seed=start['seed'],starting_text_spec=start['text_spec'],adjusted_text_spec=spec,
            manual_adjustment_passes=2 if family=='CryptBB' else 1,
            text_control_changes=changed_fields(start['text_spec'],spec),adjusted_render_controls=controls,
            edits=edits,legibility=guard,renderer_labels=T.text_labels(txt),drawing_records=cv.records)
        (OUT/f'{family}_adjustments.json').write_text(json.dumps(record,indent=2))
        comparison(family)
        report['families'][family]=dict(edits=edits,changed_text_controls=list(record['text_control_changes']),
            manual_adjustment_passes=record['manual_adjustment_passes'],legibility=guard)
    (OUT/'adjustment_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
