"""Explicit controls around the ORIGINAL full-effect renderer. Process-local hooks."""
import common as U
import copy, hashlib
import numpy as np
from feature_bank.iter2.controller import bank, initial_recipe
from feature_bank.iter2.mapping import map_recipe
from feature_bank.iter2.balanced import readability
from captcha_effects import captcha as C, render as R, styles as S, text as T

COLOUR = S.colour
DRAW_TEXT = C.draw_text_spec
NEW_CANVAS = R.new_canvas
ACTIVE = None
NEUTRAL = dict(size=1.,width=1.,spacing=0.,rotation=1.,jitter=1.,weight=0.,font=None,
               background=None,text_palette=None,effect_palette=None,color_mode='original',effects={})

def controlled_colour(rng,palette,background=(255,255,255),text_colour=None):
    original=COLOUR(rng,'contrast' if palette=='mvp_text' else palette,background,text_colour)
    if ACTIVE is None:return original
    if palette=='mvp_text':
        colors=ACTIVE['text_palette'];mode=ACTIVE['color_mode']
        if mode=='single':i=ACTIVE['_single']
        elif mode=='alternating':i=ACTIVE['_index']%len(colors)
        else:i=int(ACTIVE['_rng'].integers(len(colors)))
        ACTIVE['_index']+=1
        return tuple(colors[i])
    if ACTIVE['effect_palette'] and not ACTIVE.get('_background_phase',False) and not ACTIVE.get('_text_phase',False):
        colors=ACTIVE['effect_palette']
        return tuple(colors[int(ACTIVE['_fxrng'].integers(len(colors)))])
    return original

def controlled_text(cv,spec,rng,font_bank,**kwargs):
    if ACTIVE is not None and ACTIVE['text_palette']:
        spec=copy.deepcopy(spec);spec.update(palette='mvp_text',min_contrast=0.,multicolour=ACTIVE['color_mode']!='single')
        kwargs['colour']=tuple(ACTIVE['text_palette'][ACTIVE['_single']])
    if ACTIVE is not None:ACTIVE['_text_phase']=True
    try:return DRAW_TEXT(cv,spec,rng,font_bank,**kwargs)
    finally:
        if ACTIVE is not None:ACTIVE['_text_phase']=False

def controlled_canvas(*args,**kwargs):
    if ACTIVE is not None:ACTIVE['_background_phase']=True
    try:return NEW_CANVAS(*args,**kwargs)
    finally:
        if ACTIVE is not None:ACTIVE['_background_phase']=False

def install():S.colour=controlled_colour;C.draw_text_spec=controlled_text;R.new_canvas=controlled_canvas

def prepare(item):
    description,seed,index=item
    es,ts=initial_recipe(description,np.random.default_rng(seed))
    mapped=map_recipe(bank(),tuple(description['size']),es,ts,description,seed,passes=1)
    return dict(description=description,seed=seed,index=index,size=description['size'],effects_spec=mapped['effects_spec'],text_spec=mapped['text_spec'])

def render(item):
    global ACTIVE
    recipe,controls=item
    es=copy.deepcopy(recipe['effects_spec']);ts=copy.deepcopy(recipe['text_spec'])
    for field,key,low,high in [('size','size',.2,.95),('x_scale','width',.3,2.),('rotation','rotation',0,60),('jitter','jitter',0,.5)]:
        ts[field]=float(np.clip(ts[field]*controls[key],low,high))
    ts['spacing']=float(np.clip(ts['spacing']+controls['spacing'],-.7,1.5))
    if controls['weight']:
        ts['weight']=float(np.clip((ts.get('weight') if ts.get('weight') is not None else .5)+controls['weight'],0,1))
    if controls['font']:ts['fonts']=[controls['font']]
    if controls['background'] is not None:es['background']=controls['background']
    existing={e['name']:e for e in es['effects']}
    for name,level in controls['effects'].items():
        if name not in R.SAMPLED:raise ValueError('Unknown renderer effect '+name)
        if level==0:existing.pop(name,None)
        else:existing[name]={**existing.get(name,{}),'name':name,'level':int(level)}
    es['effects']=list(existing.values())
    ACTIVE=copy.deepcopy(controls)
    ACTIVE.update(_rng=np.random.default_rng(recipe['seed']+311),_fxrng=np.random.default_rng(recipe['seed']+313),_index=0)
    ACTIVE['_single']=int(ACTIVE['_rng'].integers(max(1,len(ACTIVE['text_palette'] or []))))
    try:out=C.render_captcha(np.random.default_rng(recipe['seed']),bank(),tuple(recipe['size']),es,ts)
    finally:ACTIVE=None
    if out is None:raise RuntimeError('Renderer produced no glyphs')
    complete=out['text']['text']==ts['text'] and len(out['text']['slots'])==len(ts['text']) and all(g['box'] is not None for g in out['text']['slots'])
    strict,guard=readability(out)
    ids=[r['id'] for r in out['records'] if r.get('head')=='text']
    mask=np.isin(out['opaque'],ids).astype(np.uint8)*255
    return dict(image=out['image'],mask=mask,background=out['background'],complete=bool(complete),usable=bool(complete and out['text']['usable']),strict=bool(complete and strict),guard=guard,
                seed=recipe['seed'],text=out['text']['text'],glyphs=out['text']['slots'],effects=out['effects'],text_features=out['text']['heads'],
                text_spec=ts,effects_spec=es,controls=controls,pixel_sha256=hashlib.sha256(out['image'].tobytes()).hexdigest())

def original(recipe):
    return C.render_captcha(np.random.default_rng(recipe['seed']),bank(),tuple(recipe['size']),recipe['effects_spec'],recipe['text_spec'])
