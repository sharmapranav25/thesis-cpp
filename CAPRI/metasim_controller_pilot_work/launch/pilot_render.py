"""Pilot-only renderer controls; original renderer files are never edited."""
import copy
import hashlib
import sys
from pathlib import Path
import numpy as np

PROJECT = Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0, str(PROJECT))
from feature_bank.iter2.controller import bank, initial_recipe
from feature_bank.iter2.mapping import map_recipe
from feature_bank.iter2.balanced import readability
from captcha_effects import captcha as C, render as R, effects as E, styles as S
from captcha_effects.canvas import Canvas

NAMES = ('background_mix','background_luminance','foreground_mix','foreground_luminance',
         'grey_effect_probability','size_multiplier','width_multiplier','spacing_delta',
         'jitter_delta','weight_fraction_delta','bold_stroke_delta','font_choice',
         'curve_keep_probability','band_level_delta','band_slope_multiplier')
BOUNDS = np.array([(0,1),(.75,1),(0,1),(0,.45),(0,1),(.55,1.2),(.65,1.3),(-.65,.5),
                   (0,.20),(-.3,.3),(0,2),(0,8),(0,1),(-3,3),(0,1)], float)
NEUTRAL = np.array([0,1,0,.2,0,1,1,0,0,0,0,0,1,0,1],float)
ORIG_COLOUR, ORIG_CANVAS, ORIG_TEXT, ORIG_BANDS = S.colour, R.new_canvas, C.draw_text_spec, E.REGISTRY['bands']['fn']
ACTIVE, IN_TEXT = None, False


def colour(rng, palette, background=(255,255,255), text_colour=None):
    value = ORIG_COLOUR(rng,palette,background,text_colour)
    if ACTIVE is not None and IN_TEXT and ACTIVE['foreground_mix'] > 0:
        mix=ACTIVE['foreground_mix']; grey=255*ACTIVE['foreground_luminance']
        value=tuple(np.clip(np.rint((1-mix)*np.asarray(value)+mix*grey),0,255).astype(int))
    return value


def canvas(w,h,spec,rng):
    value=ORIG_CANVAS(w,h,spec,rng)
    if ACTIVE is not None and ACTIVE['background_mix'] > 0:
        mix=ACTIVE['background_mix']; grey=255*ACTIVE['background_luminance']
        bg=tuple(np.clip(np.rint((1-mix)*np.asarray(value.background)+mix*grey),0,255).astype(int))
        value=Canvas(w,h,bg)
    return value


def text(*args,**kwargs):
    global IN_TEXT
    IN_TEXT=True
    try:return ORIG_TEXT(*args,**kwargs)
    finally:IN_TEXT=False


def bands(cv,level,rng,**kw):
    slope=1 if ACTIVE is None else ACTIVE['band_slope_multiplier']
    if slope >= 1-1e-12:return ORIG_BANDS(cv,level,rng,**kw)
    if level<=0:return {}
    s=level/10;n=int(round(1+4*s));pal=E._pal(rng,cv,kw,('grey','near_background','muted','pastel'))
    opacity=float(rng.choice([1.,1.,.75]));w_lo=max(3.,(.04+.06*s)*cv.h)
    params=dict(n=n,palette=pal,opacity=opacity,slope_multiplier=slope)
    for _ in range(n):
        wd=float(rng.uniform(w_lo,w_lo+(.03+.08*s)*cv.h));y0,y1=rng.uniform(0,cv.h),rng.uniform(0,cv.h)
        y1=y0+float(np.clip(y1-y0,-.6*cv.h,.6*cv.h))*slope
        pts=[(-.05*cv.w,y0),(1.05*cv.w,y1)]
        style=dict(kind='clean',antialias=bool(rng.random()<.5))
        E._polyline(cv,rng,pts,wd,S.colour(rng,pal,cv.background,cv.text_colour),'bands','bands',
                    kw.get('where','under'),opacity,style,{})
    return params


def install():
    S.colour=colour;R.new_canvas=canvas;C.draw_text_spec=text;E.REGISTRY['bands']['fn']=bands


def prepare(item):
    description,seed,index=item
    es,ts=initial_recipe(description,np.random.default_rng(seed))
    mapped=map_recipe(bank(),tuple(description['size']),es,ts,description,seed,passes=1)
    return dict(description=description,seed=seed,index=index,size=description['size'],
                effects_spec=mapped['effects_spec'],text_spec=mapped['text_spec'],mapping_calls=mapped['calls'])


def render(item):
    global ACTIVE
    recipe,vector,font_choices=item
    p=dict(zip(NAMES,np.clip(np.asarray(vector,float),BOUNDS[:,0],BOUNDS[:,1])))
    p['font_choice']=int(round(p['font_choice']));p['bold_stroke_delta']=int(round(p['bold_stroke_delta']))
    p['band_level_delta']=int(round(p['band_level_delta']))
    es,ts=copy.deepcopy(recipe['effects_spec']),copy.deepcopy(recipe['text_spec'])
    ts['size']=float(np.clip(ts['size']*p['size_multiplier'],.24,.9))
    ts['x_scale']=float(np.clip(ts['x_scale']*p['width_multiplier'],.4,1.8))
    ts['spacing']=float(np.clip(ts['spacing']+p['spacing_delta'],-.8,1.8))
    ts['jitter']=float(np.clip(ts['jitter']+p['jitter_delta'],0,.5))
    if p['font_choice']:
        ts['fonts']=[font_choices[p['font_choice']-1]]
    if abs(p['weight_fraction_delta'])>1e-12:
        ts['weight']=float(np.clip((ts.get('weight') if ts.get('weight') is not None else .5)+p['weight_fraction_delta'],0,1))
    ts['bold_stroke']=int(np.clip(ts.get('bold_stroke',0)+p['bold_stroke_delta'],0,5))
    control_rng=np.random.default_rng(recipe['seed']+913749)
    kept=[]
    for effect in es['effects']:
        if effect['name']=='curves' and control_rng.random()>=p['curve_keep_probability']:continue
        if effect['name']=='bands':effect['level']=int(np.clip(effect['level']+p['band_level_delta'],1,10))
        if control_rng.random()<p['grey_effect_probability']:
            effect.setdefault('style',{})['palette']='grey'
        kept.append(effect)
    es['effects']=kept
    ACTIVE=p
    try:out=C.render_captcha(np.random.default_rng(recipe['seed']),bank(),tuple(recipe['size']),es,ts)
    finally:ACTIVE=None
    if out is None:
        return dict(image=np.full((recipe['size'][1],recipe['size'][0],3),255,np.uint8),
                    usable=False,strict=False,text_complete=False,index=recipe['index'],error='No glyphs')
    complete=out['text']['text']==ts['text'] and len(out['text']['slots'])==len(ts['text']) and all(g['box'] is not None for g in out['text']['slots'])
    strict,guard=readability(out)
    return dict(image=out['image'],index=recipe['index'],seed=recipe['seed'],text=out['text']['text'],
                text_complete=bool(complete),usable=bool(out['text']['usable'] and complete),strict=bool(strict and complete),
                guard=guard,font=out['text']['style'],glyphs=out['text']['slots'],effects=out['effects'],
                text_features=out['text']['heads'],effects_spec=es,text_spec=ts,controls=p,
                pixel_sha256=hashlib.sha256(out['image'].tobytes()).hexdigest())


def plain(recipe):
    return C.render_captcha(np.random.default_rng(recipe['seed']),bank(),tuple(recipe['size']),recipe['effects_spec'],recipe['text_spec'])
