"""Additional generic controls. Existing renderer and v1 behavior remain available."""
from .. import runtime as U
import copy
import numpy as np
from . import base as R

DEFAULT=dict(length=None,jitter_absolute=None,rotation_absolute=None,bold_extra=0,font_pool=None,background_pool=None,block_layout=None,block_speckle=None,
             effect_styles={},keep_only=None)
def expand(p):return {**copy.deepcopy(R.NEUTRAL),**copy.deepcopy(DEFAULT),**copy.deepcopy(p)}
def install():R.install()

def render(item):
    recipe,p=item;p=expand(p);recipe=copy.deepcopy(recipe);ts=recipe['text_spec'];es=recipe['effects_spec']
    rng=np.random.default_rng(recipe['seed']+19191)
    if p['length'] is not None:
        length=int(p['length'])
        if not 2<=length<=8:raise ValueError('Length outside renderer support')
        # Fixed content group, fresh deterministic synthetic string; never target text.
        if len(ts['text'])!=length:ts['text']=R.T.make_text(rng,length,recipe['description']['content'])
    if p['jitter_absolute'] is not None:ts['jitter']=float(p['jitter_absolute'])
    if p['rotation_absolute'] is not None:ts['rotation']=float(p['rotation_absolute'])
    ts['bold_stroke']=int(ts.get('bold_stroke',0)+p['bold_extra'])
    if p['font_pool']:ts['fonts']=p['font_pool'];ts['font_assign']='random'
    applied=copy.deepcopy(p)
    if p['font_pool']:applied['font']=None
    if p['background_pool']:applied['background']=p['background_pool'][int(rng.integers(len(p['background_pool'])))]
    if p['keep_only'] is not None:
        allowed=set(p['keep_only']);es['effects']=[e for e in es['effects'] if e['name'] in allowed]
        applied['effects']={k:v for k,v in p['effects'].items() if k in allowed}
    existing={e['name']:e for e in es['effects']}
    for name,style in p['effect_styles'].items():
        e=existing.setdefault(name,dict(name=name,level=applied['effects'].get(name,3)))
        e.setdefault('style',{}).update(style)
    if p['block_layout'] is not None:existing.setdefault('blocks',dict(name='blocks',level=3)).setdefault('style',{})['layout']=p['block_layout']
    if p['block_speckle'] is not None:existing.setdefault('blocks',dict(name='blocks',level=3)).setdefault('style',{})['speckle']=p['block_speckle']
    es['effects']=list(existing.values())
    row=R.render((recipe,applied));row['controls']=p
    return row
