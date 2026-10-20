"""Semantic renderer regression checks, including nontrivial overlap and ID tests."""
import copy
import hashlib
import numpy as np
from .common import read, write, now
from .renderer import InstanceCanvas, bold, render, box
from .sampling import Sampler


def check_renderer(run):
    checks = {}
    def toy(left_id,right_id):
        cv=InstanceCanvas(15,11)
        for gid,x,col in [(left_id,4,(20,0,0)),(right_id,8,(0,0,90))]:
            m=np.zeros((11,15),np.float32);m[5,x]=1
            cv.geometry[gid]=m.copy();cv.visible[gid]=m.copy();cv.decoration[gid]=np.zeros_like(m)
            cv.opaque[5,x]=gid;cv.img[5,x]=col
        bold(cv,10)
        return cv
    a,b=toy(1,2),toy(2,1)
    assert a.opaque[5,5]==1 and b.opaque[5,5]==2
    assert np.array_equal(a.img,b.img)
    assert np.array_equal(a.geometry[1],b.geometry[2])
    checks['bold_id_permutation_and_nearest_colour']=True
    cv=InstanceCanvas(10,10)
    m=np.zeros((10,10),np.float32);m[2:7,2:7]=1
    n=np.zeros_like(m);n[4:9,4:9]=1
    x=cv.blend(m,(0,0,0),'text','text','text')
    y=cv.blend(n,(255,0,0),'text','text','text',.5)
    assert cv.geometry[x][5,5]==1 and cv.visible[x][5,5]==.5 and cv.visible[y][5,5]==.5
    checks['overlap_preserves_separate_geometry_and_alpha']=True
    manifest=read(run/'fonts/font_manifest.json')
    sampler=Sampler(manifest)
    font=next(f for f in manifest['fonts'] if f['family'].lower()=='roboto' and all(c in f['chars'] for c in 'Ai7gW2'))
    recipe=sampler.recipe(font['split'],0)
    recipe.update(size=[300,96],background=[255,255,255],effects=[])
    recipe['text'].update(text='A7',font_keys=[font['key']]*2,axes=[{},{}],colours=[[0,0,0]]*2,layout='two_layer',
                           rotation=0.,jitter=0.,curve=0.,size_jitter=0.,shear=0.,stroke=0)
    out=render(recipe,sampler.fonts)
    assert out['count']==2 and out['rows']==2 and out['valid']
    assert out['glyphs'][0]['box_core'][1] != out['glyphs'][1]['box_core'][1]
    checks['two_character_two_rows_actually_rendered']=True
    again=render(recipe,sampler.fonts)
    assert np.array_equal(out['image'],again['image']) and np.array_equal(out['visible'],again['visible'])
    checks['deterministic_replay']=True
    base=copy.deepcopy(recipe)
    base['text'].update(text='Ai7gW2',font_keys=[font['key']]*6,axes=[{}]*6,colours=[[0,0,0]]*6,layout='line')
    for name in sampler.effects:
        r=copy.deepcopy(base);r['effects']=[dict(name=name,level=4)]
        o=render(r,sampler.fonts)
        assert o['count']==6 and np.isfinite(o['geometry']).all() and np.isfinite(o['visible']).all(),name
        assert all(np.min(o[k])>=0 and np.max(o[k])<=1.00001 for k in ['geometry','visible','decoration']),name
        for i,g in enumerate(o['glyphs']):
            assert g['box_core']==box(o['geometry'][i]) and g['box_visible']==box(o['visible'][i]),name
    checks['all_effects_finite_masks_and_box_alignment']=True
    r=copy.deepcopy(base);r['text']['text']='\u2603'+r['text']['text'][1:]
    try:
        render(r,sampler.fonts)
    except ValueError as exc:
        assert 'Missing glyph' in str(exc)
    else:
        raise AssertionError('Unsupported character silently accepted')
    checks['missing_glyph_rejected']=True
    families={}
    for f in manifest['fonts']:
        assert f['family_group'] not in families or families[f['family_group']]==f['split']
        families[f['family_group']]=f['split']
    checks['font_family_split_disjoint']=True
    result=dict(created_at=now(),passed=True,checks=checks,n_effects=len(sampler.effects))
    write(run/'renderer_checks.json',result)
    return result
