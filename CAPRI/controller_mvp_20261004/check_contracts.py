"""Meaningful contract checks: replay, palette isolation, masks, and data leakage."""
import common as U
import copy
import numpy as np
from PIL import Image
import render_control as R
from comparison import Comparator,energy
from run import validate_split

def main():
    recipes=U.read(U.WORK/'feature_ablation_20261004/run_03/City/fit_recipes.json')[:12]
    originals=[R.original(r) for r in recipes];R.install()
    neutral=[R.render((r,R.NEUTRAL)) for r in recipes]
    assert all(np.array_equal(a['image'],b['image']) for a,b in zip(originals,neutral))
    for recipe,before in zip(recipes,neutral):
        p=copy.deepcopy(R.NEUTRAL);p['effect_palette']=[[150,150,150]]
        after=R.render((recipe,p));assert after['background']==before['background'];assert after['text']==before['text']
    # Remove effects so text-palette correctness can be checked on exact glyph pixels.
    recipe=copy.deepcopy(recipes[0]);recipe['effects_spec']=dict(background=[250,250,250],effects=[])
    before=R.render((recipe,R.NEUTRAL));effect_only=copy.deepcopy(R.NEUTRAL);effect_only['effect_palette']=[[150,150,150]]
    after=R.render((recipe,effect_only));assert np.array_equal(before['image'],after['image'])
    p=copy.deepcopy(R.NEUTRAL);p.update(text_palette=[[25,50,75],[175,30,80]],color_mode='alternating')
    row=R.render((recipe,p));pixels=row['image'][row['mask']>0]
    assert all(np.any(np.all(pixels==c,axis=1)) for c in p['text_palette'])
    assert row['text']==recipe['text_spec']['text'] and row['complete']
    model=Comparator();a=model.extract([row['image']]);b=model.extract([row['image'].copy()])
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert all(energy(a[k],b[k])<1e-8 for k in a if k!='mask')
    split=U.read(U.ROOT/'demo_split.json');validate_split(split)
    bad=copy.deepcopy(split);bad['test'][0]=bad['fit'][0]
    try:validate_split(bad)
    except ValueError:pass
    else:raise AssertionError('Split leakage was accepted')
    U.dump(U.ROOT/'checks.json',dict(neutral_pixel_replays=12,effect_palette_background_isolation=12,
        explicit_multicolor=True,same_pixels_same_comparison=True,leaking_split_rejected=True,passed=True))
    print('All controller contract checks passed.')
if __name__=='__main__':main()
