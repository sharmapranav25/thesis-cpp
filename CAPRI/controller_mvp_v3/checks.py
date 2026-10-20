from boot import ROOT,V1,U
import copy
import numpy as np
from search3 import C,candidates,expand_effect
from perception3 import ink_shape,Perception,Objective

def main():
    C.install();recipes=U.read(V1/'Outlined_run_01/fit_recipes.json')[:4];checks=[]
    for recipe in recipes:
        recipe=copy.deepcopy(recipe);recipe['effects_spec']['effects']=[]
        p=C.expand(C.R.NEUTRAL);p['keep_only']=[];p['bold_extra']=2
        a=C.render((recipe,p));q=expand_effect(p,'hollow',10);b=C.render((recipe,q))
        assert any(e['name']=='hollow' for e in b['effects_spec']['effects'])
        assert not np.array_equal(a['image'],b['image'])
        checks.append(dict(seed=recipe['seed'],hollow_reachable=True,before=int((a['mask']>0).sum()),after=int((b['mask']>0).sum())))
    opts=candidates(p,'all_effects',{},[],[],np.random.default_rng(1))
    assert all(any(x['effects'].get(name)==10 and name in x['keep_only'] for x in opts) for name in C.R.R.SAMPLED)
    # Sanity: topology descriptor separates hollow and filled ink without target labels.
    solid=np.zeros((64,256));solid[16:48,80:112]=1;ring=solid.copy();ring[20:44,84:108]=0
    image=np.ones((64,256,3),np.uint8)*255
    assert ink_shape(image,ring)[0]>ink_shape(image,solid)[0]
    assert np.isfinite(ink_shape(image,np.zeros_like(solid))).all()
    model=Perception();_,features=model.extract_pair([a['image'],b['image']]);obj=Objective(features,features)
    assert all(abs(v)<1e-8 for v in obj.score(features,features,[a,b])['components'].values())
    U.dump(ROOT/'checks.json',dict(passed=True,all_sampled_effects_reachable=len(C.R.R.SAMPLED),hollow=checks,empty_mask_finite=True,topology_discriminates_holes=True,self_distance_zero=True))
    print('All checks passed',flush=True)

if __name__=='__main__':main()
