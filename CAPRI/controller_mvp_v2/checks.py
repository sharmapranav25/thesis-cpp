from bootstrap import ROOT,V1,U
import numpy as np
import copy
from PIL import Image
import controls as C
from perception import Perception,Objective,extra

def main():
    C.install();checks=[]
    for family,path in [('City','run_03'),('CryptBB','CryptBB_run_01'),('DarkBay','DarkBay_run_01')]:
        recipes=U.read(V1/path/'test_recipes.json')[:4];p=U.read(V1/path/'selected_controls.json')
        for r in recipes:
            a=C.R.render((r,p));b=C.render((r,C.expand(p)))
            assert np.array_equal(a['image'],b['image'])
            q=C.expand(p);q.update(length=6,jitter_absolute=.12,rotation_absolute=12)
            row=C.render((r,q));assert len(row['text'])==6 and row['text']==row['text_spec']['text']
        checks.append(dict(family=family,exact_replays=4,length_changes=4))
    # These are algorithm contracts, not evidence of real reconstruction quality.
    zero=extra(np.full((64,256,3),255,np.uint8),np.zeros((64,256)))
    assert len(zero[0])==103 and np.isfinite(zero[0]).all()
    r=copy.deepcopy(recipes[0]);r['effects_spec']=dict(background=[240,240,240],effects=[])
    q=C.expand(C.R.NEUTRAL);q.update(length=5,text_palette=[[20,20,20]],color_mode='single',font='dejavusans' if 'dejavusans' in C.R.bank().by_key else None)
    rows=[C.render((r,q))];model=Perception();f=model.extract([rows[0]['image']]);obj=Objective(f,f);score=obj.score(f,f,rows)
    assert all(abs(v)<1e-8 for v in score['components'].values())
    U.dump(ROOT/'checks.json',dict(passed=True,replays=checks,empty_mask_descriptor_finite=True,identical_pixel_distance_zero=True))
    print('V2 checks passed: 12 exact v1 replays, 12 length changes, empty masks and self-distance.')
if __name__=='__main__':main()
