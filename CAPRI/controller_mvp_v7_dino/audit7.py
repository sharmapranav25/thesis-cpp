from setup7 import *
from build7 import fresh,RANDOM_SEED
from run import load_images
from perception3 import Perception,Objective
import render7 as R
import hashlib,json
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

def main():
    summary=U.read(ROOT/'summary.json');assert len(summary)==5
    protocol=U.read(ROOT/'protocol.json')
    for name,sha in protocol['code_sha256'].items():assert U.sha(ROOT/name)==sha
    assert U.sha(WEIGHTS_PATH)==protocol['weights_sha256']
    for name,sha in U.read(V5/'completion.json')['source_sha256'].items():assert U.sha(V5/name)==sha
    fonts=U.read(WORK/'learned_controller_20261004/fonts.json');model=Perception();cross={};probes={};imagecount=0
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:
        for family,result in summary.items():
            dest=ROOT/family;assert U.sha(dest/'selected.json')==result['selected_sha256']
            assert U.sha(dest/'archive.json')==result['archive_sha256']
            split=U.read(dest/'target_splits.json');sets=[{r['pixel_sha256'] for r in split[k]} for k in ['fit','selection','test']]
            assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
            banks=U.read(dest/'recipe_banks.json');size=banks['random']['fit'][0]['size']
            expected=fresh(random_recipes(RANDOM_SEED,48,size,fonts,300),48,300)
            assert json.dumps(expected,sort_keys=True)==json.dumps(banks['random']['test'],sort_keys=True)
            assert all(set(r['description'])=={'content'} for group in [banks['random']['fit'],*banks['random']['validation'],banks['random']['test']] for r in group)
            selection=U.read(dest/'selection.json')
            assert all(len(r['scores'])==2 for r in selection)
            selected=U.read(dest/'selected.json');valid=[r for r in selection if r['eligible']]
            assert selected['robust_score']==min(r['robust_score'] for r in (valid or selection))
            archive=U.read(dest/'archive.json');signatures=[]
            for candidate in archive:
                rows=list(pool.map(R.safe_render,[(r,candidate['controls']) for r in banks[candidate['bank']]['validation'][0][:3]]))
                signatures.append(tuple(r.get('pixel_sha256',r.get('error')) for r in rows))
            probes[family]=dict(saved=len(archive),distinct_on_three_probes=len(set(signatures)))
            target=model.extract_pair(load_images(split['test']))[1];features={};rows={}
            for arm in ['hybrid','dino_v1','dino_v2']:
                rows[arm]=U.read(dest/arm/'labels.json');assert len(rows[arm])==48;pixels=[]
                for row in rows[arm]:
                    im=np.asarray(Image.open(dest/row['image']).convert('RGB'))
                    if 'pixel_sha256' in row:assert hashlib.sha256(im.tobytes()).hexdigest()==row['pixel_sha256']
                    pixels.append(im);imagecount+=1
                features[arm]=model.extract_pair(pixels)[1]
            objective=Objective(target,features['dino_v1']);cross[family]={a:objective.score(f,target,rows[a]) for a,f in features.items()}
            print(family,'legacy',{a:round(s['appearance'],3) for a,s in cross[family].items()},flush=True)
    U.dump(ROOT/'crosscheck.json',dict(note='After selection freezes. Shares some handcrafted descriptors, so not a fully independent metric. Normalized to old DINO final.',scores=cross))
    U.dump(ROOT/'audit.json',dict(passed=True,model_hash_verified=True,previous_controller_unchanged=True,source_unchanged=True,
        disjoint_target_pixels=True,random_bank_reproduced=True,selection_matches_validation=True,generated_images_verified=imagecount,
        archive_probes=probes,all_probe_signatures_unique=all(x['saved']==x['distinct_on_three_probes'] for x in probes.values())))
    print('Audit passed:',imagecount,'images')

if __name__=='__main__':main()
