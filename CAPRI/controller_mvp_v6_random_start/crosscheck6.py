"""Post-freeze comparison with the previous non-DINO visual comparator."""
from ablate6 import ROOT,U,FAMILIES,load_images
from perception3 import Perception,Objective
from PIL import Image
import numpy as np

def main():
    summary=U.read(ROOT/'summary.json');assert len(summary)==5
    model=Perception();out={}
    for family,result in summary.items():
        dest=ROOT/family;target=model.extract_pair(load_images(U.read(dest/'target_splits.json')['test']))[1]
        features={};rows={}
        for arm,r in result['runs'].items():
            assert U.sha(dest/arm/'selected.json')==r['selected_sha256']
            rows[arm]=U.read(dest/arm/'final/labels.json')
            pixels=[np.asarray(Image.open(dest/arm/row['image']).convert('RGB')) for row in rows[arm]]
            features[arm]=model.extract_pair(pixels)[1]
        objective=Objective(target,features['hybrid'])
        out[family]={a:objective.score(f,target,rows[a]) for a,f in features.items()}
        print(family,{a:round(s['appearance'],3) for a,s in out[family].items()},flush=True)
    U.dump(ROOT/'crosscheck.json',dict(note='After all selections frozen; no feedback to search. Common diagnostic normalization per family. Previous comparator is a second learned metric, not ground truth.',scores=out))

if __name__=='__main__':main()
