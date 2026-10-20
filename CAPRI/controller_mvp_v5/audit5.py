"""Integrity and independent legacy comparisons after all DINO selections freeze."""
from setup5 import ROOT,V4,U,FAMILIES,WEIGHTS_PATH
from run import load_images
from perception3 import Perception,Objective
from comparison import energy
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import controls as C
import hashlib
import numpy as np
from PIL import Image

def main():
    summary=U.read(ROOT/'summary.json');assert len(summary)==5
    protocol=U.read(ROOT/'protocol.json');assert U.sha(WEIGHTS_PATH)==protocol['weights_sha256'];assert U.read(ROOT/'checks.json')['passed']
    for name,digest in protocol['code_sha256'].items():assert U.sha(ROOT/name)==digest,'Controller implementation changed during search'
    model=Perception();cross={};images=0;profile_probe={}
    for family in FAMILIES:
        dest=ROOT/family;s=summary[family];assert U.sha(dest/'selected.json')==s['selected_sha256'] and U.sha(dest/'archive.json')==s['archive_sha256']
        archive=U.read(dest/'archive.json');assert len({__import__('json').dumps(p,sort_keys=True) for p in archive})==len(archive)
        probes=U.read(dest/'test_recipes.json')[:3]
        with ProcessPoolExecutor(max_workers=8,initializer=C.install) as pool:
            outputs=list(pool.map(C.render,[(recipe,p) for p in archive for recipe in probes]))
        signatures=[tuple(r['pixel_sha256'] for r in outputs[i:i+3]) for i in range(0,len(outputs),3)]
        profile_probe[family]=dict(saved_profiles=len(archive),distinct_render_signatures_on_three_fixed_recipes=len(set(signatures)),note='A finite probe detects some equivalent settings, not proof of global profile uniqueness.')
        split=U.read(dest/'target_splits.json');sets=[{r['pixel_sha256'] for r in split[k]} for k in ['fit','selection','test']];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
        target=model.extract_pair(load_images(split['test']))[1];features={};rows={}
        for arm in ['hybrid','controller1','dino','cls_only']:
            rows[arm]=U.read(dest/arm/'labels.json');pixels=[]
            for row in rows[arm]:
                a=np.asarray(Image.open(dest/row['image']).convert('RGB'));assert hashlib.sha256(a.tobytes()).hexdigest()==row['pixel_sha256'];pixels.append(a);images+=1
            features[arm]=model.extract_pair(pixels)[1]
        for i in range(len(rows['hybrid'])):assert len({rows[arm][i]['seed'] for arm in rows})==1
        # Post-selection descriptive scores. Their normalization uses this diagnostic
        # target batch equally for all arms, not the fitting or selection process.
        objective=Objective(target,features['hybrid']);cross[family]={arm:objective.score(f,target,rows[arm]) for arm,f in features.items()}
        U.dump(dest/'legacy_crosscheck.json',dict(phase='After frozen selection; no feedback to DINO search',normalization='Common diagnostic test batch for all arms',scores=cross[family]))
        assert U.sha(dest/'selected.json')==s['selected_sha256']
        assert U.sha(V4/'collections'/family/'archive.json')==U.read(V4/'collections/summary.json')[family]['archive_sha256']
    U.dump(ROOT/'audit.json',dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),model_frozen=True,controller1_unchanged=True,distinct_profile_controls=True,fit_selection_test_disjoint=True,
        matched_renderer_seeds=True,verified_generated_images=images,target_answers_read=False,no_vlm=True,legacy_crosscheck=cross,
        archive_render_probe=profile_probe,
        source_sha256={p.name:U.sha(p) for p in ROOT.glob('*.py')}))
    print('Final checks passed:',images,'generated images, frozen DINO, unchanged C1')

if __name__=='__main__':main()
