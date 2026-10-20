"""Dataset selection only; the controller and search are unchanged."""
import common as U
import json,hashlib
import numpy as np
from PIL import Image
from prepare_demo import hashes
from run import validate_split

def main():
    sources=[{k:r[k] for k in ['source','family']} for r in map(json.loads,(U.WORK/'iter2_full_target_recreation_01/predictions.jsonl').read_text().splitlines())]
    for family,n in [('CryptBB',128),('DarkBay',256)]:
        previous=U.read(U.WORK/f'feature_ablation_20261004/run_03/{family}/target_splits.json')
        used=set(hashes(previous));fresh={}
        for row in sources:
            if row['family']!=family:continue
            im=np.asarray(Image.open(row['source']).convert('RGB'));digest=hashlib.sha256(im.tobytes()).hexdigest()
            if digest not in used:fresh.setdefault(digest,dict(source=row['source'],pixel_sha256=digest))
        rows=list(fresh.values());np.random.default_rng(74191).shuffle(rows)
        if len(rows)<n:raise ValueError('Insufficient unused pixels: '+family)
        split=dict(fit=previous['fit'][:128],selection=previous['selection'][:64],test=rows[:n],
            provenance=dict(family=family,excluded_previous_ablation_splits=True,previously_used_unique=len(used),previously_unused_available=len(rows),test_previously_unused_by_controller=True))
        validate_split(split);U.dump(U.ROOT/f'{family}_split.json',split)
        print(f'{family}: {len(split["fit"])} fit, {len(split["selection"])} selection, {n} fresh test; {len(rows)} unused available',flush=True)
if __name__=='__main__':main()
