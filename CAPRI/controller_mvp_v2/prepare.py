from bootstrap import ROOT,V1,U
from prepare_demo import hashes
import hashlib,json
import numpy as np
from PIL import Image
from run import validate_split
def main():
    rows=[{k:r[k] for k in ['family','source']} for r in map(json.loads,(U.WORK/'iter2_full_target_recreation_01/predictions.jsonl').read_text().splitlines())]
    for family,previous,n in [('City','run_03',128),('CryptBB','CryptBB_run_01',24),('DarkBay','DarkBay_run_01',128)]:
        old=U.read(V1/previous/'target_splits.json');excluded=set(hashes(old));excluded.update(hashes(U.read(U.WORK/f'feature_ablation_20261004/run_03/{family}/target_splits.json')))
        if family=='City':
            for path in ['run_20261004_01','feedback_ablation_20261004_02']:excluded.update(hashes(U.read(U.WORK/f'metasim_controller_pilot_work/{path}/target_split.json')))
        fresh={}
        for r in rows:
            if r['family']!=family:continue
            im=np.asarray(Image.open(r['source']).convert('RGB'));h=hashlib.sha256(im.tobytes()).hexdigest()
            if h not in excluded:fresh.setdefault(h,dict(source=r['source'],pixel_sha256=h))
        fresh=list(fresh.values());np.random.default_rng(85041).shuffle(fresh)
        if len(fresh)<n:raise ValueError('Insufficient fresh test data')
        split=dict(fit=old['fit'],selection=old['selection'],test=fresh[:n],provenance=dict(unused_available=len(fresh),excluded=len(excluded),family=family))
        validate_split(split);U.dump(ROOT/f'{family}_split.json',split);print(family,'fresh test',n,'available',len(fresh),flush=True)
if __name__=='__main__':main()
