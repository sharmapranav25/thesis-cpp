from boot import ROOT,V1,V2,U
import hashlib,json
import numpy as np
from PIL import Image
from prepare_demo import hashes
from run import validate_split

def main():
    inventory=[json.loads(line) for line in (U.WORK/'iter2_full_target_recreation_01/predictions.jsonl').read_text().splitlines()]
    for family,previous in [('City','run_03'),('CryptBB','CryptBB_run_01'),('DarkBay','DarkBay_run_01'),('Outlined','Outlined_run_01')]:
        diagnostic=U.read(V2/(family+'_run_01')/'target_splits.json');excluded=set(hashes(diagnostic))
        excluded.update(hashes(U.read(V1/previous/'target_splits.json')))
        if family!='Outlined':
            excluded.update(hashes(U.read(U.WORK/f'feature_ablation_20261004/run_03/{family}/target_splits.json')))
            paths=[r['source'] for r in inventory if r['family']==family]
            if family=='City':
                for p in ['run_20261004_01','feedback_ablation_20261004_02']:excluded.update(hashes(U.read(U.WORK/f'metasim_controller_pilot_work/{p}/target_split.json')))
        else:
            paths=[str(p) for p in (U.PROJECT/'data/external/outlined_text_captcha/images').glob('*.png')]
            prior=U.read(U.WORK/'length_instances_02/real_inference.json')['outlined_text_captcha']['paths']
            excluded.update(hashlib.sha256(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest() for p in prior)
        remaining={}
        for p in paths:
            digest=hashlib.sha256(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest()
            if digest not in excluded:remaining.setdefault(digest,dict(source=p,pixel_sha256=digest))
        remaining=list(remaining.values());np.random.default_rng(96317).shuffle(remaining)
        split=dict(fit=diagnostic['fit'],selection=diagnostic['selection'],test=diagnostic['test'],fresh=remaining[:128],
            provenance=dict(diagnostic_test='Reused v2 test; development diagnostic, not fresh evidence',fresh='Excluded known controller, ablation and localization split hashes; not a claim of no prior project exposure',fresh_available=len(remaining)))
        validate_split(split);assert not set(hashes(diagnostic))&set(r['pixel_sha256'] for r in split['fresh'])
        U.dump(ROOT/(family+'_split.json'),split);print(family,'fresh',len(split['fresh']),flush=True)

if __name__=='__main__':main()

