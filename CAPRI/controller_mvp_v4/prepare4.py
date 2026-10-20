"""Reserve target pixels without opening answer annotations or parsing filenames."""
from setup4 import ROOT,V3,V1,U,FAMILIES
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_demo import hashes

def main():
    records=[json.loads(line) for line in (U.WORK/'iter2_full_target_recreation_01/predictions.jsonl').read_text().splitlines()]
    summary={};dest=ROOT/'evaluation_pixels';dest.mkdir(exist_ok=False)
    for family in FAMILIES:
        prior=U.read(V3/(family+'_run_01')/'target_splits.json');excluded=set(hashes(prior));excluded.update(r['pixel_sha256'] for r in prior['fresh'])
        previous=V1/('run_03' if family=='City' else family+'_run_01');excluded.update(hashes(U.read(previous/'target_splits.json')))
        if family in ['City','CryptBB','DarkBay']:
            paths=[r['source'] for r in records if r['family']==family]
            excluded.update(hashes(U.read(U.WORK/f'feature_ablation_20261004/run_03/{family}/target_splits.json')))
            if family=='City':
                for run in ['run_20261004_01','feedback_ablation_20261004_02']:excluded.update(hashes(U.read(U.WORK/f'metasim_controller_pilot_work/{run}/target_split.json')))
        elif family=='Outlined':
            paths=[str(p) for p in (U.PROJECT/'data/external/outlined_text_captcha/images').glob('*.png')]
            original=U.read(U.WORK/'length_instances_02/real_inference.json')['outlined_text_captcha']['paths']
            excluded.update(hashlib.sha256(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest() for p in original)
        else:
            paths=[r['source'] for r in U.read(V3/'simple_captcha_dataset/inventory.json')['rows']]
            excluded.update(hashes(U.read(V3/'SimpleCaptcha_v1_split.json')))
        available={}
        for p in paths:
            pixels=np.asarray(Image.open(p).convert('RGB'));h=hashlib.sha256(pixels.tobytes()).hexdigest()
            if h not in excluded:available.setdefault(h,dict(source=p,pixel_sha256=h))
        available=list(available.values());np.random.default_rng(44017).shuffle(available)
        selected=available[:1000 if family=='SimpleCaptcha' else 128]
        status='fresh_known_controller_splits' if selected else 'reused_diagnostic'
        if not selected:selected=prior['test']
        U.dump(dest/(family+'.json'),dict(family=family,status=status,rows=selected,labels_read=False))
        summary[family]=dict(images=len(selected),status=status,available=len(available));print(family,summary[family],flush=True)
    U.dump(dest/'summary.json',summary)

if __name__=='__main__':main()

