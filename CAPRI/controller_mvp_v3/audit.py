from boot import ROOT,V2,U
import argparse,hashlib
import numpy as np
from PIL import Image
from run import validate_split

def main():
    ap=argparse.ArgumentParser();ap.add_argument('family');args=ap.parse_args();out=ROOT/(args.family+'_run_01')
    result=U.read(out/'results.json');split=U.read(out/'target_splits.json');validate_split(split)
    assert U.sha(out/'frozen_policies.json')==result['frozen_sha256']
    fresh={r['pixel_sha256'] for r in split['fresh']};seen_targets={r['pixel_sha256'] for key in ['fit','selection','test'] for r in split[key]}
    assert not fresh&seen_targets
    arms={};seeds=None
    for name in ['v2','old','mask','typography','full']:
        rows=U.read(out/name/'labels.json');seen=set()
        for i,r in enumerate(rows):
            pixels=np.asarray(Image.open(out/r['image']).convert('RGB'));digest=hashlib.sha256(pixels.tobytes()).hexdigest()
            assert digest==r['pixel_sha256'] and digest not in seen and digest not in fresh|seen_targets;seen.add(digest)
            mask=np.asarray(Image.open(out/r['mask']));assert mask.shape==pixels.shape[:2]
            assert r['text']==r['text_spec']['text']
            if name=='v2':
                old=np.asarray(Image.open(V2/(args.family+'_run_01')/'expanded'/f'{i:04d}.png').convert('RGB'))
                assert np.array_equal(pixels,old),'v2 replay mismatch'
        current=[r['seed'] for r in rows]
        if seeds is None:seeds=current
        else:assert current==seeds
        arms[name]=len(rows)
    protocol=U.read(out/'protocol.json')
    for filename,digest in protocol['source_sha256'].items():assert U.sha(ROOT/filename)==digest,'source changed during run: '+filename
    report=dict(passed=True,verified_images=arms,exact_v2_replays=arms['v2'],paired_seeds=True,fresh_disjoint=True,controls_frozen=True,source_unchanged=True,failed_candidates=len(result['failures']))
    U.dump(out/'audit.json',report);print(report)

if __name__=='__main__':main()

