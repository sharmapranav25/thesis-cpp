"""Verify saved generated pixels, labels, frozen policy and evaluation provenance."""
import common as U
import argparse,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from run import validate_split
def main():
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();out=Path(args.run)
    results=U.read(out/'results.json');split=U.read(out/'target_splits.json');validate_split(split)
    assert U.sha(out/'selected_controls.json')==results['controls_sha256']
    counts={};labels={}
    for group in ['baseline','refined']:
        records=U.read(out/group/'labels.json');labels[group]=records;seen=set()
        for r in records:
            im=np.asarray(Image.open(out/r['image']).convert('RGB'));mask=np.asarray(Image.open(out/r['mask']))
            digest=hashlib.sha256(im.tobytes()).hexdigest()
            assert digest==r['pixel_sha256'] and digest not in seen
            seen.add(digest);assert im.shape[:2]==mask.shape and set(np.unique(mask)).issubset({0,255})
            assert r['text']==r['text_spec']['text']
            assert len(r['glyphs'])==len(r['text'])
        counts[group]=len(records)
    assert counts['baseline']==counts['refined']
    for a,b in zip(labels['baseline'],labels['refined']):assert a['seed']==b['seed'] and a['text']==b['text']
    assert not set(r['pixel_sha256'] for rows in labels.values() for r in rows)&set(r['pixel_sha256'] for key in ['fit','selection','test'] for r in split[key])
    history=U.read(out/'history.json')
    checks=dict(passed=True,verified_images=counts,paired_seeds_and_strings=True,target_pixel_copies=False,
        frozen_controls_match=True,failed_candidates=sum('error' in r for r in history),target_splits_disjoint=True,
        source_sha256={p.name:U.sha(p) for p in U.ROOT.glob('*.py')})
    U.dump(out/'verification.json',checks);print(__import__('json').dumps(checks,indent=2))
if __name__=='__main__':main()
