"""Freeze a City demo split. Family name only selects files, never control settings."""
import common as U
import hashlib
import numpy as np
from PIL import Image
def hashes(node):
    if isinstance(node,dict):
        if 'pixel_sha256' in node:yield node['pixel_sha256']
        for value in node.values():yield from hashes(value)
    elif isinstance(node,list):
        for value in node:yield from hashes(value)
def main():
    previous=U.read(U.WORK/'feature_ablation_20261004/run_03/City/target_splits.json')
    used=set()
    for path in [U.WORK/'metasim_controller_pilot_work/run_20261004_01/target_split.json',U.WORK/'metasim_controller_pilot_work/feedback_ablation_20261004_02/target_split.json',U.WORK/'feature_ablation_20261004/run_03/City/target_splits.json']:
        used.update(hashes(U.read(path)))
    sources=[r['source'] for r in map(__import__('json').loads,(U.WORK/'iter2_full_target_recreation_01/predictions.jsonl').read_text().splitlines()) if r['family']=='City']
    fresh={}
    for source in sources:
        im=np.asarray(Image.open(source).convert('RGB'));digest=hashlib.sha256(im.tobytes()).hexdigest()
        if digest not in used:fresh.setdefault(digest,dict(source=source,pixel_sha256=digest))
    rows=list(fresh.values());np.random.default_rng(74191).shuffle(rows)
    if len(rows)<256:raise RuntimeError('Not enough previously unused targets')
    U.dump(U.ROOT/'demo_split.json',dict(fit=previous['fit'][:128],selection=previous['selection'][:64],test=rows[:256],
        provenance=dict(previously_used_unique=len(used),previously_unused_available=len(rows),test_previously_unused=True)))
    print('Frozen 128 fit / 64 selection / 256 previously unused test pixels. Available unused:',len(rows))
if __name__=='__main__':main()
