"""Independent corpus integrity/replay audit; no training or target inputs."""
import argparse
import collections
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,pixel_hash,digest
from feature_bank.labeled_bricks import compose,evidence,crop_labels,FEATURES
from feature_bank.regions import check_group_partitions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
    run=Path(args.run);parents=read(run/'parents.json');protocol=read(run/'protocol.json')
    rows=[json.loads(line) for line in (run/'bricks.jsonl').read_text().splitlines()]
    grouped=collections.defaultdict(list)
    for row in rows:grouped[row['parent_id']].append(row)
    retained=[r for r in rows if r['retained']];check_group_partitions(retained)
    assert len({r['crop_pixel_sha256'] for r in retained})==len(retained)
    assert all(r['training_eligible'] is False for r in rows)
    reserved={'straight_interference','dot_scatter'}
    heldfont=protocol['withheld_font'];hashes=[];small=[];parts=[]
    sampled=[]
    for part in sorted({p['partition'] for p in parents}):
        subset=[p for p in parents if p['partition']==part]
        sampled.extend(p['id'] for p in subset[::max(1,len(subset)//10)][:10])
    for parent in parents:
        with Image.open(run/parent['file']) as image:
            image=image.convert('RGB')
            assert pixel_hash(image)==parent['parent_pixel_sha256']
            for row in grouped[parent['id']]:
                assert pixel_hash(image.crop(row['box']))==row['crop_pixel_sha256']
            gray=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype='float32')
            frequencies=cv2.dct(gray)[:8,:8].flatten()[1:]
            hashes.append(np.packbits(frequencies>np.median(frequencies)))
            small.append(np.asarray(image.resize((96,32),Image.Resampling.BILINEAR),dtype='int16'))
        parts.append(parent['partition'])
        if parent['partition']!='withheld_font':assert parent['font']!=heldfont
        if parent['partition'] not in ('withheld_font','withheld_combination'):assert set(parent['enabled'])!=reserved
        if parent['id'] in sampled:
            folder=run/parent['base_group'];recipe=read(folder/'recipe.json')
            with Image.open(folder/'clean.png') as base:
                background=Image.new('RGB',base.size,tuple(recipe['profile']['background'][0]))
                replay,ev=evidence(base.convert('RGB'),recipe['operations'],parent['enabled'],background)
            assert pixel_hash(replay)==parent['parent_pixel_sha256']
            with np.load(run/(parent['id']+'_evidence.npz')) as saved:
                for f in FEATURES:
                    for key in ('raw','visible'):assert np.array_equal(saved[f+'_'+key],ev[f][key])
            for row in grouped[parent['id']]:
                labels,_=crop_labels(recipe['operations'],ev,row['box'])
                if parent['accepted']:
                    assert all(labels[f]==row['labels'][f] for f in FEATURES)
    # Candidate screen only: 63-bit DCT hash distance <=4 followed by RGB MAE
    # <=4 after size normalization. This is not proof of no perceptual leakage.
    hashes=np.array(hashes);parts=np.array(parts);lookup=np.array([int(i).bit_count() if hasattr(int,'bit_count') else bin(i).count('1') for i in range(256)])
    near=[];screened=0
    for start in range(0,len(parents),128):
        distances=lookup[np.bitwise_xor(hashes[start:start+128,None,:],hashes[None,:,:])].sum(axis=2)
        for a,b in np.argwhere(distances<=4):
            a=int(a)+start;b=int(b)
            if a>=b or parts[a]==parts[b]:continue
            screened+=1;mae=float(np.abs(small[a]-small[b]).mean())
            if mae<=4:near.append(dict(a=parents[a]['id'],b=parents[b]['id'],mean_absolute_rgb_difference=mae))
    result=dict(parent_hashes_verified=len(parents),crop_hashes_verified=len(rows),retained_unique_crops=len(retained),
        grouped_partitions_pass=True,withheld_font_pass=True,withheld_combination_pass=True,
        sampled_replays_masks_labels_verified=len(sampled),sampled_ids=sampled,
        near_duplicate_hash_candidates=screened,near_duplicate_pixel_candidates=near,
        near_duplicate_screen_limit='Heuristic screen, not a proof; all flagged pairs require review before training',
        training_eligible=False,source_sha256=digest(Path(__file__)))
    write(run/'integrity_audit.json',result)
    snapshot=run/'code_snapshot/scripts/audit_brick_corpus.py';snapshot.write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('sampled_ids','near_duplicate_pixel_candidates')},indent=2))
    print('Near-duplicate pairs requiring review:',len(near))

if __name__=='__main__':main()
