from bootstrap import ROOT,U
import argparse,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from run import validate_split

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');args=ap.parse_args();out=Path(args.run)
    result=U.read(out/'results.json');split=U.read(out/'target_splits.json');validate_split(split)
    assert U.sha(out/'frozen_policies.json')==result['frozen_sha256']
    features=np.load(out/'test_features.npz');summary={};labels={}
    target_hashes={r['pixel_sha256'] for group in ['fit','selection','test'] for r in split[group]}
    for group in ['mvp1','old_comparator','new_comparator','expanded']:
        rows=U.read(out/group/'labels.json');labels[group]=rows;seen=set();ious=[]
        for i,r in enumerate(rows):
            im=np.asarray(Image.open(out/r['image']).convert('RGB'));mask=np.asarray(Image.open(out/r['mask']))
            h=hashlib.sha256(im.tobytes()).hexdigest()
            assert h==r['pixel_sha256'] and h not in seen and h not in target_hashes;seen.add(h)
            assert r['text']==r['text_spec']['text'] and len(r['text'])==len(r['glyphs'])
            if r['controls']['length'] is not None:assert len(r['text'])==r['controls']['length']
            assert im.shape[:2]==mask.shape
            truth=np.asarray(Image.fromarray(mask).resize((256,64),Image.Resampling.NEAREST))>0
            pred=features[group+'_mask'][i]>.5
            ious.append(float((pred&truth).sum()/max(1,(pred|truth).sum())))
        summary[group]=dict(verified=len(rows),synthetic_mask_mean_iou=float(np.mean(ious)),
                           lengths=sorted(set(len(r['text']) for r in rows)))
    for group in labels:
        assert [r['seed'] for r in labels[group]]==[r['seed'] for r in labels['mvp1']]
    report=dict(passed=True,arms=summary,failed_candidates=len(result['failures']),frozen_policy_unchanged=True,
                paired_seeds=True,strings_may_differ_due_to_length_controls=True,
                real_mask_diagnostics=dict(uncertain_pixel_fraction=float(features['target_mask_diagnostic'][:,0].mean()),
                                           predicted_text_pixel_fraction=float(features['target_mask_diagnostic'][:,1].mean()),
                                           inferred_components=float(features['target_mask_diagnostic'][:,2].mean())),
                limitation='Synthetic mask IoU is diagnostic on generated images, not real-target localization accuracy.')
    U.dump(out/'verification.json',report);print(__import__('json').dumps(report))
if __name__=='__main__':main()
