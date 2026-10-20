"""Supplementary labeled CryptBB evaluation, added after the main answer check.

The preregistered 24 diagnostic images have no matching answers. This separate
evaluation uses all eligible native-size labeled images with the same frozen
readers; it neither replaces nor changes the primary test.
"""
from setup4 import ROOT,V3,V1,U,ARMS,CHARSET
from pathlib import Path
from datetime import datetime,timezone
from load_answers4 import pixel_hash
from prepare_demo import hashes
from solver4 import Solver,metrics,edit
from train4 import infer
import csv
import numpy as np
from PIL import Image
import torch

def main():
    frozen=U.read(ROOT/'solvers/frozen_models.json');boundary=U.read(ROOT/'evaluation/inference_complete.json')
    assert U.sha(ROOT/'solvers/frozen_models.json')==boundary['models_frozen_sha256']
    dest=ROOT/'cryptbb_supplement';dest.mkdir(exist_ok=False)
    excluded=set()
    split_paths=[V3/'CryptBB_run_01/target_splits.json',V1/'CryptBB_run_01/target_splits.json',U.WORK/'feature_ablation_20261004/run_03/CryptBB/target_splits.json']
    for path in split_paths:
        split=U.read(path);excluded.update(hashes(split));excluded.update(r['pixel_sha256'] for r in split.get('fresh',[]))
    folder=Path('E:/pranav/model 1/data/eval');csvpath=folder/'labels.csv';rows=[];seen=set();skipped={'size':0,'known_controller_pixel':0,'duplicate':0,'missing':0}
    with csvpath.open(encoding='utf-8-sig',newline='') as handle:
        for r in csv.DictReader(handle):
            p=folder/r['filename']
            if not p.exists():skipped['missing']+=1;continue
            if Image.open(p).size!=(200,60):skipped['size']+=1;continue
            h=pixel_hash(p)
            if h in excluded:skipped['known_controller_pixel']+=1;continue
            if h in seen:skipped['duplicate']+=1;continue
            assert r['label'] and all(c in CHARSET for c in r['label'])
            seen.add(h);rows.append(dict(source=str(p),pixel_sha256=h,label=r['label']))
    assert rows
    U.dump(dest/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='Supplementary, added after main answer evaluation; not preregistered primary.',
        selection='All CSV-labeled native 200x60 images, exact-pixel deduplicated and excluding known controller split hashes; no selection on predictions or correctness.',
        csv=str(csvpath),csv_sha256=U.sha(csvpath),models_frozen_sha256=boundary['models_frozen_sha256'],skipped=skipped,images=len(rows),models_or_controls_changed=False))
    U.dump(dest/'images_and_answers.json',rows)
    images=np.stack([np.asarray(Image.open(r['source']).convert('RGB').resize((192,48),Image.Resampling.BILINEAR)) for r in rows]);labels=[r['label'] for r in rows]
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';predictions={};scores={};correct={};errors={}
    for arm in ARMS:
        each=[];corr=[];err=[]
        for seed in [43103,43109]:
            name=f'{arm}_{seed}';entry=frozen[name];assert U.sha(entry['checkpoint'])==entry['sha256']
            model=Solver().to(device);model.load_state_dict(torch.load(entry['checkpoint'],map_location=device,weights_only=True));p=infer(model,images,device);predictions[name]=p
            each.append(dict(seed=seed,**metrics(labels,p)));corr.append([a==b for a,b in zip(labels,p)]);err.append([edit(a,b) for a,b in zip(labels,p)])
        scores[arm]=dict(seeds=each,mean_exact=float(np.mean([r['exact'] for r in each])),mean_cer=float(np.mean([r['cer'] for r in each])))
        correct[arm]=np.mean(corr,axis=0);errors[arm]=np.mean(err,axis=0)
    paired={};rng=np.random.default_rng(45007);lengths=np.asarray(list(map(len,labels)))
    for baseline in ['hybrid','single']:
        exact=[];cer=[]
        for _ in range(2000):
            ix=rng.integers(0,len(rows),len(rows));exact.append(float((correct['diverse'][ix]-correct[baseline][ix]).mean()));cer.append(float((errors[baseline][ix]-errors['diverse'][ix]).sum()/lengths[ix].sum()))
        paired[baseline]=dict(exact_gain=float((correct['diverse']-correct[baseline]).mean()),exact_gain_95ci=np.quantile(exact,[.025,.975]).tolist(),cer_reduction=float((errors[baseline]-errors['diverse']).sum()/lengths.sum()),cer_reduction_95ci=np.quantile(cer,[.025,.975]).tolist())
    U.dump(dest/'predictions.json',predictions)
    result=dict(images=len(rows),scores=scores,diverse_vs=paired,models_frozen_sha256=boundary['models_frozen_sha256'],supplementary=True,predictions_sha256=U.sha(dest/'predictions.json'))
    U.dump(dest/'results.json',result);print(result)

if __name__=='__main__':main()
