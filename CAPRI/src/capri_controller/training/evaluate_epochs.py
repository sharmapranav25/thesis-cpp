"""Retrospective DS1 epoch sweep on the frozen historical DarkBay test set."""
import collections
import copy
import csv
import hashlib
import math
import re
import sys
import time
from pathlib import Path
from .. import runtime as U
from .prepare_parseq import RUN, OPENOCR


def main():
    out = RUN / 'epoch_evaluation_darkbay'
    if out.exists():
        raise FileExistsError('Preserve existing epoch sweep: ' + str(out))
    out.mkdir()
    sys.path[:0] = [str(U.WORK/'.runtime/parseq_packages'), str(OPENOCR)]
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    seeds = [48,123,456]
    epochs = list(range(2,21))
    source = RUN/'evaluation_darkbay'
    manifest = U.read(source/'image_manifest.json')
    assert len(manifest) == len({r['opaque_id'] for r in manifest}) == 900
    checkpoints = [dict(seed=s, epoch=e, path=str(RUN/f'seed_{s}'/f'epoch_{e}.pth')) for e in epochs for s in seeds]
    assert all(Path(r['path']).is_file() for r in checkpoints)
    U.dump(out/'protocol.json', dict(checkpoints=checkpoints, manifest_sha256=U.sha(source/'image_manifest.json'),
        images=900, epochs=epochs, seeds=seeds, epoch1='Not saved separately',
        preprocessing='Same FP32 RGB bicubic32x128 and normalization as original test; batch64',
        decoding='Saved autoregressive PARSeq with one refinement; no forced length/case/charset correction',
        ensemble='Whole-string majority per epoch, saved mean character confidence tie-break; exact confidence ties use seed48,123,456 order',
        label_access='All57 checkpoint and19 ensemble prediction files frozen before scoring labels are loaded',
        limitation='Retrospective test-set diagnostic requested after final-epoch scores were known. Peak epochs are descriptive, not independent validation or proof of overfitting. DS2 stays stopped.',
        code_sha256=U.sha(Path(__file__))))
    tf = T.Compose([T.Resize((32,128),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(.5,.5)])
    images=[]
    for r in manifest:
        im=Image.open(r['image_path']).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
        images.append(tf(im))
    x_all=torch.stack(images)
    frozen={}; predictions={}; model=None; base_cfg=None
    for i,record in enumerate(checkpoints):
        start=time.time();s=record['seed'];e=record['epoch'];path=Path(record['path'])
        digest=U.sha(path)
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        assert ckpt['epoch']==e
        cfg=copy.deepcopy(ckpt['config'])
        if model is None:
            decoder=build_post_process(cfg['PostProcess'],cfg['Global'])
            cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
            base_cfg=copy.deepcopy(cfg)
            model=build_model(cfg['Architecture']).cuda().eval()
        else:
            cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
            assert cfg['Architecture']==base_cfg['Architecture'] and cfg['PostProcess']==base_cfg['PostProcess']
        assert all(torch.isfinite(v).all() for v in ckpt['state_dict'].values() if v.is_floating_point())
        model.load_state_dict(ckpt['state_dict'],strict=True)
        del ckpt
        rows=[]
        with torch.inference_mode():
            for j in range(0,900,64):
                p=model(x_all[j:j+64].cuda())
                assert torch.isfinite(p).all()
                outputs=decoder(p)
                assert len(outputs)==len(manifest[j:j+64])
                for r,(text,conf) in zip(manifest[j:j+64],outputs):
                    rows.append(dict(opaque_id=r['opaque_id'],prediction=text,confidence=float(conf) if math.isfinite(conf) else None))
        name=f'epoch_{e:02d}_seed_{s}'
        U.dump(out/(name+'.json'),rows)
        frozen[name]=dict(prediction_sha256=U.sha(out/(name+'.json')),checkpoint_sha256=digest,seconds=time.time()-start)
        predictions[e,s]=rows
        if e==20:
            prior=U.read(source/f'seed_{s}_predictions.json')
            assert [(r['opaque_id'],r['prediction']) for r in rows]==[(r['opaque_id'],r['prediction']) for r in prior]
        U.dump(out/'progress.json',dict(completed=i+1,total=57,epoch=e,seed=s))
        print(f'{i+1}/57 frozen: epoch {e}, seed {s}, {time.time()-start:.1f}s',flush=True)
    for e in epochs:
        rows=[]
        for i,r in enumerate(manifest):
            candidates=[predictions[e,s][i] for s in seeds]
            text,count=collections.Counter(c['prediction'] for c in candidates).most_common(1)[0]
            if count==1:
                text=max(candidates,key=lambda c:c['confidence'] if c['confidence'] is not None else float('-inf'))['prediction']
            rows.append(dict(opaque_id=r['opaque_id'],prediction=text))
        name=f'epoch_{e:02d}_ensemble';U.dump(out/(name+'.json'),rows)
        frozen[name]=dict(prediction_sha256=U.sha(out/(name+'.json')))
        predictions[e,'ensemble']=rows
    U.dump(out/'prediction_freeze.json',frozen)
    from rapidfuzz.distance import Levenshtein
    labels=U.read(source/'scoring_labels.json')
    assert set(labels)=={r['opaque_id'] for r in manifest}
    results=[];summary=[]
    for e in epochs:
        item={'epoch':e}
        for s in seeds+['ensemble']:
            name=f'epoch_{e:02d}_'+('ensemble' if s=='ensemble' else f'seed_{s}')
            assert U.sha(out/(name+'.json'))==frozen[name]['prediction_sha256']
            rows=predictions[e,s]
            correct=sum(r['prediction']==labels[r['opaque_id']] for r in rows)
            distance=sum(Levenshtein.distance(labels[r['opaque_id']],r['prediction']) for r in rows)
            results.append(dict(epoch=e,seed=s,correct=correct,total=900,accuracy=100*correct/900,CER=100*distance/sum(map(len,labels.values()))))
            item[str(s)]=correct
        item['mean_accuracy']=sum(item[str(s)] for s in seeds)/27
        summary.append(item)
    assert summary[-1]['ensemble']==496
    best={str(s):max((r for r in results if r['seed']==s),key=lambda r:r['accuracy']) for s in seeds+['ensemble']}
    # Last logged smoothed training loss is context, not training accuracy.
    losses=[]
    pattern=re.compile(r'epoch: \[(\d+)/20\].*?loss: ([0-9.eE+-]+)')
    for s in seeds:
        values={}
        for line in (RUN/f'seed_{s}'/'train.log').read_text(encoding='utf8').splitlines():
            match=pattern.search(line)
            if match:values[int(match.group(1))]=float(match.group(2))
        losses.extend(dict(seed=s,epoch=e,last_logged_smoothed_training_loss=v) for e,v in sorted(values.items()))
    U.dump(out/'training_losses.json',losses)
    U.dump(out/'summary.json',dict(epochs=summary,best_observed=best,results=results,limitation='Historical real-test diagnostic; no independent epoch selection and no held-out synthetic learning curve'))
    with (out/'results.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,8),sharex=True)
    for s in seeds+['ensemble']:
        rows=[r for r in results if r['seed']==s]
        axes[0].plot([r['epoch'] for r in rows],[r['accuracy'] for r in rows], 'o-',label=str(s),linewidth=2 if s=='ensemble' else 1)
    axes[0].set(ylabel='Whole-CAPTCHA accuracy (%)',title='DS1: saved epochs on the same 900 real DarkBay images')
    axes[0].legend();axes[0].grid(alpha=.25)
    for s in seeds:
        rows=[r for r in losses if r['seed']==s and r['epoch']>=2]
        axes[1].semilogy([r['epoch'] for r in rows],[r['last_logged_smoothed_training_loss'] for r in rows], 'o-',label=str(s))
    axes[1].set(xlabel='Epoch',ylabel='Last logged training loss (log scale)')
    axes[1].set_xticks(epochs);axes[1].grid(alpha=.25);axes[1].legend()
    fig.tight_layout();fig.savefig(out/'epoch_accuracy.png',dpi=150)
    text='Epoch | Seed48 | Seed123 | Seed456 | Ensemble (correct/900)\n'
    text+='\n'.join(f"{r['epoch']:2d} | {r['48']:3d} | {r['123']:3d} | {r['456']:3d} | {r['ensemble']:3d}" for r in summary)
    (out/'results.txt').write_text(text,encoding='utf8')
    print(text,flush=True);print('BEST OBSERVED',best,flush=True)


if __name__=='__main__':main()
