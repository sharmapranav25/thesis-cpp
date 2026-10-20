"""Post-training real DarkBay evaluation, prediction ensemble and shoot rate."""
import collections
import copy
import gc
import hashlib
import html
import json
import math
from .ds2 import ROOT, DS1_RUN, SEEDS, U, bootstrap


def main():
    bootstrap()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from rapidfuzz.distance import Levenshtein
    from .ds2_data import transform
    torch.set_num_threads(8)
    out=ROOT/'evaluation_darkbay'
    if out.exists():raise FileExistsError('Preserve evaluation: '+str(out))
    out.mkdir()
    manifest=U.read(DS1_RUN/'evaluation_darkbay/image_manifest.json')
    assert len(manifest)==900
    completed=U.read(ROOT/'completion.json')
    assert completed['all_complete']
    U.dump(out/'protocol.json',dict(checkpoints=completed['seeds'],images=900,
        selection='Best synthetic validation checkpoint for each seed; all models finished20epochs',
        decoding='FP32, normal saved autoregressive decoder, one refinement, no length forcing or case changes',
        metric='Case-sensitive exact whole-string accuracy and CER',
        manifest_sha256=U.sha(DS1_RUN/'evaluation_darkbay/image_manifest.json'),
        limitation='Historical DarkBay set already examined for DS1; this is not untouched evaluation'))
    tf=transform(); tensors=[]
    for r in manifest:
        im=Image.open(r['image_path']).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
        tensors.append(tf(im))
    x_all=torch.stack(tensors);predictions={};frozen={}
    for run in completed['seeds']:
        seed=run['seed'];assert U.sha(run['checkpoint'])==run['sha256']
        checkpoint=torch.load(run['checkpoint'],map_location='cpu',weights_only=False)
        cfg=copy.deepcopy(checkpoint['config'])
        decoder=build_post_process(cfg['PostProcess'],cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
        model=build_model(cfg['Architecture']);model.load_state_dict(checkpoint['state_dict'],strict=True);model.cuda().eval()
        rows=[]
        with torch.inference_mode():
            for start in range(0,900,64):
                p=model(x_all[start:start+64].cuda());assert torch.isfinite(p).all()
                decoded=decoder(p);probs=p.cpu().numpy()
                for r,pp,(text,conf) in zip(manifest[start:start+64],probs,decoded):
                    cp=[]
                    for index,prob in zip(pp.argmax(-1),pp.max(-1)):
                        char=decoder.character[int(index)]
                        if char==decoder.EOS:break
                        if char not in (decoder.BOS,decoder.PAD):cp.append(float(prob))
                    assert len(cp)==len(text)
                    rows.append(dict(opaque_id=r['opaque_id'],prediction=text,confidence=float(conf) if math.isfinite(conf) else None,joint_confidence=float(np.prod(cp)) if cp else 0.0))
        name=f'seed_{seed}';predictions[name]=rows
        U.dump(out/(name+'_predictions.json'),rows);frozen[name]=U.sha(out/(name+'_predictions.json'))
        del model,checkpoint;gc.collect();torch.cuda.empty_cache()
    ensemble=[]
    for i,r in enumerate(manifest):
        candidates=[predictions[f'seed_{s}'][i] for s in SEEDS]
        text,count=collections.Counter(c['prediction'] for c in candidates).most_common(1)[0]
        if count==1:text=max(candidates,key=lambda c:c['confidence'] if c['confidence'] is not None else -1)['prediction']
        joint=sum(c['joint_confidence'] for c in candidates if c['prediction']==text)/3
        ensemble.append(dict(opaque_id=r['opaque_id'],prediction=text,joint_confidence=joint))
    predictions['ensemble']=ensemble
    U.dump(out/'ensemble_predictions.json',ensemble);frozen['ensemble']=U.sha(out/'ensemble_predictions.json')
    U.dump(out/'prediction_freeze.json',frozen)
    labels=U.read(DS1_RUN/'evaluation_darkbay/scoring_labels.json')
    assert set(labels)=={r['opaque_id'] for r in manifest}
    results={};shoot={}
    for name,rows in predictions.items():
        assert U.sha(out/(name+'_predictions.json'))==frozen[name]
        correct=sum(r['prediction']==labels[r['opaque_id']] for r in rows)
        distance=sum(Levenshtein.distance(labels[r['opaque_id']],r['prediction']) for r in rows)
        results[name]=dict(correct=correct,total=900,accuracy=100*correct/900,CER=100*distance/sum(map(len,labels.values())))
        shoot[name]=[]
        for t in [.99,.95,.90,.85,.8,.7,.6,.5,.4,.3,.2,.1,0]:
            attempted=[r for r in rows if r['joint_confidence']>=t]
            good=sum(r['prediction']==labels[r['opaque_id']] for r in attempted)
            shoot[name].append(dict(threshold=t,attempts=len(attempted),correct=good,accuracy=100*good/len(attempted) if attempted else None,shoot_rate=100*len(attempted)/900,ecps=900/good if good else None))
    U.dump(out/'shoot_rate.json',shoot)
    summary=dict(dataset='Darkbay_ds2_noFont-intel',results=results,ds1_epoch20_ensemble_correct=496,
        difference_in_correct=results['ensemble']['correct']-496,
        limitation='Different synthetic validation, augmentation and checkpoint selection from DS1; cannot attribute difference solely to curriculum')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    sheet=Image.new('RGB',(840,810),'white');draw=ImageDraw.Draw(sheet)
    for x,title in [(10,'Original'),(240,'Answer'),(400,'DS1 ensemble'),(620,'DS2 ensemble')]:draw.text((x,10),title,font=font,fill='black')
    old={r['opaque_id']:r['prediction'] for r in U.read(DS1_RUN/'prediction_ensemble/predictions.json')}
    for i,r in enumerate(manifest[:12]):
        y=55+i*60;oid=r['opaque_id'];sheet.paste(Image.open(r['image_path']).convert('RGB'),(10,y))
        draw.text((240,y+10),labels[oid],font=font,fill='black')
        for x,text in [(400,old[oid]),(620,ensemble[i]['prediction'])]:draw.text((x,y+10),text or '(empty)',font=font,fill='green' if text==labels[oid] else 'red')
    sheet.save(out/'examples.png')
    page='<!doctype html><meta charset="utf-8"><title>DarkBay DS2 results</title><style>body{font:18px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Darkbay_ds2_noFont-intel</h1><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre><p>First12 images in fixed hash order.</p><img src="examples.png"><h2>Ensemble shoot rate</h2><pre>'+html.escape(json.dumps(shoot['ensemble'],indent=2))+'</pre>'
    (out/'report.html').write_text(page,encoding='utf8')
    U.dump(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
