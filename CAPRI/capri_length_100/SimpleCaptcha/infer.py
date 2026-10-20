"""Raw frozen-hybrid length predictions on 100 SimpleCaptcha fit images; no cleanup."""
from pathlib import Path
import sys,hashlib,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'controller_mvp_20261004'))
import common as U
import torch
import numpy as np
from PIL import Image
from feature_bank.iter2.model import Iter2Model
from feature_bank.iter2.data import MARGIN

def main():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    refs=U.read(ROOT.parents[1]/'controller_mvp_v7_dino/SimpleCaptcha/target_splits.json')['fit'][:100]
    assert len(refs)==100 and len({r['pixel_sha256'] for r in refs})==100
    model=Iter2Model().to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(U.HYBRID,map_location=device,weights_only=True),strict=True)
    rows=[];start=time.time()
    with torch.inference_mode():
        for i,ref in enumerate(refs):
            with Image.open(ref['source']) as im:im=im.convert('RGB');a=np.asarray(im)
            assert hashlib.sha256(a.tobytes()).hexdigest()==ref['pixel_sha256']
            h,w=a.shape[:2];rw=max(8,min(960,round(w*96/h)));cols=(rw+7)//8;wp=((cols*8+MARGIN+7)//8)*8
            a=np.asarray(Image.fromarray(a).resize((rw,96),Image.Resampling.LANCZOS))
            a=np.pad(a,((0,0),(0,wp-rw),(0,0)),mode='edge')
            out=model(torch.tensor(a.transpose(2,0,1).copy(),device=device).float()[None]/255,torch.tensor([cols],device=device))
            logits=out['length'][0].float().cpu();probs=logits.softmax(0);pred=int(logits.argmax())+2
            rows.append(dict(index=i,source=ref['source'],pixel_sha256=ref['pixel_sha256'],predicted_length=pred,
                probabilities={str(k+2):float(v) for k,v in enumerate(probs)},logits=logits.tolist()))
    counts={str(k):sum(r['predicted_length']==k for r in rows) for k in range(2,9)};winner=max(counts,key=counts.get)
    U.dump(ROOT/'predictions.json',rows)
    summary=dict(images=100,counts=counts,most_common_length=int(winner),most_common_count=counts[winner],
        clears_proposed_60_percent_threshold=counts[winner]>=60,checkpoint=str(U.HYBRID),checkpoint_sha256=U.sha(U.HYBRID),
        selection='First100 distinct decoded-pixel images from existing SimpleCaptcha fit split. Selection/test images excluded.',
        preprocessing='Existing controller hybrid inference: RGB, height96, aspect-preserving width capped960, edge-padding, /255; length argmax+2.',
        raw_predictions=True,target_labels_used=False,ctc_decoded=False,seconds=time.time()-start)
    U.dump(ROOT/'summary.json',summary);print(summary)

if __name__=='__main__':main()
