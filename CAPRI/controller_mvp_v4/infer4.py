"""Freeze all six readers before target inference; no answer annotations loaded here."""
from setup4 import ROOT,U,FAMILIES
import numpy as np
from PIL import Image
import torch
from datetime import datetime,timezone
from solver4 import Solver
from train4 import infer

def main():
    frozen=U.read(ROOT/'solvers/frozen_models.json');assert len(frozen)==6
    assert len({r['steps'] for r in frozen.values()})==1
    dest=ROOT/'evaluation';dest.mkdir(exist_ok=False);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    U.dump(dest/'boundary.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),models_frozen_sha256=U.sha(ROOT/'solvers/frozen_models.json'),target_answers_read=False,models=frozen))
    for family in FAMILIES:
        refs=U.read(ROOT/'evaluation_pixels'/f'{family}.json');images=[]
        for r in refs['rows']:
            pixels=np.asarray(Image.open(r['source']).convert('RGB'))
            import hashlib
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==r['pixel_sha256']
            images.append(np.asarray(Image.fromarray(pixels).resize((192,48),Image.Resampling.BILINEAR)))
        images=np.stack(images);predictions={}
        for name,entry in frozen.items():
            assert U.sha(entry['checkpoint'])==entry['sha256']
            model=Solver().to(device);model.load_state_dict(torch.load(entry['checkpoint'],map_location=device,weights_only=True));predictions[name]=infer(model,images,device)
        U.dump(dest/(family+'_predictions.json'),dict(pixel_sha256=[r['pixel_sha256'] for r in refs['rows']],predictions=predictions,status=refs['status'],answers_used=False))
        print(family,len(images),'predicted',flush=True)
    U.dump(dest/'inference_complete.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),models_frozen_sha256=U.sha(ROOT/'solvers/frozen_models.json'),answers_read=False,
        prediction_sha256={p.name:U.sha(p) for p in dest.glob('*_predictions.json')}))

if __name__=='__main__':main()
