"""Infer selected DS2 readers on validation, then reuse DS1's frozen-policy method."""
from pathlib import Path
import sys,copy,io,math,gc,hashlib
WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'src'))
from capri_controller.training.ds2 import ROOT,U,bootstrap

def main():
    bootstrap()
    import torch,lmdb
    import numpy as np
    from PIL import Image
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from capri_controller.training.ds2_data import transform
    torch.set_num_threads(4)
    out=ROOT/'synthetic_policy_inference';out.mkdir(exist_ok=True)
    valid=[r for r in U.read(ROOT/'curriculum_samples.json') if r['split']=='validation']
    assert len(valid)==5000
    metadata={r['index']:r for r in U.read(ROOT/'samples.json')}
    env=lmdb.open(str(ROOT/'all_lmdb'),readonly=True,lock=False,readahead=False)
    tf=transform();images=[]
    with env.begin() as txn:
        for r in valid:
            im=Image.open(io.BytesIO(txn.get(f"image-{r['index']:09d}".encode()))).convert('RGB')
            assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==metadata[r['index']]['pixel_sha256']
            images.append(tf(im))
    env.close();images=torch.stack(images)
    for run in U.read(ROOT/'completion.json')['seeds']:
        seed=run['seed'];p=out/f'seed_{seed}.json';receipt=p.with_suffix('.receipt.json')
        assert U.sha(run['checkpoint'])==run['sha256']
        if receipt.exists():
            assert U.sha(p)==U.read(receipt)['predictions_sha256']
            assert U.read(receipt)['checkpoint_sha256']==run['sha256']
            continue
        ck=torch.load(run['checkpoint'],map_location='cpu',weights_only=False)
        cfg=copy.deepcopy(ck['config']);decoder=build_post_process(cfg['PostProcess'],cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
        model=build_model(cfg['Architecture']);model.load_state_dict(ck['state_dict'],strict=True);model.cuda().eval()
        rows=[]
        with torch.inference_mode():
            for start in range(0,5000,64):
                outputs=model(images[start:start+64].cuda());assert torch.isfinite(outputs).all()
                for r,(text,conf) in zip(valid[start:start+64],decoder(outputs)):
                    rows.append(dict(index=r['index'],prediction=text,confidence=float(conf) if math.isfinite(conf) else None))
        U.dump(p,rows);U.dump(receipt,dict(predictions_sha256=U.sha(p),checkpoint_sha256=run['sha256'],epoch=run['epoch'],images=5000,decoding='FP32 batch64, original RGB bicubic32x128; same as real evaluation',code_sha256=U.sha(Path(__file__))))
        print(f'Seed {seed}: 5000 predictions frozen',flush=True)
        del model,ck;gc.collect();torch.cuda.empty_cache()
    import synthetic_shoot_policy
    synthetic_shoot_policy.main(model2=True)

if __name__=='__main__':main()
