"""User-requested interim test; never feeds checkpoint selection or training."""
from pathlib import Path
import argparse,copy,hashlib,json,sys
WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'src'))
from capri_controller.training.ds2 import ROOT, DS1_RUN, U, bootstrap


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--epoch',type=int,required=True);args=parser.parse_args()
    bootstrap()
    import torch
    import numpy as np
    from PIL import Image
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from capri_controller.training.ds2_data import transform
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20)
    path=ROOT/f'seed_{args.seed}'/f'epoch_{args.epoch}.pth'
    out=ROOT/f'interim_evaluation/seed_{args.seed}_epoch_{args.epoch}'
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    digest=U.sha(path);saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['epoch']==args.epoch and U.sha(path)==digest
    cfg=copy.deepcopy(saved['config']);decoder=build_post_process(cfg['PostProcess'],cfg['Global'])
    cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
    model=build_model(cfg['Architecture']);model.load_state_dict(saved['state_dict'],strict=True);model.cuda().eval()
    source=DS1_RUN/'evaluation_darkbay';records=U.read(source/'image_manifest.json');assert len(records)==900
    U.dump(out/'protocol.json',dict(checkpoint=str(path),sha256=digest,seed=args.seed,epoch=args.epoch,images=900,
        decoding='Same FP32 RGB bicubic32x128, autoregressive decoder, one refinement, batch64; no forced length or case correction',
        limitation='User-requested interim historical real-test diagnostic; not used for training changes or checkpoint selection',
        code_sha256=U.sha(Path(__file__))))
    tf=transform();rows=[]
    with torch.inference_mode():
        for start in range(0,len(records),64):
            chunk=records[start:start+64];images=[]
            for r in chunk:
                im=Image.open(r['image_path']).convert('RGB')
                assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
                images.append(tf(im))
            outputs=decoder(model(torch.stack(images).cuda()))
            for r,(text,conf) in zip(chunk,outputs):
                rows.append(dict(opaque_id=r['opaque_id'],prediction=text,confidence=float(conf) if np.isfinite(conf) else None))
    U.dump(out/'predictions.json',rows);frozen=U.sha(out/'predictions.json')
    U.dump(out/'prediction_freeze.json',dict(sha256=frozen,count=len(rows)))
    from rapidfuzz.distance import Levenshtein
    labels=U.read(source/'scoring_labels.json')
    assert {r['opaque_id'] for r in rows}==set(labels)
    correct=sum(r['prediction']==labels[r['opaque_id']] for r in rows)
    distance=sum(Levenshtein.distance(labels[r['opaque_id']],r['prediction']) for r in rows)
    val=U.read(ROOT/f'seed_{args.seed}/validation_history.json')
    validation=next(r for r in val if r['epoch']==args.epoch)
    result=dict(seed=args.seed,epoch=args.epoch,correct=correct,total=900,accuracy=100*correct/900,
        CER=100*distance/sum(map(len,labels.values())),synthetic_validation=validation)
    assert U.sha(out/'predictions.json')==frozen
    U.dump(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
