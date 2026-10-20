"""One discarded gradient check. No optimizer step or saved reader weights."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/parseq_packages'),'E:/pranav/Final sem/Models/level 1/upstream/OpenOCR']
import torch
from tools.engine.config import Config
from tools.engine.trainer import Trainer

def main():
    run=ROOT/'runs/parseq_darkbay_100k_20261005';cfg=Config(str(run/'seed_48.yml'))
    cfg.cfg['Global']['output_dir']=str(run/'preflight_no_updates')
    torch.set_num_threads(8);trainer=Trainer(cfg,mode='train',task='rec')
    assert trainer.valid_dataloader is None
    batch=[v.to(trainer.device) for v in next(iter(trainer.train_dataloader))]
    start=time.time()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pred=trainer.model(batch[0],data=batch[1:]);loss=trainer.loss_class(pred,batch)['loss']
    assert torch.isfinite(loss)
    loss.backward();torch.cuda.synchronize()
    result=dict(dataset_images=len(trainer.train_dataloader.dataset),batch_shape=list(batch[0].shape),loss=float(loss),
        forward_backward_seconds=time.time()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,optimizer_steps=0,training_started=False,
        note='Readiness check only; random weights discarded, not used as initialization for requested run.')
    (run/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
