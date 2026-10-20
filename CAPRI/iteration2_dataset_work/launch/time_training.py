"""Short disposable throughput benchmark; no saved model or corpus changes."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from feature_bank.iter2 import data as D, losses as L, metrics as M
from feature_bank.iter2.model import Iter2Model
from feature_bank.iter2.train import EMA, to_dev

if __name__ == '__main__':
    root=Path(r'E:\pranav\Final sem\sourcefree_synth\runs\iter2_baseline_15_20261003_02')
    cfg=yaml.safe_load(Path(r'E:\pranav\Final sem\sourcefree_synth\configs\iter2_baseline.yaml').read_text())
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    sets=D.open_split(root,'steering')
    items,widths=D.items_of(sets)
    batches=D.width_batches(items,widths,128,np.random.default_rng(1))
    model=Iter2Model(**cfg['model'],ctc_frames=1).cuda()
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4)
    ema=EMA(model,.999)
    weights=L.FixedWeights().cuda()
    elapsed=[]
    sizes=[]
    for k,cb in enumerate(D.prefetch(batches)):
        torch.cuda.synchronize()
        begin=time.perf_counter()
        b=to_dev(cb,torch.device('cuda'))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(b['image'],b['cols'])
        groups,_=L.compute(out,b)
        loss=weights(groups)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        ema.update(model)
        torch.cuda.synchronize()
        if k>=3:
            elapsed.append(time.perf_counter()-begin)
            sizes.append(len(b['image']))
        if len(elapsed)>=12:
            break
    begin=time.perf_counter()
    M.predict(model,sets,torch.device('cuda'))
    torch.cuda.synchronize()
    result=dict(measured_batches=len(elapsed),training_images=sum(sizes),seconds=sum(elapsed),
                images_per_second=sum(sizes)/sum(elapsed),mean_batch_size=float(np.mean(sizes)),
                evaluation_2000_seconds=time.perf_counter()-begin,
                peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
    print(json.dumps(result),flush=True)
