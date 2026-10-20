"""Full training-batch precision/memory test on synthetic data only."""
from pathlib import Path
import json,time
import numpy as np
import torch
from common import *
from models import Reader
from train import batch
torch.set_num_threads(4)
assert torch.cuda.is_bf16_supported()
data=Path('C:/Users/pranav/Documents/ChatGPT/thesis/phase3_city_data_01')
rows=[json.loads(l) for l in (data/'partitions.jsonl').open()];ids=np.array([r['index'] for r in rows if r['partition']=='train'][:128]);labels=[rows[i]['label'] for i in ids];cache=np.load(data/'synthetic.npy',mmap_mode='r')
for name in ['crnn','svtr','parseq']:
    torch.manual_seed(731);m=Reader(name,True).to('cuda').train();opt=torch.optim.AdamW(m.parameters(),lr=.0003);start=time.time()
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=m.loss(batch(cache,ids,'cuda'),labels)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        print(name,step,'loss',float(loss),'gradient_norm',float(norm),flush=True)
    print(name,'3 steps seconds',time.time()-start,'peak_memory',torch.cuda.max_memory_allocated(),flush=True)
    del m,opt;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
