"""Frozen CRNN synthetic follow-up: distinguish padding edges from shrink sensitivity."""
from pathlib import Path
import sys,json
import numpy as np
import torch
import torch.nn.functional as F
BASE=Path(__file__).resolve().parent;PROJECT=Path('E:/pranav/Final sem/sourcefree_synth');sys.path.insert(0,str(PROJECT/'style_pipeline/phase3_reader_v2'))
from common import *
from models import Reader
from train import batch
root=PROJECT/'runs/capri_phase3_city_02';out=BASE/'phase3_layout_diagnostic_02';out.mkdir(exist_ok=False);torch.set_num_threads(2)
rows=[json.loads(l) for l in (root/'data/partitions.jsonl').open()];ids=np.array([r['index'] for r in rows if r['partition']=='test'][:1024]);labels=[rows[i]['label'] for i in ids];cache=np.load(root/'data/synthetic.npy',mmap_mode='r')
checkpoint=torch.load(root/'crnn_plain/best.pt',map_location='cpu',weights_only=False);digest=sha(root/'crnn_plain/best.pt');m=Reader('crnn',False).to('cuda').eval();m.load_state_dict(checkpoint['state_dict']);m.requires_grad_(False);results={}
for view in ['original','shrink_white_pad','shrink_replicated_edge_pad','resolution_only']:
    predictions=[]
    for start in range(0,len(ids),64):
        x=batch(cache,ids[start:start+64],'cuda')
        if view!='original':
            x=F.interpolate(x,size=(32,116),mode='bilinear',align_corners=False)
            if view=='shrink_white_pad':x=F.pad(x,(38,38),value=1.)
            elif view=='shrink_replicated_edge_pad':x=F.pad(x,(38,38,0,0),mode='replicate')
            else:x=F.interpolate(x,size=(32,192),mode='bilinear',align_corners=False)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p,_,_=m.predict(x)
        predictions.extend(p)
    results[view]=metrics(labels,predictions)
    with (out/(view+'.jsonl')).open('w') as f:
        for i,t,p in zip(ids,labels,predictions):f.write(json.dumps(dict(index=int(i),label=t,prediction=p))+'\n')
    print(view,json.dumps({k:results[view][k] for k in ['cer','exact_string_accuracy','decoded_count_accuracy']}),flush=True)
assert sha(root/'crnn_plain/best.pt')==digest
write(out/'results.json',dict(checkpoint_sha256=digest,samples=len(ids),source='Previously evaluated synthetic test first1024; now development diagnostic, not fresh unseen test',views=results,real_labels_used=False,training_or_selection_changed=False))
