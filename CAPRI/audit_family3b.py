from pathlib import Path
import sys,json
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
out=Path(__file__).parent/'family_phase3b_01';results=json.loads((out/'results.json').read_text());assert len(results)==8
torch.set_num_threads(4);cache=Path.home()/'.cache/torch/hub';sys.path.insert(0,str(cache/'facebookresearch_dino_main'));import vision_transformer as vit
model=vit.vit_small(patch_size=8).cuda().eval();model.load_state_dict(torch.load(cache/'checkpoints/dino_deitsmall8_pretrain.pth',map_location='cpu',weights_only=True))
def embed(images):
    x=torch.tensor(np.stack([np.asarray(im.convert('RGB').resize((224,72)),dtype=np.float32).transpose(2,0,1)/255 for im in images]),device='cuda');x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.no_grad():return F.normalize(model(x),dim=1).cpu().numpy().mean(0)
def dist(a,b):return float(1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
audit={}
for name,r in results.items():
    folder=out/name;splits=json.loads((folder/'splits.json').read_text());groups=[set(row['pixel_sha256'] for row in rows) for rows in splits.values()];assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    allowed=set(row['path'] for row in splits['fit']);records=json.loads((folder/'fit_synthetic_records.json').read_text());assert all(row['reference'] in allowed for row in records)
    before=[Image.open(folder/f'{i:02d}_before.png').convert('RGB') for i in range(r['real_report'])];after=[Image.open(folder/f'{i:02d}_after.png').convert('RGB') for i in range(r['real_report'])];real=[Image.open(row['path']).convert('RGB') for row in splits['report']];reference=embed(real)
    audit[name]=dict(dino_before=dist(embed(before),reference),dino_after=dist(embed(after),reference),mean_rgb_shift=float(np.mean([np.abs(np.asarray(a,dtype=float).mean((0,1))-np.asarray(b,dtype=float).mean((0,1))).mean() for a,b in zip(after,before)])),family_fit_membership_verified=True,partition_hash_disjoint=True)
assert len(set(r['checkpoint_sha256'] for r in results.values()))==8
(out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
