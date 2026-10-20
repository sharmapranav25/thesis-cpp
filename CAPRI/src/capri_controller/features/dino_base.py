"""Frozen official DINO ViT-S/8: attention is saliency, not a text probability."""
from ..runtime import model_path
WEIGHTS_PATH=model_path('dino')
from ..vendor.dino.vision_transformer import vit_small
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from scipy.spatial.distance import cdist

WEIGHTS=dict(global_embedding=.20,focus_embedding=.35,context_embedding=.25,attention_layout=.10,region_colors=.10)

class DinoComparator:
    def __init__(self):
        torch.set_num_threads(4);self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=vit_small(patch_size=8).to(self.device).eval()
        self.model.load_state_dict(torch.load(WEIGHTS_PATH,map_location='cpu',weights_only=True),strict=True)
        self.model.requires_grad_(False);self.last_attention=None
        self.model.blocks[-1].attn.register_forward_hook(self.capture)
    def capture(self,module,args,output):self.last_attention=output[1][:,:,0,1:]
    def tensor(self,images):
        pixels=np.stack([np.asarray(Image.fromarray(im).convert('RGB').resize((256,64),Image.Resampling.BILINEAR)) for im in images])
        x=torch.from_numpy(pixels.transpose(0,3,1,2).copy()).to(self.device).float()/255
        return x,(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    @torch.inference_mode()
    def extract(self,images):
        result={k:[] for k in [*WEIGHTS,'focus','heads']}
        for start in range(0,len(images),16):
            pixels,x=self.tensor(images[start:start+16]);tokens=self.model.get_intermediate_layers(x,n=1)[0]
            attention=self.last_attention;attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-9)
            lo=torch.quantile(attention,.10,dim=-1,keepdim=True);hi=torch.quantile(attention,.90,dim=-1,keepdim=True)
            focus=((attention-lo)/(hi-lo).clamp_min(1e-8)).clamp(0,1);context=1-focus
            def pooled(w):
                w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
                regions=torch.einsum('bhn,bnd->bhd',w,tokens[:,1:])
                return F.normalize(F.normalize(regions,dim=-1).flatten(1),dim=-1)
            result['global_embedding'].append(F.normalize(tokens[:,0],dim=-1).cpu().numpy())
            result['focus_embedding'].append(pooled(focus).cpu().numpy());result['context_embedding'].append(pooled(context).cpu().numpy())
            grid=focus.reshape(-1,6,8,32)
            result['attention_layout'].append(F.adaptive_avg_pool2d(grid,(4,16)).flatten(1).cpu().numpy())
            region=F.interpolate(grid.mean(1,keepdim=True),size=(64,256),mode='bilinear',align_corners=False)
            cols=[]
            for w in [region,1-region]:
                for c in range(3):
                    v=pixels[:,c:c+1]
                    hist=torch.stack([((v>=i/8)&(v<(i+1)/8 if i<7 else v<=1)).float().mul(w).sum((1,2,3)) for i in range(8)],1)
                    cols.append(hist.cumsum(1)/w.sum((1,2,3))[:,None].clamp_min(1e-8))
            result['region_colors'].append(torch.cat(cols,1).cpu().numpy())
            result['focus'].append(region[:,0].cpu().numpy());result['heads'].append(grid.cpu().numpy())
        return {k:np.concatenate(v) for k,v in result.items()}

def energy(a,b):
    return max(0.,float((2*cdist(a,b).mean()-cdist(a,a).mean()-cdist(b,b).mean())/np.sqrt(a.shape[1])))

class Objective:
    def __init__(self,target,baseline):
        self.std={k:np.maximum(target[k].std(0),.025) for k in WEIGHTS}
        self.scale={k:max(.05,energy(baseline[k]/self.std[k],target[k]/self.std[k])) for k in WEIGHTS}
    def score(self,features,target,rows):
        components={k:energy(features[k]/self.std[k],target[k]/self.std[k])/self.scale[k] for k in WEIGHTS}
        appearance=sum(WEIGHTS[k]*components[k] for k in WEIGHTS)
        complete=float(np.mean([r['complete'] for r in rows]));usable=float(np.mean([r['usable'] for r in rows]))
        return dict(components=components,appearance=appearance,objective=appearance+4*(1-complete)+2*(1-usable),complete=complete,usable=usable)
    def vector(self,f):return np.concatenate([f[k].mean(0)/self.std[k]*WEIGHTS[k]**.5 for k in WEIGHTS])

def palettes(images,features):
    pixels=np.stack([np.asarray(Image.fromarray(im).resize((256,64),Image.Resampling.BILINEAR)) for im in images]);focus=features['focus']
    def reduce(mask):
        values=pixels[mask][::7]
        if len(values)==0:values=pixels.reshape(-1,3)[::7]
        im=Image.fromarray(values.astype(np.uint8).reshape(1,-1,3)).quantize(colors=6,method=Image.Quantize.MEDIANCUT)
        table=im.getpalette();return [table[3*i:3*i+3] for _,i in sorted(im.getcolors(),reverse=True)]
    return dict(text=reduce(focus>.65),background=reduce(focus<.35),effect=reduce(np.ones_like(focus,dtype=bool)))
