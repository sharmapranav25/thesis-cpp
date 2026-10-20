"""Two genuine frozen visual backbones; no substitutes when weights are missing."""
from pathlib import Path
import sys,hashlib,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'controller_mvp_v7_dino'))
from setup7 import U,WORK,FAMILIES
from perception import extra
from perception3 import ink_shape
from dino5 import energy
import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi
from PIL import Image

WEIGHTS=dict(embedding=.25,text_color=.12,background_color=.12,glyph_shape=.13,ink_shape=.10,layout=.10,background_structure=.18)

def pixels(im):return np.asarray(Image.fromarray(im).convert('RGB').resize((256,64),Image.Resampling.BILINEAR))

def color(im,mask):
    result=[]
    for c in range(3):
        h=np.histogram(im[:,:,c],bins=16,range=(0,256),weights=mask)[0]
        result.extend(h.cumsum()/max(1e-8,h.sum()))
    return np.asarray(result,np.float32)

def describe(im,mask,embedding):
    glyph,layout,bg,_=extra(im,mask)
    return dict(embedding=embedding,text_color=color(im,mask),background_color=color(im,1-mask),
        glyph_shape=glyph,ink_shape=ink_shape(im,mask),layout=layout,background_structure=bg,
        mask=mask,contrast=np.asarray([np.std(im.astype(float)/255)]))

class SAM:
    def __init__(self,checkpoint=None):
        sys.path.insert(0,str(ROOT/'vendor/segment-anything-main'))
        from segment_anything import sam_model_registry,SamAutomaticMaskGenerator
        torch.set_num_threads(4);self.device='cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint=Path(checkpoint or ROOT/'weights/sam_vit_b_01ec64.pth')
        if not checkpoint.is_file():raise FileNotFoundError('Official SAM ViT-B checkpoint required: '+str(checkpoint))
        self.model=sam_model_registry['vit_b'](checkpoint=str(checkpoint)).to(self.device).eval().requires_grad_(False)
        # Rectangular point grid generated from pixels' canvas, never target boxes.
        points=np.stack(np.meshgrid(np.linspace(.04,.96,12),np.linspace(.12,.88,4)),axis=-1).reshape(-1,2)
        self.generator=SamAutomaticMaskGenerator(self.model,points_per_side=None,point_grids=[points],points_per_batch=48,
            pred_iou_thresh=.75,stability_score_thresh=.85,crop_n_layers=0,min_mask_region_area=0)
        self.name='sam';self.weight_sha=U.sha(checkpoint);self.cache={}
    @torch.inference_mode()
    def one(self,im):
        masks=self.generator.generate(im);m=np.zeros((64,256),np.float32);kept=[]
        # SAM proposes objects, not text. This generic geometry filter is a heuristic.
        for item in masks:
            x,y,w,h=item['bbox'];area=item['area']
            if 7<=h<=63 and 2<=w<=128 and 8<=area<=.30*64*256 and w/max(h,1)<5:
                m=np.maximum(m,item['segmentation'].astype(np.float32));kept.append(item)
        # AMG resets its predictor; encode once more only if no embedding is cached.
        # Store image-encoder output during the existing AMG pass via the hook below.
        z=self.last_embedding
        emb=F.adaptive_avg_pool2d(z[:,:,:16,:],(2,8)).flatten().cpu().numpy()
        result=describe(im,m,emb);result['proposal_count']=np.asarray([len(masks),len(kept)])
        return result
    def extract(self,images):
        if not hasattr(self,'hook'):
            self.hook=self.model.image_encoder.register_forward_hook(lambda mod,args,out:setattr(self,'last_embedding',out.detach()))
        rows=[]
        for image in images:
            im=pixels(image);key=hashlib.sha256(im.tobytes()).hexdigest()
            if key not in self.cache:self.cache[key]=self.one(im)
            rows.append(self.cache[key])
        return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}

class DINOv3:
    def __init__(self,checkpoint=None,architecture_only=False):
        sys.path.insert(0,str(ROOT/'vendor/dinov3-main'))
        torch.set_num_threads(4);self.device='cuda' if torch.cuda.is_available() else 'cpu'
        from dinov3.hub.backbones import dinov3_vits16
        checkpoint=Path(checkpoint or ROOT/'weights/dinov3_vits16.pth')
        if not checkpoint.is_file() and not architecture_only:
            raise FileNotFoundError('DINOv3 needs an approved official ViT-S/16 .pth checkpoint. No random-weight or older-DINO fallback is allowed.')
        self.model=dinov3_vits16(pretrained=False)
        if not architecture_only:self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        self.model=self.model.to(self.device).eval().requires_grad_(False)
        self.name='dinov3';self.weight_sha='UNTRAINED_ARCHITECTURE_TEST_ONLY' if architecture_only else U.sha(checkpoint);self.cache={}
    @torch.inference_mode()
    def extract(self,images):
        rows=[]
        for image in images:
            im=pixels(image);key=hashlib.sha256(im.tobytes()).hexdigest()
            if key not in self.cache:
                x=torch.tensor(im.transpose(2,0,1).copy(),device=self.device)[None].float()/255
                x=F.interpolate(x,size=(128,512),mode='bilinear',align_corners=False)
                x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
                out=self.model.forward_features(x);tokens=F.normalize(out['x_norm_patchtokens'][0],dim=-1).float().cpu().numpy()
                grid=tokens.reshape(8,32,-1);border=np.zeros((8,32),bool);border[[0,-1],:]=True;border[:,[0,-1]]=True
                center=grid[border].mean(0);dist=np.linalg.norm(tokens-center,axis=1)
                centers=np.stack([center,tokens[dist.argmax()]])
                for _ in range(8):
                    labels=((tokens[:,None]-centers[None])**2).sum(-1).argmin(1)
                    for j in range(2):
                        if (labels==j).any():centers[j]=tokens[labels==j].mean(0)
                labels=labels.reshape(8,32);background=int(np.bincount(labels[border],minlength=2).argmax())
                mask=np.asarray(Image.fromarray(((labels!=background)*255).astype(np.uint8)).resize((256,64),Image.Resampling.BILINEAR),np.float32)/255
                z=torch.tensor(grid.transpose(2,0,1))[None];emb=F.adaptive_avg_pool2d(z,(2,8)).flatten().numpy()
                self.cache[key]=describe(im,mask,emb)
            rows.append(self.cache[key])
        return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}

class Objective:
    def __init__(self,target,baseline):
        self.std={k:np.maximum(np.concatenate([target[k],baseline[k]]).std(0),.025) for k in WEIGHTS}
        self.scale={k:max(.05,energy(baseline[k]/self.std[k],target[k]/self.std[k])) for k in WEIGHTS}
    def score(self,features,target,rows):
        components={k:energy(features[k]/self.std[k],target[k]/self.std[k])/self.scale[k] for k in WEIGHTS}
        appearance=sum(WEIGHTS[k]*components[k] for k in WEIGHTS)
        complete=float(np.mean([r['complete'] for r in rows]));usable=float(np.mean([r['usable'] for r in rows]))
        return dict(appearance=appearance,objective=appearance+2*(1-complete)+.25*(1-usable),complete=complete,usable=usable,components=components)
