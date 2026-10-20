from .. import runtime as U
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from .base import descriptors,energy
from .regions import Perception as V2Perception,extra,Objective as V2Objective
from ..networks.mask import MaskNet

def ink_shape(im,m):
    """OCR-free topology, stroke survival and local contrast, across thresholds."""
    gray=im.mean(2)/255.;features=[]
    for threshold in [.35,.5,.65]:
        mask=m>threshold;area=max(1,mask.sum());filled=ndi.binary_fill_holes(mask)
        holes=filled&~mask;labels,n=ndi.label(holes);sizes=np.bincount(labels.ravel())[1:]
        dist=ndi.distance_transform_edt(mask)
        features.extend([holes.sum()/max(1,filled.sum()),min(20,int((sizes>=3).sum()))/20])
        features.extend([ndi.binary_erosion(mask,iterations=i).sum()/area for i in [1,2,3]])
        features.extend(np.quantile(dist[mask],[.1,.5,.9])/16 if mask.any() else [0,0,0])
    local=ndi.gaussian_filter(gray,2);contrast=abs(gray-local)
    gx=ndi.sobel(gray,axis=1)/8;gy=ndi.sobel(gray,axis=0)/8;mag=np.hypot(gx,gy)
    for values in [contrast,mag]:
        hist=np.histogram(values,bins=16,range=(0,.5),weights=m)[0];features.extend(np.cumsum(hist)/max(1,hist.sum()))
    return np.asarray(features)

class Perception(V2Perception):
    def __init__(self):
        super().__init__();self.segmenter=MaskNet().to(self.device).eval()
        self.segmenter.load_state_dict(torch.load(U.model_path('text_mask'),map_location=self.device,weights_only=False)['model']);self.segmenter.requires_grad_(False)
    @torch.inference_mode()
    def extract_pair(self,images):
        old=super().extract(images)
        pixels=np.stack([np.asarray(Image.fromarray(im).resize((256,64),Image.Resampling.BILINEAR)) for im in images]);masks=[]
        for i in range(0,len(pixels),32):
            x=torch.tensor(pixels[i:i+32].transpose(0,3,1,2).copy(),device=self.device).float()/255
            masks.extend(self.segmenter(x).sigmoid()[:,0].cpu().numpy())
        masks=np.asarray(masks);new=descriptors(pixels,masks,old['whole']);new['mask']=masks
        extras=[extra(im,m) for im,m in zip(pixels,masks)]
        for i,key in enumerate(['glyph_shape','layout_v2','background_structure','mask_diagnostic']):new[key]=np.asarray([x[i] for x in extras])
        new['ink_shape']=np.asarray([ink_shape(im,m) for im,m in zip(pixels,masks)])
        return old,new

WEIGHTS=dict(glyph_shape=.25,ink_shape=.20,layout_v2=.10,text_color=.10,background_structure=.20,background_color=.10,whole=.05)
class Objective:
    def __init__(self,target,baseline):
        self.std={k:np.maximum(target[k].std(0),.025) for k in WEIGHTS}
        self.scales={k:max(.05,energy(baseline[k]/self.std[k],target[k]/self.std[k])) for k in WEIGHTS}
    def score(self,f,target,rows):
        components={k:energy(f[k]/self.std[k],target[k]/self.std[k])/self.scales[k] for k in WEIGHTS}
        text=sum(WEIGHTS[k]*components[k] for k in ['glyph_shape','ink_shape','layout_v2','text_color'])/.65
        bg=sum(WEIGHTS[k]*components[k] for k in ['background_structure','background_color'])/.30
        appearance=sum(WEIGHTS[k]*components[k] for k in WEIGHTS)
        complete=float(np.mean([r['complete'] for r in rows]));usable=float(np.mean([r['usable'] for r in rows]))
        return dict(components=components,text=text,background=bg,appearance=appearance,objective=appearance+4*(1-complete)+2*(1-usable),complete=complete,usable=usable,strict=float(np.mean([r['strict'] for r in rows])),audit=energy(f['audit'],target['audit']))

