"""Frozen DINO plus pixel-only, uncertainty-aware typography diagnostics."""
from .. import runtime as U
from .dino_base import DinoComparator,Objective,WEIGHTS as DINO_WEIGHTS,energy
from .regions import extra
from .custom import ink_shape
from PIL import Image
from scipy import ndimage as ndi
from concurrent.futures import ThreadPoolExecutor
import numpy as np

WEIGHTS={**{k:v*.5 for k,v in DINO_WEIGHTS.items()},'glyph_shape':.12,'ink_shape':.12,'layout_v2':.08,'background_structure':.12,'clarity':.06}

def regions(im,attention):
    rgb=im.astype(float)/255
    # Colour residuals retain isoluminant coloured text; multiple scales keep thin and thick strokes.
    residual=np.maximum.reduce([np.linalg.norm(rgb-ndi.gaussian_filter(rgb,(sigma,sigma,0)),axis=2)/np.sqrt(3) for sigma in [1.,2.,4.]])
    scale=max(.018,float(np.quantile(residual,.92)))
    strength=np.clip(residual/scale,0,1)
    support=strength*(.35+.65*attention)
    labels,n=ndi.label(support>.20);weights=np.ones_like(support)
    for i,sl in enumerate(ndi.find_objects(labels),1):
        h=sl[0].stop-sl[0].start;w=sl[1].stop-sl[1].start;mask=labels[sl]==i;area=mask.sum()
        factor=.25 if h<=2 or w/max(1,h)>10 or area<4 else 1.
        weights[sl][mask]=factor
    mask=np.clip(support*weights*1.5,0,1)
    return mask,residual

def pixel_features(item):
    im,att=item;mask,residual=regions(im,att);glyph,layout,bg,diagnostic=extra(im,mask)
    gray=im.mean(2)/255.;lap=abs(ndi.laplace(gray));gx=ndi.sobel(gray,1)/8;gy=ndi.sobel(gray,0)/8
    values=[residual,np.hypot(gx,gy),lap]
    clarity=[]
    for value in values:
        hist=np.histogram(value,bins=12,range=(0,.5),weights=mask)[0]
        clarity.extend(hist.cumsum()/max(1e-8,hist.sum()))
        clarity.append(float((value*mask).sum()/max(1e-8,mask.sum())))
    clarity.extend([mask.mean(),np.mean(mask>.5),np.mean((mask>.2)&(mask<.8))])
    return dict(glyph_shape=glyph,layout_v2=layout,background_structure=bg,ink_shape=ink_shape(im,mask),
        clarity=np.asarray(clarity),region_mask=mask,contrast=float((residual*mask).sum()/max(1e-8,mask.sum())))

class Comparator(DinoComparator):
    def __init__(self):super().__init__();self.workers=ThreadPoolExecutor(max_workers=4)
    def extract(self,images):
        f=super().extract(images)
        pixels=[np.asarray(Image.fromarray(im).convert('RGB').resize((256,64),Image.Resampling.BILINEAR)) for im in images]
        extras=list(self.workers.map(pixel_features,zip(pixels,f['focus'])))
        for k in extras[0]:f[k]=np.asarray([r[k] for r in extras])
        return f

class Objective7(Objective):
    def __init__(self,target,baseline):
        self.std={k:np.maximum(target[k].std(0),.025) for k in WEIGHTS}
        self.scale={k:max(.05,energy(baseline[k]/self.std[k],target[k]/self.std[k])) for k in WEIGHTS}
        self.contrast_floor=.65*float(np.mean(target['contrast']))
    def score(self,f,target,rows):
        components={k:energy(f[k]/self.std[k],target[k]/self.std[k])/self.scale[k] for k in WEIGHTS}
        appearance=sum(WEIGHTS[k]*components[k] for k in WEIGHTS)
        contrast=float(np.mean(f['contrast']));collapse=max(0,1-contrast/max(.005,self.contrast_floor))
        complete=float(np.mean([r['complete'] for r in rows]));usable=float(np.mean([r['usable'] for r in rows]))
        return dict(components=components,appearance=appearance,contrast=contrast,contrast_penalty=collapse,
            objective=appearance+collapse+4*(1-complete)+2*(1-usable),complete=complete,usable=usable)
    def vector(self,f):return np.concatenate([f[k].mean(0)/self.std[k]*WEIGHTS[k]**.5 for k in WEIGHTS])
