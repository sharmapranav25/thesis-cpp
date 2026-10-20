"""Pixels-only comparison with an internal, synthetic-supervised soft text mask.

No target recipe regression, recognition output, or learned adjustment policy.
Real and synthetic images go through exactly the same mask prediction path.
"""
from .. import runtime as U
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from ..networks.encoder import Encoder
from feature_bank.iter2.model import Iter2Model
from feature_bank.iter2.data import MARGIN
from feature_bank.iter2.generate import TEXT_HEADS,CATEGORIES,STROKES
from captcha_effects import render as R
from captcha_effects.effects import REGISTRY

WEIGHTS=dict(text_color=.2,text_geometry=.2,background_color=.2,background_texture=.2,whole=.2)
def energy(x,y):
    return max(0.,float((2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean())/np.sqrt(x.shape[1])))
def hist(values,weight,bins=16):
    return np.histogram(values,bins=bins,range=(0,1),weights=weight)[0]/max(1e-8,weight.sum())
def color_features(im,w):
    # Cumulative histograms retain color order; soft weights include uncertain pixels.
    return np.concatenate([np.cumsum(hist(im[:,:,c],w)) for c in range(3)])
def descriptors(pixels,masks,z):
    out={k:[] for k in [*WEIGHTS,'audit']}
    for im,m,embedding in zip(pixels/255.,masks,z):
        gray=im@np.array([.299,.587,.114]);bg=1-m
        gx=ndi.sobel(gray,axis=1)/8;gy=ndi.sobel(gray,axis=0)/8;mag=np.hypot(gx,gy)
        # Geometry retains spatial occupancy and interior stroke thickness.
        binary=m>.5;distance=ndi.distance_transform_edt(binary)/16
        geom=np.r_[m.mean(),m.mean(0).reshape(16,16).mean(1),m.mean(1).reshape(8,8).mean(1),
                   np.cumsum(hist(np.clip(distance,0,1),m)),(m*mag).sum()/max(1e-8,m.sum())]
        texture=[]
        for sigma in [0,1,3]:
            g=ndi.gaussian_filter(gray,sigma) if sigma else gray
            for axis in [0,1]:texture.extend(np.cumsum(hist(np.clip(abs(ndi.sobel(g,axis=axis))/4,0,1),bg)))
        for shift in [1,4,12]:
            for axis in [0,1]:
                a=np.roll(gray,shift,axis);weight=bg*np.roll(bg,shift,axis)
                # Do not count wrapped boundaries as texture.
                if axis==0:weight[:shift]=0
                else:weight[:,:shift]=0
                texture.append((abs(gray-a)*weight).sum()/max(1e-8,weight.sum()))
        # Independent audit: grayscale pooled pixels and gradient orientation, no mask/network.
        angle=np.arctan2(gy,gx);orientation=np.histogram(angle,bins=12,range=(-np.pi,np.pi),weights=mag)[0];orientation/=max(1e-8,orientation.sum())
        audit=np.r_[gray.reshape(8,8,16,16).mean((1,3)).ravel(),orientation,np.cumsum(hist(mag,np.ones_like(mag)))]
        out['text_color'].append(color_features(im,m));out['background_color'].append(color_features(im,bg))
        out['text_geometry'].append(geom);out['background_texture'].append(texture)
        out['whole'].append(embedding);out['audit'].append(audit)
    return {k:np.asarray(v) for k,v in out.items()}

class Comparator:
    def __init__(self):
        torch.set_num_threads(4);self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder=Encoder().to(self.device).eval()
        self.encoder.load_state_dict(torch.load(U.ENCODER,map_location=self.device,weights_only=False)['model'])
        self.hybrid=Iter2Model().to(self.device).eval()
        self.hybrid.load_state_dict(torch.load(U.HYBRID,map_location=self.device,weights_only=True))
        self.encoder.requires_grad_(False);self.hybrid.requires_grad_(False)
    @torch.inference_mode()
    def extract(self,images):
        pixels=np.stack([np.asarray(Image.fromarray(im).resize((256,64),Image.Resampling.BILINEAR)) for im in images])
        masks=[];zs=[]
        for start in range(0,len(pixels),32):
            x=torch.tensor(pixels[start:start+32].transpose(0,3,1,2).copy(),device=self.device).float()/255
            o=self.encoder(x);masks.extend(o['mask'].sigmoid()[:,0].cpu().numpy());zs.extend(o['z'].cpu().numpy())
        masks=np.asarray(masks);out=descriptors(pixels,masks,np.asarray(zs));out['mask']=masks
        return out
    @torch.inference_mode()
    def describe(self,images):
        """Frozen hybrid starting labels. CTC strings are never decoded or saved."""
        effects=self.effect_names
        descriptions=[]
        for image in images:
            h,w=image.shape[:2];rw=max(8,min(960,round(w*96/h)));cols=(rw+7)//8;wp=((cols*8+MARGIN+7)//8)*8
            a=np.asarray(Image.fromarray(image).resize((rw,96),Image.Resampling.LANCZOS));a=np.pad(a,((0,0),(0,wp-rw),(0,0)),mode='edge')
            o=self.hybrid(torch.tensor(a.transpose(2,0,1).copy(),device=self.device).float()[None]/255,torch.tensor([cols],device=self.device))
            groups={}
            for block,prefix,names in [('effects','eff',effects),('text_features','txt',TEXT_HEADS)]:
                groups[block]={name:dict(present=bool(o[prefix+'_presence'][0,i]>0),level=int((o[prefix+'_ordinal'][0,i]>0).sum()),probability=float(o[prefix+'_presence'][0,i].sigmoid())) for i,name in enumerate(names)}
            font=dict(category=CATEGORIES[int(o['font_cat'].argmax())],stroke=STROKES[int(o['font_stroke'].argmax())],italic=bool(o['font_italic'][0]>0))
            font.update({name:int((o['font_ordinal'][0,i]>0).sum()) for i,name in enumerate(['weight','width','x_height','slant'])})
            content=['digits','upper','lower','mixed'][int(np.argmax([groups['text_features'][n]['probability'] for n in ['digits_only','uppercase_only','lowercase_only','mixed_case_alnum']]))]
            descriptions.append(dict(size=[w,h],length=int(o['length'].argmax())+2,font=font,content=content,**groups))
        return descriptions
    @property
    def effect_names(self):
        from feature_bank.iter2.generate import effect_heads
        return effect_heads()

class Objective:
    def __init__(self,target,baseline):
        self.scales={k:max(.003,energy(baseline[k],target[k])) for k in WEIGHTS}
    def score(self,features,target,rows):
        parts={k:energy(features[k],target[k]) for k in WEIGHTS}
        normalized={k:parts[k]/self.scales[k] for k in parts}
        text=(normalized['text_color']+normalized['text_geometry'])/2
        background=(normalized['background_color']+normalized['background_texture'])/2
        appearance=sum(WEIGHTS[k]*normalized[k] for k in parts)
        complete=float(np.mean([r['complete'] for r in rows]));usable=float(np.mean([r['usable'] for r in rows]))
        return dict(components=parts,normalized=normalized,text=text,background=background,appearance=appearance,
                    objective=appearance+4*(1-complete)+2*(1-usable),complete=complete,usable=usable,
                    strict=float(np.mean([r['strict'] for r in rows])),audit=energy(features['audit'],target['audit']))

def palette(images,features,region,count=3):
    colors=[];weights=[]
    for image,m in zip(images,features['mask']):
        im=np.asarray(Image.fromarray(image).resize((256,64)),float)
        weight=m if region=='text' else 1-m
        if region=='effect':
            g=im.mean(2)/255;weight=weight*np.clip(np.hypot(ndi.sobel(g,axis=0),ndi.sobel(g,axis=1)),0,1)
        colors.append(im.reshape(-1,3));weights.append(weight.ravel()**3)
    colors=np.concatenate(colors);weights=np.concatenate(weights);q=(colors/16).astype(int);key=q[:,0]*256+q[:,1]*16+q[:,2]
    mass=np.bincount(key,weights=weights,minlength=4096);chosen=[]
    for i in np.argsort(mass)[::-1]:
        if mass[i]<mass.max()*.02:break
        rgb=np.average(colors[key==i],axis=0,weights=weights[key==i])
        if not chosen or min(np.linalg.norm(rgb-c) for c in chosen)>30:chosen.append(rgb)
        if len(chosen)>=count:break
    return [np.rint(c).astype(int).tolist() for c in chosen]
