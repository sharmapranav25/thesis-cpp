"""Appearance descriptors from inferred regions; never supplied target labels."""
from bootstrap import U
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from comparison import Comparator,energy

def extra(im,m):
    gray=im.mean(2)/255.;binary=m>.5
    labels,n=ndi.label(ndi.binary_closing(binary,structure=np.ones((2,2))))
    glyphs=[];count=0
    for label,sl in enumerate(ndi.find_objects(labels),1):
        h=sl[0].stop-sl[0].start;w=sl[1].stop-sl[1].start
        mask=(labels[sl]==label);area=mask.sum()
        if h<7 or w<2 or area<10:continue
        count+=1
        crop=np.asarray(Image.fromarray((mask*255).astype('uint8')).resize((24,32)),float)/255
        gx=ndi.sobel(crop,axis=1);gy=ndi.sobel(crop,axis=0);mag=np.hypot(gx,gy)
        hist=np.histogram(np.arctan2(gy,gx),bins=12,range=(-np.pi,np.pi),weights=mag)[0];hist/=max(1e-8,hist.sum())
        dist=ndi.distance_transform_edt(mask);stroke=np.quantile(dist[mask],[.25,.5,.9])/max(1,h)
        glyphs.append(np.r_[w/h,h/64,w/256,area/(h*w),stroke,mask.sum()/max(1,(mask^ndi.binary_erosion(mask)).sum())/h,
                           crop.mean(0).reshape(6,4).mean(1),crop.mean(1).reshape(8,4).mean(1),hist])
    gd=len(glyphs[0]) if glyphs else 34
    glyph=np.r_[np.quantile(glyphs,[.2,.5,.8],axis=0).ravel() if glyphs else np.zeros(gd*3),min(count,12)/12]
    col=ndi.gaussian_filter1d(m.sum(0),1);active=col>max(.8,col.max()*.13)
    runs=ndi.label(active)[1]
    layout=np.r_[min(count,12)/12,min(runs,12)/12,m.mean(),np.quantile(col,[.1,.5,.9])/64,
                 m.mean(0).reshape(16,16).mean(1),m.mean(1).reshape(8,8).mean(1)]
    # Exclude uncertain and dilated text boundaries from all structural measurements.
    outside=~ndi.binary_dilation(m>.2,iterations=2)
    if outside.sum()<100:outside=m<.5
    structure=[]
    for sigma in [.6,1.5,3.]:
        g=ndi.gaussian_filter(gray,sigma);gx=ndi.sobel(g,axis=1)/8;gy=ndi.sobel(g,axis=0)/8
        magnitude=np.hypot(gx,gy)*outside;angle=np.arctan2(gy,gx)%np.pi
        hist=np.histogram(angle,bins=12,range=(0,np.pi),weights=magnitude)[0]
        hist/=max(1e-8,hist.sum());structure.extend(hist)
        structure.extend([np.sum(abs(gx)*outside)/max(1,outside.sum()),np.sum(abs(gy)*outside)/max(1,outside.sum())])
    for axis in [0,1]:
        for shift in [1,2,4,8,16,24]:
            a=np.roll(gray,shift,axis);valid=outside*np.roll(outside,shift,axis)
            if axis==0:valid[:shift]=False
            else:valid[:,:shift]=False
            structure.extend([np.sum(abs(gray-a)*valid)/max(1,valid.sum()),np.sum((gray-a)**2*valid)/max(1,valid.sum())])
    # Local color variation at several scales captures flat blocks versus fine clutter.
    for size in [3,7,15]:
        variance=np.maximum(0,ndi.uniform_filter(gray*gray,size)-ndi.uniform_filter(gray,size)**2)
        structure.extend(np.quantile(np.sqrt(variance)[outside],[.25,.5,.75,.95]) if outside.any() else np.zeros(4))
    confidence=np.array([np.mean((m>.2)&(m<.8)),m.mean(),float(count)])
    return glyph,layout,np.asarray(structure),confidence

class Perception(Comparator):
    def extract(self,images):
        f=super().extract(images)
        extra_features=[extra(np.asarray(Image.fromarray(im).resize((256,64))),m) for im,m in zip(images,f['mask'])]
        for j,key in enumerate(['glyph_shape','layout_v2','background_structure','mask_diagnostic']):f[key]=np.array([r[j] for r in extra_features])
        return f

WEIGHTS=dict(glyph_shape=.30,layout_v2=.15,text_color=.10,background_structure=.25,background_color=.10,whole=.10)
class Objective:
    def __init__(self,target,baseline):
        self.std={k:np.maximum(target[k].std(0),.025) for k in WEIGHTS}
        self.scales={k:max(.05,energy(baseline[k]/self.std[k],target[k]/self.std[k])) for k in WEIGHTS}
    def score(self,f,target,rows):
        components={k:energy(f[k]/self.std[k],target[k]/self.std[k])/self.scales[k] for k in WEIGHTS}
        text=sum(WEIGHTS[k]*components[k] for k in ['glyph_shape','layout_v2','text_color'])/.55
        bg=sum(WEIGHTS[k]*components[k] for k in ['background_structure','background_color'])/.35
        appearance=sum(WEIGHTS[k]*components[k] for k in WEIGHTS)
        complete=float(np.mean([r['complete'] for r in rows]));usable=float(np.mean([r['usable'] for r in rows]))
        return dict(components=components,text=text,background=bg,appearance=appearance,objective=appearance+4*(1-complete)+2*(1-usable),
                    complete=complete,usable=usable,strict=float(np.mean([r['strict'] for r in rows])),audit=energy(f['audit'],target['audit']))
