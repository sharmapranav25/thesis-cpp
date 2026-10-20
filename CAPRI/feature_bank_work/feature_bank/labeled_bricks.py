"""Instrumented Pillow overlays and conservative local supervision.

The masks describe final counterfactual contribution, not human ground truth.
All numeric settings are proposed pilot settings; no real images are read.
"""
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw

FEATURES = ('straight_interference','curved_interference','dot_scatter')
POLICY = dict(version='local-evidence-v3',contrast_delta=20,min_mark_pixels=8,
              min_visible_fraction=0.65,min_line_span=8,min_curve_deviation=1.5,
              min_complete_dots=2,min_dot_visible_fraction=0.7)

def sample_operations(width,height,seed,dark=False):
    """Sample ONCE. Store all actual geometry; composition consumes no RNG."""
    rng=np.random.default_rng(seed)
    # Identical sampling distribution for every effect; no effect-specific color.
    lo,hi=(25,126) if not dark else (130,236)
    palette=[tuple(int(v) for v in rng.integers(lo,hi,3)) for _ in FEATURES]
    y=int(rng.integers(height//5,4*height//5))
    line=[(3,y),(width-4,int(rng.integers(height//5,4*height//5)))]
    amplitude=float(rng.uniform(8,16)); period=float(rng.uniform(48,78)); phase=float(rng.uniform(0,2*math.pi))
    cy=float(rng.uniform(height*0.35,height*0.65))
    curve=[(x,round(cy+amplitude*math.sin(2*math.pi*x/period+phase))) for x in range(3,width-3)]
    dots=[]
    for _ in range(55):
        radius=int(rng.integers(1,3)); x=int(rng.integers(radius,width-radius)); y=int(rng.integers(radius,height-radius))
        dots.append([x,y,radius])
    return [dict(feature=FEATURES[0],kind='polyline',points=line,width=int(rng.integers(1,3)),color=palette[0]),
            dict(feature=FEATURES[1],kind='polyline',points=curve,width=int(rng.integers(1,3)),color=palette[1]),
            dict(feature=FEATURES[2],kind='dots',dots=dots,color=palette[2])]

def operation_mask(size,operation):
    mask=Image.new('L',size,0);draw=ImageDraw.Draw(mask)
    if operation['kind']=='polyline':
        draw.line([tuple(p) for p in operation['points']],fill=255,width=operation['width'])
    elif operation['kind']=='dots':
        for x,y,r in operation['dots']: draw.ellipse((x-r,y-r,x+r,y+r),fill=255)
    else: raise ValueError('Unknown operation')
    return mask

def compose(base,operations,enabled,omit=None):
    image=base.copy()
    for op in operations:
        if op['feature'] in enabled and op['feature']!=omit:
            image.paste(tuple(op['color']),mask=operation_mask(image.size,op))
    return image

def evidence(base,operations,enabled,background=None):
    final=compose(base,operations,enabled); arr=np.asarray(final).astype('int16'); result={}
    for op in operations:
        raw=np.asarray(operation_mask(base.size,op))>0 if op['feature'] in enabled else np.zeros(arr.shape[:2],bool)
        without=compose(base,operations,enabled,omit=op['feature'])
        visible=np.max(np.abs(arr-np.asarray(without).astype('int16')),axis=2)>=POLICY['contrast_delta']
        if np.any(visible & ~raw): raise ValueError('Counterfactual changed pixels outside effect mask')
        context=None if background is None else np.max(np.abs(
            np.asarray(without).astype('int16')-np.asarray(background).astype('int16')),axis=2)>=12
        result[op['feature']]=dict(raw=raw,visible=visible,without=without,context=context)
    return final,result

def local_label(op,ev,box):
    x0,y0,x1,y1=box; raw=ev['raw'][y0:y1,x0:x1]; vis=ev['visible'][y0:y1,x0:x1]
    nr,nv=int(raw.sum()),int(vis.sum())
    info=dict(raw_pixels=nr,visible_pixels=nv,visible_fraction=nv/nr if nr else None)
    if nr==0: return 'absent',dict(info,reason='no added operation pixels within this crop')
    if nv<POLICY['min_mark_pixels'] or nv/nr<POLICY['min_visible_fraction']:
        return 'uncertain',dict(info,reason='too little visible contribution or substantial occlusion')
    if op['kind']=='dots':
        if ev.get('context') is None:
            return 'uncertain',dict(info,reason='background context unavailable for mark isolation')
        complete_components=set()
        _,components=cv2.connectedComponents(vis.astype('uint8'),connectivity=8)
        for x,y,r in op['dots']:
            if not(x-r>=x0 and x+r<x1 and y-r>=y0 and y+r<y1): continue
            mask=np.asarray(operation_mask((x1-x0,y1-y0),dict(kind='dots',dots=[[x-x0,y-y0,r]])))>0
            if int((mask&vis).sum())/max(int(mask.sum()),1)>=POLICY['min_dot_visible_fraction']:
                identifiers=set(int(v) for v in np.unique(components[mask&vis]))-{0}
                if len(identifiers)==1:
                    identifier=next(iter(identifiers))
                    # Check the entire component and its one-pixel halo in the
                    # parent, including pixels just outside the crop boundary.
                    full=np.zeros_like(ev['visible'],dtype='uint8')
                    full[y0:y1,x0:x1]=(components==identifier).astype('uint8')
                    halo=cv2.dilate(full,np.ones((3,3),dtype='uint8'))>0
                    if not np.any(halo & ev['context']):complete_components.add(identifier)
        complete=len(complete_components)
        info['complete_visible_dots']=complete
        return ('present' if complete>=POLICY['min_complete_dots'] else 'uncertain'),dict(info,reason='require two complete distinct marks isolated from text and other effects')
    count,components=cv2.connectedComponents(vis.astype('uint8'),connectivity=8)
    sizes=np.bincount(components.ravel());sizes[0]=0
    component=int(np.argmax(sizes));connected=components==component
    if sizes[component]<POLICY['min_mark_pixels'] or sizes[component]/nv<POLICY['min_visible_fraction']:
        return 'uncertain',dict(info,reason='visible stroke fragmented into disconnected pieces')
    ys,xs=np.where(connected); span=float(np.hypot(np.ptp(xs),np.ptp(ys)));info['visible_span']=span
    if span<POLICY['min_line_span']: return 'uncertain',dict(info,reason='fragment too short')
    if op['feature']=='curved_interference':
        points=np.array([p for p in op['points'] if x0<=p[0]<x1 and y0<=p[1]<y1
                         and connected[p[1]-y0,p[0]-x0]],dtype=float)
        # A crop can contain disconnected arc fragments. Require consecutive x
        # positions and a visibly non-straight arc; otherwise abstain.
        if len(points)<8 or np.any(np.diff(points[:,0])>1):
            return 'uncertain',dict(info,reason='insufficient continuous arc context')
        center=points-points.mean(axis=0);_,_,axes=np.linalg.svd(center,full_matrices=False)
        deviation=float(np.max(np.abs(center@axes[1])));info['curve_deviation_pixels']=deviation
        if deviation<POLICY['min_curve_deviation']:
            return 'uncertain',dict(info,reason='arc looks locally approximately straight')
    return 'present',dict(info,reason='visible geometry meets fixed pilot evidence rules')

def crop_labels(operations,ev,box):
    labels={};details={}
    for op in operations:
        label,detail=local_label(op,ev[op['feature']],box)
        labels[op['feature']]=label;details[op['feature']]=detail
    # Do not claim a negative straight label when a locally ambiguous curve
    # could look straight, or a negative dot label for a tiny line fragment.
    if labels['straight_interference']=='absent' and labels['curved_interference']=='uncertain':
        labels['straight_interference']='uncertain';details['straight_interference']['reason']='curve fragment may look straight'
    if labels['dot_scatter']=='absent' and any(labels[f]=='uncertain' for f in FEATURES[:2]):
        labels['dot_scatter']='uncertain';details['dot_scatter']['reason']='small interference fragment could resemble a mark'
    return labels,details
