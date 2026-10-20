"""Untrained MSER proposals with topology diagnostics; not transcription."""
import cv2
import numpy as np

def propose(gray):
    h,w=gray.shape
    mser=cv2.MSER_create(5,max(8,int(h*w*.0008)),max(9,int(h*w*.14)),.35,.2)
    regions,boxes=mser.detectRegions(gray)
    candidates=[]
    for points,(x,y,bw,bh) in zip(regions,boxes):
        fill=len(points)/max(1,bw*bh)
        if not (.23*h<bh<.94*h and .025*w<bw<.3*w and .08<fill<.85):continue
        if x<=0 or y<=0 or x+bw>=w or y+bh>=h:continue
        local=np.zeros((bh+2,bw+2),np.uint8);local[points[:,1]-y+1,points[:,0]-x+1]=1
        dist=cv2.distanceTransform(local,cv2.DIST_L2,3);stroke=float(np.quantile(dist[local>0],.85)*2)
        contours,hierarchy=cv2.findContours(local,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        holes=0.;outer=0.
        if hierarchy is not None:
            for c,hi in zip(contours,hierarchy[0]):
                if hi[3]>=0:holes+=cv2.contourArea(c)
                else:outer+=cv2.contourArea(c)
        candidates.append(dict(points=points,box=[int(x),int(y),int(x+bw),int(y+bh)],
                               height=int(bh),stroke=stroke,hollow=float(holes/max(outer,1))))
    # Retain disjoint glyph-like regions; nested threshold replicas are not
    # independent characters. Prefer more complete, taller candidates.
    accepted=[]
    for c in sorted(candidates,key=lambda c:(-c['height'],-len(c['points']))):
        x0,y0,x1,y1=c['box'];area=(x1-x0)*(y1-y0);overlap=False
        for p in accepted:
            a,b,d,e=p['box'];intersection=max(0,min(x1,d)-max(x0,a))*max(0,min(y1,e)-max(y0,b))
            if intersection/max(1,min(area,(d-a)*(e-b)))>.45:overlap=True;break
        if not overlap:accepted.append(c)
        if len(accepted)>=12:break
    return sorted(accepted,key=lambda c:c['box'][0])
