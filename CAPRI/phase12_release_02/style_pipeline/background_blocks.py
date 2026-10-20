"""Conservative rectangle evidence; color variability alone is not evidence."""
import cv2
import numpy as np

def measure_blocks(rgb,textmask):
    h,w=rgb.shape[:2];smooth=cv2.medianBlur(rgb,3);q=smooth.astype('int32')//32;codes=q[:,:,0]*64+q[:,:,1]*8+q[:,:,2]
    values,counts=np.unique(codes,return_counts=True);candidates=[]
    for code in values[np.argsort(counts)[-32:]]:
        mask=(codes==code).astype('uint8');n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for i in range(1,n):
            x,y,bw,bh,area=map(int,stats[i]);boxarea=bw*bh
            if bw<max(5,w*.025) or bh<max(5,h*.09) or boxarea>.65*w*h or area/boxarea<.88:continue
            if (textmask[y:y+bh,x:x+bw]>0).mean()>.20:continue
            # Require two sharp straight boundaries; soft gradients cannot pass
            # just because quantization turns them into stripe-like regions.
            scores=[];a=smooth.astype('int16')
            if x>=2:scores.append(float((np.max(abs(a[y:y+bh,x+1]-a[y:y+bh,x-2]),axis=1)>=16).mean()))
            if x+bw+1<w:scores.append(float((np.max(abs(a[y:y+bh,x+bw-2]-a[y:y+bh,x+bw+1]),axis=1)>=16).mean()))
            if y>=2:scores.append(float((np.max(abs(a[y+1,x:x+bw]-a[y-2,x:x+bw]),axis=1)>=16).mean()))
            if y+bh+1<h:scores.append(float((np.max(abs(a[y+bh-2,x:x+bw]-a[y+bh+1,x:x+bw]),axis=1)>=16).mean()))
            if sum(v>=.65 for v in scores)<2:continue
            color=np.median(rgb[labels==i],axis=0).astype(int).tolist()
            candidates.append(dict(box=[x,y,x+bw,y+bh],width_fraction=bw/w,height_fraction=bh/h,color=color,fill=area/boxarea,sharp_sides=sum(v>=.65 for v in scores)))
    supported=len(candidates)>=2
    return dict(version='rectangles-v1',state='supported' if supported else 'uncertain',candidates=candidates,count=len(candidates) if supported else 0,reason='requires >=2 solid rectangular color regions, low text overlap and >=2 sharp straight sides each; no color-variation fallback')
