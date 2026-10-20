"""Compact exploratory descriptors. Only image pixels enter this API."""
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
from .views import image_views,padded_canvas

def summaries(a):
    a=np.asarray(a,dtype=float)/255
    return [float(a.mean()),float(a.std()),*np.quantile(a,[.1,.5,.9]).tolist()]

def shape_stats(gray):
    edges=cv2.Canny(gray,60,150)
    n,_,stats,_=cv2.connectedComponentsWithStats((edges>0).astype('uint8'),8)
    sizes=stats[1:,cv2.CC_STAT_AREA]
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=8,minLineLength=8,maxLineGap=1)
    lengths=[]
    if lines is not None:
        lengths=[float(np.hypot(x2-x1,y2-y1)) for x1,y1,x2,y2 in lines[:,0]]
    return [float((edges>0).mean()),min(n-1,200)/200,
            float(np.mean(sizes<=8)) if len(sizes) else 0,
            float(sizes.max()/gray.size) if len(sizes) else 0,
            min(len(lengths),100)/100,max(lengths,default=0)/max(gray.shape)]

def pooled_hog(gray,valid):
    raw=hog(gray,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),
            block_norm='L2-Hys',feature_vector=False)
    # Ignore blocks touching padding, including its boundary gradient.
    eroded=cv2.erode(valid.astype('uint8'),np.ones((3,3),np.uint8))>0
    cells=eroded.reshape(eroded.shape[0]//8,8,eroded.shape[1]//8,8).all(axis=(1,3))
    selected=cells[:-1,:-1]&cells[1:,:-1]&cells[:-1,1:]&cells[1:,1:]
    if not selected.any():return [0.0]*27
    values=raw[selected].reshape(-1,9)
    return np.concatenate([values.mean(0),values.std(0),values.max(0)]).tolist()

def extract(image):
    native=image_views(image);rgb=native['rgb']
    base=[]
    for channel in range(3):base+=summaries(rgb[:,:,channel])
    base+=summaries(native['gray'])+shape_stats(native['gray'])
    canvas,valid,_=padded_canvas(image)
    gray=cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2GRAY)
    base+=pooled_hog(gray,valid)
    extra=[]
    for key in ('clahe','feature_pop','white_tophat','black_tophat','adaptive_light'):
        extra+=summaries(native[key])+shape_stats(native[key])
    # Transform before padding so local contrast does not see padding pixels.
    enhanced,enhanced_valid,_=padded_canvas(Image.fromarray(native['clahe']))
    extra+=pooled_hog(np.asarray(enhanced)[:,:,0],enhanced_valid)
    values=np.asarray(base+extra,dtype='float32')
    if not np.isfinite(values).all():raise ValueError('Invalid image descriptors')
    return values,len(base)
