"""Deterministic pixel-only diagnostic views, never feature segmentation."""
import cv2
import numpy as np
from PIL import Image

VERSION='feature-pop-v1'

def image_views(image):
    rgb=np.asarray(image.convert('RGB')).copy()
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    edges=cv2.Canny(gray,60,150)
    white=[];black=[]
    for size in (3,7,15):
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
        white.append(cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel))
        black.append(cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel))
    top=np.maximum.reduce(white);bottom=np.maximum.reduce(black)
    response=np.maximum(top,bottom)
    # A fixed gain preserves the distinction between faint and strong evidence.
    # Per-image min/max stretching would make weak noise look fully confident.
    pop=np.clip(response.astype('float32')*2,0,255).astype('uint8')
    block=min(31,min(gray.shape)//2*2-1)
    if block>=3:
        threshold=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block,5)
    else:threshold=np.where(gray>=128,255,0).astype('uint8')
    return dict(rgb=rgb,gray=gray,clahe=clahe,canny=edges,white_tophat=top,black_tophat=bottom,
                adaptive_light=threshold,adaptive_dark=255-threshold,feature_pop=pop)

def padded_canvas(image,size=(256,96)):
    """Original aspect ratio and an explicit valid mask; padding is not evidence."""
    scale=min(size[0]/image.width,size[1]/image.height)
    width=max(1,round(image.width*scale));height=max(1,round(image.height*scale))
    resized=image.convert('RGB').resize((width,height),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',size,(127,127,127));x=(size[0]-width)//2;y=(size[1]-height)//2
    canvas.paste(resized,(x,y));valid=np.zeros((size[1],size[0]),bool);valid[y:y+height,x:x+width]=True
    return canvas,valid,dict(scale=scale,offset=[x,y],valid_box=[x,y,x+width,y+height],original_size=list(image.size))
