"""Synthetic-only visual font classifier. No OCR, VLM or real-image fitting."""
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import torch
from torch import nn
from torch.nn import functional as F

@lru_cache(maxsize=64)
def font_at(path,size):return ImageFont.truetype(path,size)

def synthetic(path,seed,clutter=1.,geometry=False):
    rng=np.random.default_rng(seed);w,h=160,64;n=int(rng.integers(3,8));text=''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'),n))
    hollow=bool(rng.random()<.30);font=font_at(path,44);mask=Image.new('L',(w,h))
    geometry_rng=np.random.default_rng(seed+91731)
    outline_kind=int(geometry_rng.integers(3)) if geometry else 0
    spacing=float(rng.uniform(.65,.9));centers=np.linspace((1-spacing)/2,(1+spacing)/2,n)
    for i,c in enumerate(text):
        box=font.getbbox(c);g=Image.new('L',(box[2]-box[0]+8,box[3]-box[1]+8))
        ImageDraw.Draw(g).text((4-box[0],4-box[1]),c,font=font,fill=255)
        g=g.crop(g.getbbox());height=int(rng.integers(20,55) if geometry else rng.integers(29,49));width=max(3,min(int(g.width*height/g.height*rng.uniform(.8,1.2)),int(w*.85/n)))
        g=g.resize((width,height),Image.Resampling.BILINEAR)
        if hollow:
            a=np.pad(np.asarray(g),2);kernel=np.ones((3,3),np.uint8)
            if outline_kind==1:a=cv2.dilate(a,kernel)-a
            elif outline_kind==2:
                large=cv2.resize(a,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
                a=cv2.resize(large-cv2.erode(large,kernel),(a.shape[1],a.shape[0]),interpolation=cv2.INTER_AREA)
            else:a=a-cv2.erode(a,kernel)
            g=Image.fromarray(a)
        g=g.rotate(float(rng.uniform(-12,12)),Image.Resampling.BILINEAR,expand=True)
        mask.paste(g,(round(centers[i]*w-g.width/2),round((h-g.height)/2+rng.uniform(-4,4))))
    background=float(rng.uniform(160,255));foreground=float(rng.uniform(0,100));a=np.asarray(mask)/255
    ramp=np.linspace(-rng.uniform(0,35),rng.uniform(0,35),w)[None,:]
    arr=(background+ramp)*(1-a)+foreground*a+rng.normal(0,rng.uniform(0,9*clutter),(h,w))
    im=Image.fromarray(np.clip(arr,0,255).astype('uint8'));d=ImageDraw.Draw(im)
    for _ in range(int(rng.integers(0,max(4,round(4*clutter))))):
        d.line((0,int(rng.integers(h)),w-1,int(rng.integers(h))),fill=int(rng.integers(80,220)),width=1)
    if rng.random()<.4:im=im.filter(ImageFilter.GaussianBlur(float(rng.uniform(.2,.7))))
    if rng.random()<.3:im=Image.fromarray(255-np.asarray(im))
    return im,int(hollow),text

def encode(image):
    gray=np.asarray(image.convert('L').resize((160,64),Image.Resampling.BILINEAR))
    if np.median(gray)<128:gray=255-gray
    edges=cv2.Canny(gray,50,130)
    return np.stack([gray,edges]).astype('float32')/255

def views(image):
    w,h=image.size;cropw=min(w,round(h*1.8))
    crops=[image]+[image.crop((round(t*(w-cropw)),0,round(t*(w-cropw))+cropw,h)) for t in (0.,.5,1.)]
    return np.stack([encode(im) for im in crops])

class FontMatcher(nn.Module):
    def __init__(self,count):
        super().__init__();self.body=nn.Sequential(nn.Conv2d(2,24,5,2,2),nn.ReLU(),nn.Conv2d(24,48,3,2,1),nn.ReLU(),
            nn.Conv2d(48,96,3,2,1),nn.ReLU(),nn.Conv2d(96,96,3,1,1),nn.ReLU(),nn.AdaptiveAvgPool2d((2,5)),nn.Flatten(),nn.Linear(960,128),nn.ReLU())
        self.font=nn.Linear(128,count);self.hollow=nn.Linear(128,2)
    def forward(self,x):
        z=self.body(x);return self.font(z),self.hollow(z),F.normalize(z,dim=1)

def infer(model,images,device='cuda'):
    xx=np.concatenate([views(im) for im in images]);font=[];hollow=[];emb=[]
    with torch.no_grad():
        for i in range(0,len(xx),64):
            a,b,z=model(torch.from_numpy(xx[i:i+64]).to(device));font.append(a.softmax(1).cpu().numpy());hollow.append(b.softmax(1).cpu().numpy());emb.append(z.cpu().numpy())
    return tuple(np.concatenate(x).reshape(len(images),4,-1).mean(1) for x in (font,hollow,emb))
