"""Independent synthetic operations with leave-one-operation-out visibility labels."""
import hashlib,json,math
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont,ImageFilter
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');RELEASE=ROOT/'releases/phase12_v2'
FEATURES=['straight_interference','curved_interference','dot_scatter','decorative_shapes','periodic_grid','background_texture','background_gradient','background_blocks','hollow_lettering','glyph_border','glyph_shadow','nonlinear_text_deformation','blur','glyph_rotation','multicolor_text']
def fonts():
 bank=json.loads((RELEASE/'assets/fonts.json').read_text())[:9]
 for f in bank:assert hashlib.sha256((RELEASE/f['path']).read_bytes()).hexdigest()==f['sha256']
 return bank
def recipe(seed,part):
 rng=np.random.default_rng(seed)
 # Joint combinations split independently of nuisance draws. Never split related counterfactuals.
 for _ in range(1000):
  flags=rng.random(len(FEATURES))<rng.uniform(.12,.50)
  if flags[8] and flags[9]:flags[rng.choice([8,9])]=False
  code=sum(int(v)<<i for i,v in enumerate(flags))
  held=int(hashlib.sha256(str(code).encode()).hexdigest()[:8],16)%7==0
  if held==(part=='locked_combinations'):break
 else:raise RuntimeError('combination allocation failed')
 return flags

def render(seed,flags,bank,part,omit=-1):
 active=flags.copy()
 if omit>=0:active[omit]=False
 w,h=192,96
 def rng(i):return np.random.default_rng(np.random.SeedSequence([int(seed),int(i)+1]))
 r=rng(40);bg=r.integers(0,256,3);im=Image.new('RGB',(w,h),tuple(bg))
 if active[6]:
  rr=rng(6);end=np.clip(bg+rr.choice([-1,1])*rr.integers(60,150,3),0,255)
  if part=='locked_recipe':t=np.linspace(0,1,h)[:,None,None];a=np.repeat(bg*(1-t)+end*t,w,axis=1)
  else:t=np.linspace(0,1,w)[None,:,None];a=np.repeat(bg*(1-t)+end*t,h,axis=0)
  im=Image.fromarray(np.uint8(a))
 if active[7]:
  rr=rng(7);d=ImageDraw.Draw(im)
  for _ in range(int(rr.integers(3,13))):
   x,y=rr.integers([0,0],[w-10,h-10]);bw,bh=rr.integers([10,10],[55,60]);d.rectangle((x,y,x+bw,y+bh),fill=tuple(rr.integers(0,256,3)))
 if active[5]:
  rr=rng(5);noise=rr.normal(0,rr.uniform(12,35),(h,w,1));noise=np.repeat(noise,3,2)
  if part=='locked_recipe':noise=cv2.GaussianBlur(noise,(3,3),.5)
  im=Image.fromarray(np.uint8(np.clip(np.array(im).astype(float)+noise,0,255)))
 if active[4]:
  rr=rng(4);d=ImageDraw.Draw(im);color=tuple(rr.integers(0,256,3));step=int(rr.integers(9,27))
  for x in range(0,w,step):d.line((x,0,x,h),fill=color,width=1)
  for y in range(0,h,step):d.line((0,y,w,y),fill=color,width=1)
 # Glyph geometry comes from independent licensed fonts, never target families.
 rr=rng(45);font=ImageFont.truetype(str(RELEASE/bank[int(rr.integers(len(bank)))]['path']),int(rr.integers(30,48)));text=''.join(rr.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'),int(rr.integers(2,9))))
 masks=[];colours=[];base_color=rr.integers(0,256,3)
 for j,c in enumerate(text):
  g=Image.new('L',(64,64));ImageDraw.Draw(g).text((5,0),c,font=font,fill=255)
  if active[13]:g=g.rotate(float(rng(80+j).choice([-1,1])*rng(90+j).uniform(12,35)),Image.Resampling.BILINEAR)
  mask=Image.new('L',(w,h));mask.paste(g,(int(6+j*(w-30)/len(text)),int(rr.integers(5,25))));a=np.array(mask)
  if active[11]:
   yy,xx=np.indices((h,w),dtype=np.float32);yy+=5*np.sin(xx/rng(11).uniform(16,28));a=cv2.remap(a,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
  masks.append(a);colours.append(tuple(rng(110+j).integers(0,256,3)) if active[14] else tuple(base_color))
 for a,color in zip(masks,colours):
  if active[10]:
   shadow=np.roll(np.roll(a,4,0),4,1);im.paste(tuple(np.clip(np.array(color)+80,0,255)),mask=Image.fromarray(shadow))
  if active[9]:
   border=cv2.dilate(a,np.ones((3,3),np.uint8));im.paste(tuple(255-np.array(color)),mask=Image.fromarray(border))
  if active[8]:a=a-cv2.erode(a,np.ones((3,3),np.uint8))
  im.paste(color,mask=Image.fromarray(a))
 for k in [0,1,2,3]:
  if not active[k]:continue
  rr=rng(k);d=ImageDraw.Draw(im)
  for _ in range(int(rr.integers(1,5)) if k!=2 else int(rr.integers(8,65))):
   color=tuple(rr.integers(0,256,3));width=int(rr.integers(1,4))
   if k==0:
    pts=[tuple(v) for v in rr.integers([0,0],[w,h],(2,2))];d.line(pts,fill=color,width=width)
   elif k==1:
    t=np.linspace(0,1,70);cy=rr.uniform(20,h-20);amp=rr.uniform(9,24)
    yy=cy+amp*(4*(t-.5)**2-1) if part=='locked_recipe' else cy+amp*np.sin(t*rr.uniform(3,7)+rr.uniform(-2,2))
    d.line(list(zip(np.rint(t*(w-1)).astype(int),np.rint(yy).astype(int))),fill=color,width=width)
   elif k==2:
    x,y=rr.integers([0,0],[w,h]);rad=int(rr.integers(1,4));d.ellipse((x-rad,y-rad,x+rad,y+rad),fill=color)
   else:
    x,y=rr.integers([0,0],[w-20,h-20]);size=int(rr.integers(10,30));d.ellipse((x,y,x+size,y+size),outline=color,width=width)
 if active[12]:im=im.filter(ImageFilter.GaussianBlur(float(rng(12).uniform(.8,1.5))))
 return np.array(im)

def sample(seed,part,bank):
 flags=recipe(seed,part);im=render(seed,flags,bank,part);labels=flags.astype(np.float32);valid=np.ones(len(flags),np.float32);effects=np.zeros(len(flags),np.float32)
 for k in np.where(flags)[0]:
  delta=np.max(np.abs(im.astype(float)-render(seed,flags,bank,part,int(k)).astype(float)),2)
  effects[k]=float((delta>=12).mean())
  if np.count_nonzero(delta>=12)<20:valid[k]=0 # ambiguous/invisible operation is ignored, not falsely labelled absent
 return im,labels,valid,dict(seed=seed,flags=flags.astype(int).tolist(),visible_changed_fraction=effects.tolist())
