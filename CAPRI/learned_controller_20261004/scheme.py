"""Family-agnostic, labelled recipe grammar with independent component RNGs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,colorsys,hashlib,copy
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'E:/pranav/Final sem/captcha-effects')
sys.path.insert(0,'E:/pranav/Final sem/sourcefree_synth')
from captcha_effects import text as T,styles as S
from captcha_effects.canvas import Canvas
from captcha_effects.captcha import legibility
from feature_bank.iter2.balanced import readability

CONT={
 'size':(.4,1.15),'width':(.55,1.4),'weight':(0.,1.),'spacing':(-.05,1.3),
 'rotation':(0.,30.),'jitter':(0.,.24),'size_jitter':(0.,.25),'slant':(-.25,.25),
 'bg_h':(0.,1.),'bg_s':(0.,.65),'bg_v':(.12,1.),'bg_hspread':(0.,1.),
 'fg_h':(0.,1.),'fg_s':(0.,.9),'fg_v':(.05,.95),'fg_hspread':(0.,1.),'fg_opacity':(.6,1.),
 'fx_h':(0.,1.),'fx_s':(0.,.75),'fx_v':(.12,.98),'fx_hspread':(0.,1.),'fx_opacity':(.2,1.),
 'block_count':(0.,30.),'block_width':(.025,.35),'block_height':(.1,1.35),'block_contrast':(.02,.35),'outline_fraction':(0.,1.),
 'band_count':(0.,8.),'band_width':(.01,.12),'band_slope':(0.,.55),
 'grid_count':(0.,30.),'line_count':(0.,35.),'curve_count':(0.,25.),'curve_amplitude':(.03,.35),
 'dots':(0.,700.),'noise':(0.,.15),'pixel':(1.,3.),'overlay':(0.,.5)}
CAT={'font':96,'length':7,'charset':5,'fg_mode':4,'block_mode':2,'fx_mode':3,'bold':2,'border':2,'font_mix':2}
CN=list(CONT);KN=list(CAT);CHARSETS=['digits','upper','lower','upper_digits','mixed']
_BANK=None
def bank():
 global _BANK
 if _BANK is None:_BANK=T.FontBank()
 return _BANK
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
def make_fonts():
 rng=np.random.default_rng(44981);fonts=[]
 for category in T.FONT_CATEGORIES:
  pool=[f for f in bank().pool('train',category=category) if not f['caps_only'] and all(c in f.get('chars','') for c in 'aAgG079')]
  seen=set();pool=[f for f in pool if not (f['family'] in seen or seen.add(f['family']))]
  for i in rng.permutation(len(pool))[:20]:fonts.append(pool[i])
 pool=[f for f in bank().pool('train') if not f['caps_only'] and f not in fonts]
 while len(fonts)<96:fonts.append(pool.pop(0))
 fonts=fonts[:96];dump(ROOT/'fonts.json',fonts);return fonts
def fonts():return json.loads((ROOT/'fonts.json').read_text())
def decode(c,k):
 return {n:float(CONT[n][0]+float(c[i])*(CONT[n][1]-CONT[n][0])) for i,n in enumerate(CN)}|{n:int(k[i]) for i,n in enumerate(KN)}
def encode(p):return np.array([(p[n]-lo)/(hi-lo) for n,(lo,hi) in CONT.items()],np.float32),np.array([p[n] for n in KN],np.int64)
def sample(seed):
 r=np.random.default_rng(seed);p={n:float(r.uniform(*b)) for n,b in CONT.items()};p.update({n:int(r.integers(v)) for n,v in CAT.items()})
 # Sparse independent layer grammar; no target-family branches or presets.
 for key,prob in [('block_count',.45),('band_count',.35),('grid_count',.3),('line_count',.25),('curve_count',.25),('dots',.35),('noise',.35)]:
  if r.random()>prob:p[key]=0.
 for key,prob in [('rotation',.65),('jitter',.6),('size_jitter',.5),('slant',.3)]:
  if r.random()>prob:p[key]=0.
 if r.random()<.55:p['pixel']=1.
 if r.random()<.35:p.update(bg_s=0.,bg_v=1.,bg_hspread=0.)
 if r.random()<.45:p.update(fg_s=0.,fg_v=float(r.choice([.1,.9])),fg_hspread=0.)
 if r.random()<.5:p.update(fx_s=0.,fx_hspread=0.)
 if r.random()<.75:p['fg_opacity']=1.
 if r.random()<.65:p['font_mix']=0
 p['border']=int(r.random()<.3)
 # Contrast is part of generic sampling, never a real-family preset.
 if p['fg_mode']==0 and abs(p['fg_v']-p['bg_v'])<.3:p['fg_v']=.1 if p['bg_v']>.5 else .92
 return p
def perturb(p,seed):
 r=np.random.default_rng(seed);c,k=encode(p)
 active=r.random(len(c))<.3;c=np.clip(c+r.normal(0,.18,len(c))*active,0,1)
 for i,n in enumerate(KN):
  if r.random()<.2:k[i]=r.integers(CAT[n])
 return decode(c,k)
def component_rng(seed,name):return np.random.default_rng([int(seed),int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')])
def hsv(h,s,v):return tuple(int(round(x*255)) for x in colorsys.hsv_to_rgb(float(h)%1,float(np.clip(s,0,1)),float(np.clip(v,0,1))))
def rgb(p,prefix,r,spread=True):return hsv(p[prefix+'_h']+(r.uniform(-.5,.5)*p[prefix+'_hspread'] if spread else 0),p[prefix+'_s'],p[prefix+'_v'])
def line(cv,pts,width,color,head,opacity=1.,aa=True):
 cov=cv.mask(lambda d,s:d.line([(x*s,y*s) for x,y in pts],fill=255,width=max(1,round(width*s))),antialias=aa)
 cv.blend(cov,color,'controller_'+head,head,'under',opacity,dict(points=pts,width=width))
def rect(cv,box,color,outline=False,opacity=1.,head='bg_blocks'):
 cov=cv.mask(lambda d,s:d.rectangle([v*s for v in box],fill=None if outline else 255,outline=255 if outline else None,width=max(1,s)),antialias=False)
 cv.blend(cov,color,'controller_'+head,head,'under',opacity,dict(box=box,outline=outline))

def render(item):
 p,seed,size,keep=item
 fs=fonts();w,h=map(int,size);r=component_rng(seed,'canvas');bg=rgb(p,'bg',r)
 cv=Canvas(w,h,bg,supersample=2)
 def fxcolor(r):
  if p['fx_mode']==1:
   delta=float(r.uniform(-1,1))*p['block_contrast']*255
   return tuple(int(np.clip(v+delta,0,255)) for v in bg)
  if p['fx_mode']==2:
   v=int(p['fx_v']*255);return (v,v,v)
  return rgb(p,'fx',r)
 r=component_rng(seed,'blocks')
 for i in range(int(round(p['block_count']))):
  bw=w*p['block_width']*r.uniform(.6,1.4);bh=h*p['block_height']*r.uniform(.6,1.4)
  x=r.uniform(-.05*w,w-bw*.3) if p['block_mode']==0 else (i+.2)*w/max(1,p['block_count'])
  y=r.uniform(-.1*h,max(-.09*h,h-bh))
  rect(cv,[x,y,x+bw,y+bh],fxcolor(r),r.random()<p['outline_fraction'],p['fx_opacity'])
 r=component_rng(seed,'bands')
 for i in range(int(round(p['band_count']))):
  y=r.uniform(0,h);dy=r.uniform(-1,1)*p['band_slope']*h
  line(cv,[(-2,y),(w+2,y+dy)],max(1,h*p['band_width']*r.uniform(.6,1.4)),fxcolor(r),'bands',p['fx_opacity'],aa=p['pixel']<1.5)
 r=component_rng(seed,'grid');n=int(round(p['grid_count']))
 if n:
  step=w/max(2,n)
  for x in np.arange(0,w,step):line(cv,[(float(x),0),(float(x),h)],1,fxcolor(r),'bg_grid',p['fx_opacity'],False)
  for y in np.arange(0,h,step):line(cv,[(0,float(y)),(w,float(y))],1,fxcolor(r),'bg_grid',p['fx_opacity'],False)
 r=component_rng(seed,'lines')
 for i in range(int(round(p['line_count']))):line(cv,[(float(r.uniform(0,w)),float(r.uniform(0,h))),(float(r.uniform(0,w)),float(r.uniform(0,h)))],1,fxcolor(r),'straight_lines',p['fx_opacity'])
 r=component_rng(seed,'curves')
 for i in range(int(round(p['curve_count']))):
  xs=np.linspace(-2,w+2,50);ys=r.uniform(0,h)+p['curve_amplitude']*h*np.sin(xs/r.uniform(.035*w,.15*w)+r.uniform(0,7))
  line(cv,list(zip(xs.tolist(),ys.tolist())),1,fxcolor(r),'curves',p['fx_opacity'])
 r=component_rng(seed,'dots')
 for i in range(int(round(p['dots']))):
  x,y=int(r.integers(w)),int(r.integers(h));rect(cv,[x,y,x+1,y+1],fxcolor(r),False,p['fx_opacity'],'dot_noise')
 pre=cv.img.copy()
 tr=component_rng(seed,'string');n=p['length']+2;charset=CHARSETS[p['charset']]
 text=T.make_text(tr,n,charset)
 fr=component_rng(seed,'glyph_palette');palette=[]
 for i in range(n):
  if p['fg_mode']==1:c=(245,245,245) if i%2==0 else (15,15,15)
  elif p['fg_mode']==2:c=rgb(p,'fg',fr)
  elif p['fg_mode']==3:c=(245,245,245) if fr.random()<.5 else (15,15,15)
  else:c=rgb(p,'fg',component_rng(seed,'single_colour'))
  palette.append(c)
 ts=dict(text=text,charset=charset,layout='line',fonts=[fs[p['font']]['key']],size=p['size'],size_jitter=p['size_jitter'],weight=p['weight'],bold_stroke=p['bold'],x_scale=p['width'],slant=p['slant'],rotation=p['rotation'],jitter=p['jitter'],spacing=p['spacing'],curve=0.,multicolour=p['fg_mode']!=0,palette='controller_explicit',min_contrast=0.)
 if p['font_mix']:ts['fonts'].append(fs[(p['font']+1+int(tr.integers(len(fs)-1)))%len(fs)]['key']);ts['font_assign']='random'
 orig=S.colour;state=[0]
 def color(r,name,*args,**kwargs):
  if name!='controller_explicit':return orig(r,name,*args,**kwargs)
  c=palette[min(state[0],n-1)];state[0]+=1;return c
 S.colour=color
 try:txt=T.draw_text_spec(cv,ts,component_rng(seed,'text_layout'),bank(),ss=2,colour=palette[0])
 finally:S.colour=orig
 if txt is None:raise RuntimeError('No glyphs')
 if p['fg_opacity']<1:cv.img=pre+(cv.img-pre)*p['fg_opacity']
 for rec in cv.records:
  if rec['head']=='text':rec['opacity']=p['fg_opacity']
 r=component_rng(seed,'grain')
 if p['noise']>0:cv.img+=r.normal(0,p['noise']*255,(h,w,1)).astype(np.float32)
 if p['overlay']>0:
  r=component_rng(seed,'overlay')
  for i in range(int(round(p['line_count']*p['overlay']))):line(cv,[(0,float(r.uniform(0,h))),(w,float(r.uniform(0,h)))],1,fxcolor(r),'noise_over_text',p['fx_opacity']*.5)
 if p['border']:rect(cv,[0,0,w-1,h-1],(30,30,30),True,1.,'border_frame')
 if p['pixel']>1.1:
  sz=(max(1,round(w/p['pixel'])),max(1,round(h/p['pixel'])))
  cv.img=np.asarray(Image.fromarray(cv.image()).resize(sz,Image.Resampling.BOX).resize((w,h),Image.Resampling.NEAREST),np.float32)
  cv.opaque=np.asarray(Image.fromarray(cv.opaque).resize(sz,Image.Resampling.NEAREST).resize((w,h),Image.Resampling.NEAREST))
 mask=np.isin(cv.opaque,txt['ids']).astype(np.uint8)*255
 tl=T.text_labels(txt)
 for slot,g in zip(tl['slots'],txt['glyphs']):
  yy,xx=np.nonzero(cv.opaque==g['id']);slot['box']=None if not len(xx) else [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
 lg=legibility(cv,txt);tl['legibility']=lg;tl['usable']=bool(tl['usable'] and lg['ok'])
 complete=tl['text']==text and len(tl['slots'])==n and all(g['box'] is not None for g in tl['slots'])
 out=dict(image=cv.image(),opaque=cv.opaque,records=cv.records,text=tl)
 strict,guard=readability(out)
 row=dict(image=out['image'],mask=mask,text=text,seed=int(seed),complete=bool(complete),usable=bool(complete and tl['usable']),strict=bool(complete and strict),guard=guard)
 if keep:row.update(recipe=p,text_labels=tl,records=cv.records,pixel_sha256=hashlib.sha256(row['image'].tobytes()).hexdigest())
 return row
