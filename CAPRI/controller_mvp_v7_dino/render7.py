"""Complete text controls and generic glyph outlines/frames, without target metadata."""
from setup7 import U
import controls as C
import copy,math
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi

ORIGINAL=C.R.T.glyph_mask
ACTIVE=None
DEFAULT=dict(charset=None,layout=None,size_absolute=None,width_absolute=None,spacing_absolute=None,
    weight_absolute=None,bold_absolute=None,slant_absolute=None,size_jitter_absolute=None,curve_absolute=None,
    glyph_outline=0,frame_width=0,frame_padding=2)

def expand(p):return {**C.expand(p),**DEFAULT,**copy.deepcopy(p)}

def glyph_mask(meta,ch,size_px,ss=4,weight=None,width_axis=None,stroke=0,x_scale=1.,slant=0.,angle=0.):
    p=ACTIVE or DEFAULT
    if not p['glyph_outline'] and not p['frame_width']:
        return ORIGINAL(meta,ch,size_px,ss,weight,width_axis,stroke,x_scale,slant,angle)
    m=ORIGINAL(meta,ch,size_px,ss,weight,width_axis,stroke,1.,0.,0.)
    if m is None:return None
    if p['glyph_outline']:
        radius=int(p['glyph_outline']);m=np.maximum(0,m-ndi.minimum_filter(m,size=2*radius+1,mode='constant',cval=0))
    im=Image.fromarray((m*255).astype(np.uint8))
    if p['frame_width']:
        pad=int(p['frame_padding']);side=max(im.size)+2*pad+2*int(p['frame_width'])
        canvas=Image.new('L',(side,side));canvas.paste(im,((side-im.width)//2,(side-im.height)//2))
        ImageDraw.Draw(canvas).rectangle((0,0,side-1,side-1),outline=255,width=int(p['frame_width']));im=canvas
    im=im.resize((im.width*ss,im.height*ss),Image.Resampling.BICUBIC)
    w,h=im.size
    if x_scale!=1 or slant!=0:
        sx=max(.3,x_scale);nw=int(math.ceil(w*sx+abs(slant)*h))+2
        im=im.transform((nw,h),Image.Transform.AFFINE,(1/sx,-slant/sx,(-nw/2+slant*h/2)/sx+w/2,0,1,0),resample=Image.Resampling.BICUBIC)
    if angle:im=im.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True)
    m=np.asarray(im.resize((max(1,round(im.width/ss)),max(1,round(im.height/ss))),Image.Resampling.BOX),np.float32)/255
    y,x=np.nonzero(m>.05)
    return m[y.min():y.max()+1,x.min():x.max()+1] if len(x) else None

def install():C.install();C.R.T.glyph_mask=glyph_mask

def render(item):
    global ACTIVE
    recipe,p=item;p=expand(p);r=copy.deepcopy(recipe);ts=r['text_spec'];applied=copy.deepcopy(p)
    mapping={'size':'size','width':'x_scale','spacing':'spacing','weight':'weight','bold':'bold_stroke',
        'slant':'slant','size_jitter':'size_jitter','curve':'curve'}
    for k,field in mapping.items():
        value=p[k+'_absolute']
        if value is not None:
            ts[field]=value
            if k in ['size','width']:applied[k]=1.
            elif k in ['spacing','weight']:applied[k]=0.
            elif k=='bold':applied['bold_extra']=0
    if p['layout'] is not None:
        if p['layout'] not in C.R.T.LAYOUTS:raise ValueError('Unsupported layout')
        ts['layout']=p['layout']
        if p['layout']!='curve':ts['curve']=0.
    if p['charset'] is not None:
        if p['charset'] not in C.R.T.CHARSETS:raise ValueError('Unsupported charset')
        length=int(p['length'] or len(ts['text']))
        ts['text']=C.R.T.make_text(np.random.default_rng(r['seed']+719),length,p['charset'])
        ts['charset']=p['charset'];r['description']['content']=p['charset']
    ACTIVE=p
    try:row=C.render((r,applied))
    finally:ACTIVE=None
    row['controls']=p;row['glyph_decoration']=dict(outline=p['glyph_outline'],frame_width=p['frame_width'],frame_padding=p['frame_padding'])
    return row

def safe_render(item):
    try:return render(item)
    except Exception as exc:return dict(error=repr(exc),complete=False,usable=False,strict=False)
