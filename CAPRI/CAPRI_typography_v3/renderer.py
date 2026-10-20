"""General-purpose procedural mapping. No family-specific branches or assets."""
import math
import hashlib
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def render(style,text,font_path,seed,variant=0):
    rng=np.random.default_rng(seed);w,h=style['size'];bg=style['background_palette'];fg=style['foreground_palette']
    image=Image.new('RGB',(w,h),tuple(bg[0]))
    if style.get('gradient_strength',0)>30:
        t=np.linspace(0,1,w)[None,:,None]
        ramp=np.array(style['gradient_left'])[None,None,:]*(1-t)+np.array(style['gradient_right'])[None,None,:]*t
        image=Image.fromarray(np.repeat(ramp,h,axis=0).clip(0,255).astype('uint8'))
    d=ImageDraw.Draw(image)
    operations=[]
    evidence=style.get('background_block_evidence',{})
    # Separate stream prevents block count from changing downstream random draws.
    block_rng=np.random.default_rng(np.random.SeedSequence([int(seed),197]))
    if evidence.get('state')=='supported':
        candidates=evidence.get('candidates',[])
        for _ in range(min(35,int(evidence['count']))):
            item=candidates[int(block_rng.integers(len(candidates)))];bw=max(2,round(w*item['width_fraction']));bh=max(2,round(h*item['height_fraction']));x=int(block_rng.integers(max(1,w-bw+1)));y=int(block_rng.integers(max(1,h-bh+1)));color=item['color']
            d.rectangle((x,y,min(w-1,x+bw-1),min(h-1,y+bh-1)),fill=tuple(color));operations.append(dict(kind='block',box=[x,y,min(w,x+bw),min(h,y+bh)],color=color,evidence='measured rectangle geometry/color; randomized placement'))
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),201]))
    density=style['background_edge_fraction'];linecount=int(np.clip(density*w*h/(max(w,h)*2)*style.get('line_count_gain',1),0,60))
    learned=style.get('classifier_features')
    if learned is not None and learned.get('straight_interference')!='present':linecount=0
    if 'straight_line_cap' in style:linecount=min(linecount,max(0,int(style['straight_line_cap'])))
    hist=np.array(style['line_angle_histogram']);angles=[]
    if hist.sum()>0:
        for _ in range(linecount):
            measured=style.get('lines',[])
            angle=(float(measured[int(rng.integers(len(measured)))]['angle']) if measured else (int(rng.choice(12,p=hist/hist.sum()))+.5)*15)+float(rng.uniform(-2,2));angles.append(angle)
            rad=math.radians(angle);x=float(rng.uniform(0,w));y=float(rng.uniform(0,h));length=float(rng.uniform(w*.3,w*1.2))
            color=np.array(bg[int(rng.integers(len(bg)))],float)
            contrast=40 if style['dark_background'] else -40;color=np.clip(color+contrast,0,255).astype(int).tolist()
            ends=[round(x-length*math.cos(rad)/2),round(y-length*math.sin(rad)/2),round(x+length*math.cos(rad)/2),round(y+length*math.sin(rad)/2)]
            width=1 if variant%2==0 else max(1,round(h*.035));width=max(1,round(width*style.get('line_width_gain',1)));d.line(ends,fill=tuple(color),width=width)
            operations.append(dict(kind='line',points=ends,color=color,width=width))
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),202]))
    if learned is not None and learned.get('curved_interference')=='present':
        # Classification supplies presence only. Geometry is an explicitly
        # generic adapter, not an inferred curve-density or curvature label.
        for _ in range(int(np.clip(round(2*style.get('curve_count_gain',1)),0,10))):
            cy=float(rng.uniform(.2,.8)*h);amp=float(rng.uniform(.05,.15)*h)*style.get('curve_amplitude_gain',1);phase=float(rng.uniform(0,2*math.pi));amp=style.get('pred_curve_amplitude',amp)
            points=[(int(x),round(cy+amp*math.sin(2*math.pi*x/max(w-1,1)+phase))) for x in np.linspace(0,w-1,40)]
            color=np.clip(np.array(bg[0])+ (40 if style['dark_background'] else -40),0,255).astype(int).tolist()
            curve_width=max(1,round(style.get('pred_curve_width',1)));d.line(points,fill=tuple(color),width=curve_width);operations.append(dict(kind='classifier_curve',points=points,color=color,width=curve_width,geometry='learned amplitude/width when supplied; generic position'))
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),203]))
    count=int(np.clip(max(style['small_component_density']*2,style.get('texture_fraction',0)*(.6 if variant%2 else 1.0))*w*h*style.get('dot_count_gain',1),0,w*h*.35))
    if learned is not None and learned.get('dot_scatter')!='present':count=0
    for _ in range(count):
        x=int(rng.integers(w));y=int(rng.integers(h));color=bg[int(rng.integers(len(bg)))]
        noise=rng.normal(0,35,3)
        if style.get('achromatic_noise',False):noise[:]=noise.mean()
        color=np.clip(np.array(color)+noise,0,255).astype(int).tolist();d.point((x,y),fill=tuple(color))
        operations.append(dict(kind='point',xy=[x,y],color=color))
    if style.get('multilabel_grid',False):
        spacing=max(6,int(style.get('multilabel_grid_spacing',12)));color=tuple(np.clip(np.array(bg[0],float)+(35 if style['dark_background'] else -35),0,255).astype(int));d=ImageDraw.Draw(image)
        for gx in range(0,w,spacing):d.line((gx,0,gx,h-1),fill=color,width=1)
        for gy in range(0,h,spacing):d.line((0,gy,w-1,gy),fill=color,width=1)
        operations.append(dict(kind='multilabel_grid',spacing=spacing,color=color,geometry='generic spacing, not estimated'))
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),204]))
    background=image.copy();textmask=Image.new('L',(w,h));records=[];layers=[]
    proposals=style['glyph_proposals'];height=float(np.clip(style['text_height'],.25,.75))
    span=float(np.clip(.76*style.get('spacing_gain',1),.4,.90));centers=np.linspace(.5-span/2,.5+span/2,len(text))
    controls=style.get('_typography', [{} for _ in text])
    if len(controls)!=len(text):raise ValueError('Typography/label mismatch')
    for i,char in enumerate(text):
        tc=controls[i]
        glyph_font=font_path[i] if isinstance(font_path,(list,tuple)) else font_path
        glyph_height=height*tc.get('height_gain',1.)
        g=proposals[round(i*(len(proposals)-1)/max(1,len(text)-1))] if proposals else None
        cy=g['center'][1] if g else .5
        font=ImageFont.truetype(glyph_font,max(8,round(h*glyph_height)))
        box=font.getbbox(char);glyph=Image.new('L',(box[2]-box[0]+8,box[3]-box[1]+8));ImageDraw.Draw(glyph).text((4-box[0],4-box[1]),char,font=font,fill=255)
        crop=glyph.getbbox();glyph=glyph.crop(crop)
        targeth=max(6,round(h*glyph_height));width_gain=style.get('glyph_width_gain',1.)*tc.get('width_gain',1.)
        targetw=max(2,min(round(glyph.width*targeth/max(glyph.height,1)*width_gain),round(w*(.75 if width_gain==1. else .9)/len(text))))
        glyph=glyph.resize((targetw,targeth),Image.Resampling.BILINEAR)
        effect=tc.get('effect',style.get('font_effect'))
        hollow=(style.get('hollow_score',0)>.55 and variant>=2) if effect is None else effect=='hollow'
        if effect in ('bold','hollow'):
            glyph=Image.fromarray(np.pad(np.asarray(glyph),2))
        if effect=='bold':glyph=Image.fromarray(cv2.dilate(np.asarray(glyph),np.ones((3,3),np.uint8)))
        if hollow:
            array=np.asarray(glyph);eroded=cv2.erode(array,np.ones((3,3),np.uint8));glyph=Image.fromarray(array-eroded)
        # Rotation stays conservative because measured glyph axes are ambiguous.
        tilt=float(np.clip(style['text_tilt_proxy'] or 0,0,25));angle=float(rng.uniform(-tilt,tilt)) if variant>=2 else 0
        angle=float(np.clip(angle+tc.get('angle_delta',0.),-30,30))
        if tc.get('stroke',0)>0:glyph=Image.fromarray(cv2.dilate(np.pad(np.asarray(glyph),1),np.ones((2,2),np.uint8)))
        glyph=glyph.rotate(angle,Image.Resampling.BILINEAR,expand=True)
        left=1 if i==0 else int(math.ceil((centers[i-1]+centers[i])*w/2))
        right=w-1 if i==len(text)-1 else int(math.floor((centers[i]+centers[i+1])*w/2))
        if right-left<2 or h<4:raise ValueError('Canvas too small for requested text')
        fit_scale=min(1.,(right-left)/glyph.width,(h-2)/glyph.height)
        if fit_scale<1:
            glyph=glyph.resize((max(1,int(glyph.width*fit_scale)),max(1,int(glyph.height*fit_scale))),Image.Resampling.BILINEAR)
        x=round(centers[i]*w-glyph.width/2);y=int(np.clip(round((cy+tc.get('vertical_delta',0.))*h-glyph.height/2),1,max(1,h-glyph.height-1)))
        x=int(np.clip(x,left,right-glyph.width))
        if x<0 or y<0 or x+glyph.width>w or y+glyph.height>h:raise ValueError('Glyph would clip')
        mask=Image.new('L',(w,h));mask.paste(glyph,(x,y));support=np.asarray(mask)>100
        if not support.any():raise ValueError('Glyph mask has no significant pixels')
        backdrop=np.asarray(background)[support].astype(float)
        alpha=np.asarray(mask)[support].astype(float)/255
        def contrast_score(color):
            return float((np.max(np.abs(backdrop-np.array(color)),axis=1)*alpha>=24).mean())
        choices=[c for c in fg if contrast_score(c)>=.75]
        fallback=False
        if not choices:
            # Record a readability fallback instead of silently accepting an
            # invisible glyph; synthetic labels still refer to the same text.
            choices=[max(([0,0,0],[255,255,255]),key=contrast_score)];fallback=True
        color=choices[int(rng.integers(len(choices)))]
        image.paste(tuple(color),mask=mask);textmask=Image.fromarray(np.maximum(np.asarray(textmask),np.asarray(mask)))
        layers.append((color,mask))
        records.append(dict(font=glyph_font,typography=tc,character=char,box=[x,y,x+glyph.width,y+glyph.height],angle=angle,color=color,hollow=hollow,font_effect=effect,contrast_fallback=fallback,rotation_fit_scale=fit_scale))
    delta=np.max(np.abs(np.asarray(image).astype(float)-np.asarray(background).astype(float)),axis=2)
    visible=float((delta[np.asarray(textmask)>100]>=12).mean())
    if not np.isfinite(visible) or visible<.55:raise ValueError('Low visible text contribution')
    # Compare with an exact counterfactual omitting each glyph, so other glyphs
    # cannot hide one invisible character in a union-mask score.
    final=np.asarray(image).astype(float)
    for i,(_,mask) in enumerate(layers):
        counterfactual=background.copy()
        for j,(color,other_mask) in enumerate(layers):
            if i!=j:counterfactual.paste(tuple(color),mask=other_mask)
        contribution=np.max(np.abs(final-np.asarray(counterfactual).astype(float)),axis=2)
        fraction=float((contribution[np.asarray(mask)>100]>=12).mean())
        records[i]['visible_fraction']=fraction
        if fraction<.55:raise ValueError('Low individual glyph contribution')
    if style.get('multilabel_blur',False):
        from PIL import ImageFilter
        image=image.filter(ImageFilter.GaussianBlur(.6));operations.append(dict(kind='multilabel_blur',radius=.6,geometry='generic strength, not estimated'))
    return image,textmask,dict(text=text,font=font_path,seed=seed,variant=variant,operations=operations,glyphs=records,visible_fraction=visible,
        background_sha256=hashlib.sha256(background.tobytes()).hexdigest(),
        mapping=['Pillow.ImageDraw.rectangle','Pillow.ImageDraw.line','Pillow.ImageDraw.point','Pillow.ImageFont.truetype','Pillow.Image.paste'])
