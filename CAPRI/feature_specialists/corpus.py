import hashlib,json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter

ROOT=Path('E:/pranav/Final sem/sourcefree_synth');RELEASE=ROOT/'releases/phase12_v2'
SPLITS={'train':(81000000,6000),'selection':(82000000,1000),'calibration':(83000000,1000),'locked_same':(84000000,500),'locked_font':(85000000,500),'locked_shape':(86000000,500),'locked_raster':(87000000,500)}

def fonts():
    f=json.loads((RELEASE/'assets/fonts.json').read_text())[:9]
    for a in f:assert hashlib.sha256((RELEASE/a['path']).read_bytes()).hexdigest()==a['sha256']
    return f

def pop(images):
    result=[]
    for im in images:
        gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        mean=cv2.GaussianBlur(gray,(0,0),2.0);var=np.maximum(cv2.GaussianBlur(gray*gray,(0,0),2.0)-mean*mean,0)
        result.append(np.uint8(np.clip(np.abs(gray-mean)/(np.sqrt(var)+.04)/3,0,1)*255))
    return np.stack(result)

def draw_mask(points,width,raster):
    if raster=='opencv':
        a=np.zeros((64,128),np.uint8);cv2.polylines(a,[np.rint(points).astype(np.int32)],False,255,width,cv2.LINE_8);return a
    m=Image.new('L',(128,64));ImageDraw.Draw(m).line([tuple(p) for p in np.rint(points).astype(int)],fill=255,width=width);return np.asarray(m)

def pair(seed,part,fontbank):
    # All rejection attempts remain traceable; both pair members reuse every nuisance pixel.
    for attempt in range(100):
        rng=np.random.default_rng(np.random.SeedSequence([seed,attempt]));angle_bin=seed%12;theta=(angle_bin+rng.uniform(.05,.95))*math.pi/12
        bg=rng.integers(0,256,3);arr=np.broadcast_to(bg,(64,128,3)).copy().astype(float)
        kind=int(rng.integers(4))
        if kind==1:arr+=np.linspace(-45,45,128)[None,:,None]
        canvas=Image.fromarray(np.uint8(np.clip(arr,0,255)));d=ImageDraw.Draw(canvas)
        if kind==2:
            for _ in range(int(rng.integers(1,7))):
                x,y=rng.integers([0,0],[110,50]);d.rectangle((x,y,x+int(rng.integers(5,25)),y+int(rng.integers(5,30))),fill=tuple(rng.integers(0,256,3)))
        if kind==3:
            spacing=int(rng.integers(8,25));color=tuple(rng.integers(0,256,3))
            for x in range(0,128,spacing):d.line((x,0,x,63),fill=color)
            for y in range(0,64,spacing):d.line((0,y,127,y),fill=color)
        for _ in range(int(rng.integers(0,160))):d.point(tuple(rng.integers([0,0],[128,64])),fill=tuple(rng.integers(0,256,3)))
        for _ in range(int(rng.integers(0,4))):d.line(tuple(rng.integers([0,0,0,0],[128,64,128,64])),fill=tuple(rng.integers(0,256,3)),width=int(rng.integers(1,3)))
        text=Image.new('RGBA',(128,64));fontrec=fontbank[int(rng.integers(len(fontbank)))];font=ImageFont.truetype(str(RELEASE/fontrec['path']),int(rng.integers(18,37)))
        chars=''.join(rng.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'),int(rng.integers(2,9))))
        for c,x in zip(chars,np.linspace(8,113,len(chars))):
            glyph=Image.new('RGBA',(48,48));ImageDraw.Draw(glyph).text((3,0),c,font=font,fill=tuple(rng.integers(0,256,3))+(255,));glyph=glyph.rotate(float(rng.uniform(-25,25)),Image.Resampling.BILINEAR);text.alpha_composite(glyph,(int(x)-10,int(rng.integers(0,15))))
        t=np.linspace(-1,1,100);recipe='cubic' if part=='locked_shape' else ['sine','arc','quadratic'][int(rng.integers(3))]
        if recipe=='sine':xy=np.stack([t,.30*np.sin(t*rng.uniform(2,5)+rng.uniform(-2,2))],1)
        elif recipe=='arc':q=np.linspace(-1.1,1.1,100);xy=np.stack([np.sin(q),np.cos(q)-.72],1)
        elif recipe=='quadratic':xy=np.stack([t,rng.uniform(.25,.65)*(1-t*t)],1)
        else:
            u=(t+1)/2;pts=np.array([[-1,0],[-.4,rng.uniform(.5,1)],[.4,rng.uniform(-1,-.5)],[1,0]]);xy=(1-u[:,None])**3*pts[0]+3*(1-u[:,None])**2*u[:,None]*pts[1]+3*(1-u[:,None])*u[:,None]**2*pts[2]+u[:,None]**3*pts[3]
        xy-=xy.mean(0);rotation=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]]);xy=xy@rotation.T;scale=min(float(rng.uniform(24,50)),50/max(np.ptp(xy[:,0]),1e-6),43/max(np.ptp(xy[:,1]),1e-6));xy*=scale
        center=np.array([rng.uniform(32,96),rng.uniform(24,40)]);xy+=center
        xy[:,0]=np.clip(xy[:,0],2,125);xy[:,1]=np.clip(xy[:,1],2,61)
        # A geometric check on source centerline prevents nominal-but-straight positives.
        values=np.linalg.svd(xy-xy.mean(0),compute_uv=False);bend=float(values[1]/math.sqrt(len(xy)))
        if bend<2:continue
        width=int(rng.integers(1,4));raster='opencv' if part=='locked_raster' else 'pillow';cm=draw_mask(xy,width,raster)
        length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum());vec=np.array([math.cos(theta),math.sin(theta)])*min(length/2,44);sm=draw_mask(np.stack([center-vec,center+vec]),width,raster)
        ink=tuple(rng.integers(0,256,3));front=bool(rng.integers(2));blur=float(rng.choice([0,0,0,.4,.7]))
        if not .75<=(cm>0).sum()/max(1,(sm>0).sum())<=1.33:continue
        def compose(mask):
            im=canvas.convert('RGBA')
            if front:im=Image.alpha_composite(im,text)
            im.paste(ink,mask=Image.fromarray(mask))
            if not front:im=Image.alpha_composite(im,text)
            im=im.convert('RGB')
            return np.asarray(im.filter(ImageFilter.GaussianBlur(blur)) if blur else im)
        clean=compose(np.zeros_like(cm));positive=compose(cm);negative=compose(sm)
        delta=np.max(np.abs(positive.astype(float)-clean.astype(float)),2);visible=(delta>=12)&(cm>0);support=int((cm>0).sum());fraction=float(visible.sum()/max(support,1))
        negdelta=np.max(np.abs(negative.astype(float)-clean.astype(float)),2);negvis=(negdelta>=12)&(sm>0)
        if visible.sum()<24 or fraction<.4 or negvis.sum()<24 or negvis.sum()/max(1,(sm>0).sum())<.4:continue
        record=dict(group_seed=seed,attempt=attempt,part=part,angle_bin=angle_bin,axis_degrees=theta*180/math.pi,recipe=recipe,rasterizer=raster,width=width,font=fontrec['family'],text=chars,front=front,blur=blur,background_kind=kind,curve_bend=bend,visible_fraction=fraction,visible_pixels=int(visible.sum()),contrast=float(np.median(delta[visible])),curve_pixels=support,negative_pixels=int((sm>0).sum()))
        return np.stack([negative,positive]),record
    raise RuntimeError(f'Failed valid curve pair {seed} after100 attempts')

def prepare(out,event):
    out=Path(out);bank=fonts();(out/'fonts.json').write_text(json.dumps(bank,indent=2));audit={};allseeds=set();pixel_groups={}
    for part,(seed,count) in SPLITS.items():
        target=out/(part+'.npz');metadata=out/(part+'_records.json')
        if target.exists() and metadata.exists():
            records=json.loads(metadata.read_text());assert len(records)==count
        else:
            chosen=bank[7:] if part=='locked_font' else bank[:7] if part=='selection' else bank[:6]
            images=[];records=[]
            for i in range(count):
                im,r=pair(seed+i,part,chosen);images.extend(im);records.append(r)
                if (i+1)%250==0:event('corpus',part=part,groups=i+1,total=count)
            images=np.stack(images);np.savez_compressed(target,images=images,pop=pop(images),labels=np.tile(np.array([0,1],np.float32),count));metadata.write_text(json.dumps(records))
        for r in records:assert r['group_seed'] not in allseeds;allseeds.add(r['group_seed'])
        with np.load(target) as stored:
            assert np.array_equal(stored['labels'],np.tile(np.array([0,1],np.float32),count))
            for index,im in enumerate(stored['images']):
                digest=hashlib.sha256(im.tobytes()).hexdigest();group=records[index//2]['group_seed']
                assert digest not in pixel_groups,'Duplicate full image across corpus'
                pixel_groups[digest]=group
        assert all(r['visible_pixels']>=24 and r['visible_fraction']>=.4 and r['curve_bend']>=2 for r in records)
        hist=np.bincount([r['angle_bin'] for r in records],minlength=12);assert hist.min()>0
        audit[part]=dict(groups=count,orientation_counts=hist.tolist(),max_rejection_attempt=max(r['attempt'] for r in records))
    (out/'corpus_audit.json').write_text(json.dumps(dict(passed=True,group_seeds_disjoint=True,pixel_duplicates=0,images=len(pixel_groups),splits=audit),indent=2));return audit
