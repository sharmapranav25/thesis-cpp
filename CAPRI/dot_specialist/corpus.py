import hashlib,json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter

ROOT=Path('E:/pranav/Final sem/sourcefree_synth');RELEASE=ROOT/'releases/phase12_v2'
SPLITS={'train':(91000000,6000),'selection':(92000000,1000),'calibration':(93000000,1000),'locked_same':(94000000,500),'locked_font':(95000000,500),'locked_shape':(96000000,500),'locked_raster':(97000000,500)}

def fonts():
    f=json.loads((RELEASE/'assets/fonts.json').read_text())[:9]
    for a in f:assert hashlib.sha256((RELEASE/a['path']).read_bytes()).hexdigest()==a['sha256']
    return f

def pop(images):
    result=[]
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    for im in images:
        gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        bright=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel)
        dark=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
        result.append(np.maximum(bright,dark))
    return np.stack(result)

def draw_mask(points,width,raster):
    if raster=='opencv':
        a=np.zeros((64,128),np.uint8);cv2.polylines(a,[np.rint(points).astype(np.int32)],False,255,width,cv2.LINE_8);return a
    m=Image.new('L',(128,64));ImageDraw.Draw(m).line([tuple(p) for p in np.rint(points).astype(int)],fill=255,width=width);return np.asarray(m)

def pair(seed,part,fontbank):
    # All rejection attempts remain traceable; both pair members reuse every nuisance pixel.
    for attempt in range(100):
        rng=np.random.default_rng(np.random.SeedSequence([seed,attempt]));density_bin=seed%12
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
        for _ in range(int(rng.integers(0,4))):d.line(tuple(rng.integers([0,0,0,0],[128,64,128,64])),fill=tuple(rng.integers(0,256,3)),width=int(rng.integers(1,3)))
        text=Image.new('RGBA',(128,64));fontrec=fontbank[int(rng.integers(len(fontbank)))];font=ImageFont.truetype(str(RELEASE/fontrec['path']),int(rng.integers(18,37)))
        chars=''.join(rng.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'),int(rng.integers(2,9))))
        for c,x in zip(chars,np.linspace(8,113,len(chars))):
            glyph=Image.new('RGBA',(48,48));ImageDraw.Draw(glyph).text((3,0),c,font=font,fill=tuple(rng.integers(0,256,3))+(255,));glyph=glyph.rotate(float(rng.uniform(-25,25)),Image.Resampling.BILINEAR);text.alpha_composite(glyph,(int(x)-10,int(rng.integers(0,15))))
        # Dot positives versus compact dash negatives; no scatter in common backdrop.
        count=int(rng.integers(6+(density_bin*43)//12,6+((density_bin+1)*43)//12));radius=int(rng.integers(1,4));recipe='clustered_disks' if part=='locked_shape' else 'uniform_disks'
        raster='opencv' if part=='locked_raster' else 'pillow';cm=np.zeros((64,128),np.uint8);sm=np.zeros_like(cm)
        circles=Image.fromarray(cm);lines=Image.fromarray(sm);dc=ImageDraw.Draw(circles);dl=ImageDraw.Draw(lines)
        cluster=rng.uniform([12,12],[116,52]);centers=[]
        for _ in range(count):
            center=(np.clip(cluster+rng.normal(0,[13,8]),[4,4],[123,59]) if part=='locked_shape' else rng.uniform([4,4],[123,59]));x,y=np.rint(center).astype(int);centers.append([int(x),int(y)])
            half=max(2,round(1.5*radius));angle=float(rng.uniform(0,np.pi));dx,dy=np.cos(angle)*half,np.sin(angle)*half;ends=[(round(x-dx),round(y-dy)),(round(x+dx),round(y+dy))]
            if raster=='opencv':cv2.circle(cm,(int(x),int(y)),radius,255,-1,cv2.LINE_8);cv2.line(sm,ends[0],ends[1],255,max(1,radius),cv2.LINE_8)
            else:dc.ellipse((x-radius,y-radius,x+radius,y+radius),fill=255);dl.line(ends,fill=255,width=max(1,radius))
        if raster=='pillow':cm=np.asarray(circles);sm=np.asarray(lines)
        negative_kind='dash' if rng.random()<.8 else 'no_added_marks'
        if negative_kind=='no_added_marks':sm=np.zeros_like(sm)
        ink=tuple(rng.integers(0,256,3));front=bool(rng.integers(2));blur=float(rng.choice([0,0,0,.4,.7]))
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
        if visible.sum()<24 or fraction<.4:continue
        if negative_kind=='dash' and (negvis.sum()<24 or negvis.sum()/max(1,(sm>0).sum())<.4):continue
        record=dict(group_seed=seed,attempt=attempt,part=part,density_bin=density_bin,recipe=recipe,rasterizer=raster,radius=radius,count=count,negative_kind=negative_kind,centers=centers,font=fontrec['family'],text=chars,front=front,blur=blur,background_kind=kind,visible_fraction=fraction,visible_pixels=int(visible.sum()),contrast=float(np.median(delta[visible])),dot_pixels=support,negative_pixels=int((sm>0).sum()))
        return np.stack([negative,positive]),record
    raise RuntimeError(f'Failed valid dot pair {seed} after100 attempts')

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
        assert all(r['visible_pixels']>=24 and r['visible_fraction']>=.4 for r in records)
        hist=np.bincount([r['density_bin'] for r in records],minlength=12);assert hist.min()>0
        audit[part]=dict(groups=count,density_counts=hist.tolist(),max_rejection_attempt=max(r['attempt'] for r in records))
    (out/'corpus_audit.json').write_text(json.dumps(dict(passed=True,group_seeds_disjoint=True,pixel_duplicates=0,images=len(pixel_groups),splits=audit),indent=2));return audit
