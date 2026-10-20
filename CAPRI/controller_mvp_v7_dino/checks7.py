from setup7 import ROOT,U,WORK,random_recipes
from perception7 import Comparator,Objective7,WEIGHTS
import render7 as R
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
import copy

def main():
    fonts=U.read(WORK/'learned_controller_20261004/fonts.json');recipes=random_recipes(72001,24,[256,64],fonts)
    R.install();neutral=R.expand(R.C.R.NEUTRAL)
    for recipe in recipes[:4]:
        old=R.C.render((recipe,R.C.expand(R.C.R.NEUTRAL)));new=R.render((recipe,neutral))
        assert old['pixel_sha256']==new['pixel_sha256']
    p={**neutral,'charset':'digits','length':5,'layout':'line','size_absolute':.6,'width_absolute':.9,
        'spacing_absolute':.1,'weight_absolute':.4,'bold_absolute':0,'slant_absolute':.2,'size_jitter_absolute':.1,
        'curve_absolute':0.,'glyph_outline':1,'frame_width':1}
    row=R.render((recipes[0],p));assert row['text'].isdigit() and len(row['text'])==5
    for k,v in dict(size=.6,x_scale=.9,spacing=.1,weight=.4,bold_stroke=0,slant=.2,size_jitter=.1,curve=0.).items():
        assert row['text_spec'][k]==v,(k,row['text_spec'][k],v)
    same=R.render((recipes[0],p));assert same['pixel_sha256']==row['pixel_sha256']
    unframed=R.render((recipes[0],{**p,'frame_width':0}));assert unframed['pixel_sha256']!=row['pixel_sha256']
    # Generic synthetic probe only; no target-family pixels tune this implementation.
    rows=[R.render((r,neutral)) for r in recipes];model=Comparator();features=model.extract([r['image'] for r in rows])
    truth=np.stack([np.asarray(Image.fromarray(r['mask']).resize((256,64),Image.Resampling.NEAREST))>127 for r in rows])
    iou=lambda m:float(np.mean((m&truth).sum((1,2))/np.maximum(1,(m|truth).sum((1,2)))))
    obj=Objective7(features,features);assert obj.score(features,features,rows)['appearance']<1e-6
    assert all(np.isfinite(features[k]).all() for k in WEIGHTS)
    # A clean controlled text reference must not prefer its blank/blurred variants.
    clean={**neutral,'font':fonts[0]['key'],'background':[255,255,255],'text_palette':[[0,0,0]],'color_mode':'single',
        'effects':{n:0 for n in R.C.R.R.SAMPLED},'charset':'upper','length':5,'size_absolute':.7,'layout':'line',
        'rotation_absolute':0,'jitter_absolute':0,'size_jitter_absolute':0,'slant_absolute':0,'bold_absolute':0}
    cleanrows=[R.render((r,clean)) for r in recipes[:8]];pixels=[r['image'] for r in cleanrows]
    f=model.extract(pixels);o=Objective7(f,features)
    blank=model.extract([np.full_like(im,255) for im in pixels]);blur=model.extract([np.asarray(Image.fromarray(im).filter(ImageFilter.GaussianBlur(3))) for im in pixels])
    scores={'self':o.score(f,f,cleanrows),'blank':o.score(blank,f,cleanrows),'blur':o.score(blur,f,cleanrows)}
    assert scores['blank']['objective']>scores['self']['objective']+.1
    assert scores['blur']['objective']>scores['self']['objective']+.1
    assert not any(p.requires_grad for p in model.model.parameters())
    sheet=Image.new('RGB',(768,24+64*8),'white');draw=ImageDraw.Draw(sheet)
    for x,s in [(0,'Synthetic image'),(256,'DINO attention'),(512,'Pixel + DINO region')]:draw.text((x+4,5),s,fill='black')
    for i in range(8):
        sheet.paste(Image.fromarray(rows[i]['image']).resize((256,64)),(0,24+64*i))
        for j,k in [(1,'focus'),(2,'region_mask')]:sheet.paste(Image.fromarray((features[k][i]*255).astype(np.uint8)).convert('RGB'),(256*j,24+64*i))
    sheet.save(ROOT/'synthetic_regions.png');Image.fromarray(row['image']).save(ROOT/'frame_control_check.png')
    U.dump(ROOT/'checks.json',dict(passed=True,neutral_pixel_equivalence=True,absolute_controls_verified=True,deterministic=True,
        frame_changes_pixels=True,model_frozen=True,self_distance_zero=True,blank_and_blur_rejected=True,
        synthetic_probe_count=24,raw_attention_iou=iou(features['focus']>.5),new_region_iou=iou(features['region_mask']>.5),
        degradation_scores=scores,note='Generic synthetic probe, not target segmentation accuracy. Frames belong to synthetic glyph masks.'))
    print('Checks passed. Region IoU:',iou(features['focus']>.5),iou(features['region_mask']>.5),flush=True)

if __name__=='__main__':main()
