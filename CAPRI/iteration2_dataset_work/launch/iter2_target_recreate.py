"""Frozen epoch-15 descriptions -> own renderer; no real answers used."""
import sys,json,hashlib,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch,yaml
from PIL import Image,ImageDraw
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT.parent/'captcha-effects'))
from feature_bank.iter2.model import Iter2Model
from feature_bank.iter2.generate import effect_heads,TEXT_HEADS,CATEGORIES,STROKES
from feature_bank.iter2.metrics import decode
from feature_bank.train.data import TOK_W,MARGIN
from captcha_effects import text as T,render as R
from captcha_effects.effects import REGISTRY
from captcha_effects.captcha import render_captcha
from feature_bank.iter2.balanced import transformed_weight

out=Path('C:/Users/pranav/Documents/ChatGPT/thesis/iter2_target_recreation_01');out.mkdir(exist_ok=False)
root=PROJECT/'runs/iter2_baseline_15_20261003_04'
checkpoint=root/'baseline_training/epoch_15/ema.pt'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
model=Iter2Model(**yaml.safe_load((root/'training_config.yaml').read_text())['model']).cpu().eval()
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.requires_grad_(False)
bank=T.FontBank();effects=effect_heads()
families={'CryptBB':Path('E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023'),
          'City':Path('E:/pranav/City (live)-20260503T045450Z-3-001/City (live)'),
          'DarkBay':Path('E:/pranav/cryptbb_sourcefree/data/darkbay/pool')}
rows=[];start=time.perf_counter()
for family,directory in families.items():
    paths=sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
    selected=np.random.default_rng(42).choice(len(paths),2,replace=False)
    for j,index in enumerate(selected):
        path=paths[int(index)];original=Image.open(path).convert('RGB');rng=np.random.default_rng(1700+len(rows))
        w=min(480,round(original.width*96/original.height));im=np.asarray(original.resize((w,96),Image.Resampling.BILINEAR),dtype=np.uint8)
        cols=(w+7)//8;wp=((cols*8+MARGIN+7)//8)*8
        pixels=np.pad(im,((0,0),(0,wp-w),(0,0)),mode='edge').transpose(2,0,1).copy()[None]
        with torch.inference_mode():
            pred=model(torch.from_numpy(pixels).float()/255,torch.tensor([cols]))
        p={k:v[0].numpy() for k,v in pred.items()}
        ep=1/(1+np.exp(-p['eff_presence']));tp=1/(1+np.exp(-p['txt_presence']))
        el=(p['eff_ordinal']>0).sum(-1);tl=(p['txt_ordinal']>0).sum(-1)
        ef={h:dict(probability=round(float(ep[i]),4),present=bool(ep[i]>.5),level=int(el[i])) for i,h in enumerate(effects)}
        tf={h:dict(probability=round(float(tp[i]),4),present=bool(tp[i]>.5),level=int(tl[i])) for i,h in enumerate(TEXT_HEADS)}
        n=int(p['length'].argmax())+2;category=CATEGORIES[int(p['font_cat'].argmax())];stroke=STROKES[int(p['font_stroke'].argmax())]
        weight,width,xh,slant=[int(v) for v in (p['font_ordinal']>0).sum(-1)]
        italic=bool(p['font_italic']>0)
        contentheads=['digits_only','uppercase_only','lowercase_only','mixed_case_alnum']
        content=['digits','upper','lower','mixed'][int(np.argmax([tf[h]['probability'] for h in contentheads]))]
        fresh=T.make_text(rng,n,content)
        fonts=bank.pool('train',category=category,caseless=content in ('digits','upper'))
        rank=lambda f: (int(bool(f.get('italic'))!=italic),int(f['stroke']!=stroke),abs((T.x_height_level(f) if T.x_height_level(f) is not None else 5)-xh))
        fonts=sorted(fonts,key=rank)
        candidates=fonts[:8];settings=[]
        for f in candidates:
            xs=float(np.clip((.38+.077*width)/(f.get('width_rel') or .8),.6,1.45))
            applied_slant=.04*slant
            fit=transformed_weight(f,fresh,original.height*.62,weight,xs,applied_slant)
            if fit is not None:settings.append((fit[0],f,xs,fit))
        if not settings:raise RuntimeError('No usable own-bank font')
        _,font,xs,(_,frac,bold)=min(settings,key=lambda x:x[0])
        keys=[font['key']]
        if tf['font_mixture']['present'] and len(fonts)>1:keys.append(next(f['key'] for f in fonts if f['key']!=font['key']))
        spec=T.sample_text_spec(rng,bank,length=n,charset=content,layout='line',font_mix=1)
        spec.update(text=fresh,fonts=keys,size=.62,size_jitter=0.,rotation=0.,jitter=0.,spacing=.02,curve=0.,slant=.04*slant,
                    multicolour=tf['multicolor_text']['present'],palette='contrast',weight=frac,bold_stroke=bold,x_scale=xs)
        on=lambda h:tf[h]['present']
        lv=lambda h:max(1,tf[h]['level'])
        if on('per_char_rotation'):spec['rotation']=3+4.2*lv('per_char_rotation')
        if on('vertical_jitter'):spec['jitter']=.015*lv('vertical_jitter')
        if on('size_variation'):spec['size_jitter']=.04*lv('size_variation')
        if on('text_on_curve'):spec.update(layout='curve',curve=.02*lv('text_on_curve'))
        if on('two_layer'):spec.update(layout='two_layer',size=.45)
        if on('overlap_tight_spacing'):spec['spacing']=-.025*lv('overlap_tight_spacing')
        if on('wide_spacing'):spec['spacing']=.1*lv('wide_spacing')
        if on('condensed'):spec['x_scale']=1-.035*lv('condensed')
        elif on('expanded'):spec['x_scale']=1+.04*lv('expanded')
        es=dict(background='random',effects=[])
        for h,v in ef.items():
            if v['present']:
                name=next(name for name in R.SAMPLED if REGISTRY[name]['head']==h)
                es['effects'].append(dict(name=name,level=max(1,v['level'])))
        # No image-based colour/layout fitting or family-specific overrides.
        generated=render_captcha(rng,bank,original.size,effects_spec=es,text_spec=spec)
        if generated is None:raise RuntimeError('Renderer could not draw the inferred recipe')
        recreated=Image.fromarray(generated['image'])
        prefix=f'{family}_{j:02d}';original.save(out/f'{prefix}_original.png');recreated.save(out/f'{prefix}_render.png')
        scale=2 if original.width>250 else 3
        widthpanel=max(600,original.width*scale)
        panel=Image.new('RGB',(widthpanel,original.height*scale*2+105),'white');draw=ImageDraw.Draw(panel)
        draw.text((8,5),f'{family} real reference (top); predicted-description recreation (bottom)',fill='black')
        panel.paste(original.resize((original.width*scale,original.height*scale),Image.Resampling.NEAREST),(0,25))
        y=original.height*scale+48
        draw.text((8,y-18),f'Predicted length {n}; fresh text {fresh}; font class {category}',fill='black')
        panel.paste(recreated.resize((original.width*scale,original.height*scale),Image.Resampling.NEAREST),(0,y))
        draw.text((8,y+original.height*scale+8),'Epoch 15 EMA; colours/positions use renderer defaults; no answer labels',fill='black')
        panel.save(out/f'{prefix}_comparison.png')
        row=dict(family=family,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),length=n,
                 ctc_diagnostic=decode(p['ctc'][None],np.asarray([cols]))[0],effects=ef,text_features=tf,
                 font=dict(category=category,stroke=stroke,weight=weight,width=width,x_height=xh,slant=slant,italic=italic,chosen_family=font['family']),
                 rendered_text=fresh,effects_spec=es,text_spec=spec,actual_rendered_effects=generated['effects'],render_usable=generated['text']['usable'],comparison=str(out/f'{prefix}_comparison.png'))
        rows.append(row)
        print(json.dumps(dict(family=family,index=j,length=n,font=category,effects=[h for h,v in ef.items() if v['present']],render_usable=row['render_usable'])),flush=True)
report=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),epoch=15,seed=42,real_labels_used=False,device='cpu',seconds=time.perf_counter()-start,
    mapping_notes=['Presence uses frozen .5 threshold. Ordinal levels count logits above zero.','Effect level is passed as renderer requested level; actual measured level may differ.','An active head with level zero uses minimum render level 1, explicitly recorded.','Content uses highest of four content-head probabilities. Fresh text uses independent predicted length.','Choose own-bank font nearest category/italic/stroke/x-height, then fit weight/width. No target fonts.','Unpredicted colours, sizes, signed directions, positions and effect subtype use fixed/random defaults.','No family rules, real answer labels, training, tuning or image-based recipe optimisation.'],results=rows)
(out/'report.json').write_text(json.dumps(report,indent=2))
print(f'Completed {len(rows)} comparisons in {report["seconds"]:.1f}s',flush=True)
