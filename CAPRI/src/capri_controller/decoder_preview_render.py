"""Explicit approximate decoder-to-renderer preview, without controller fitting.

This adapter does not invert the renderer. Predicted observed intensities/colours
are reused as requested controls; no claim of calibrated reconstruction follows.
"""
import copy
import string
import numpy as np
from PIL import Image
from .decoder import renderer as R


POLICY = dict(version=1, examples_per_dataset=4, variants=2, seed=81073,
    count='count-head argmax; top-N presence queries positioned by predicted boxes',
    font='top1 retrieved family; lowest stable font key within family; default axes',
    text='fresh random text from predicted composition, restricted to library glyph coverage',
    effects='probability >= .5; rounded 10*predicted intensity, clipped 1..10; fixed exclusivity',
    background='median predicted under-background RGB over fully valid cells; fresh procedural effects',
    defaults='zero rotation/shear/stroke; glyph masks resized to predicted boxes; no source pixels copied',
    limitations='Observed boxes/colours/levels are not latent renderer controls. Effects may transform boxes again. '
        'No controller refinement, no per-scheme settings, no real labels, no retry or attractive-example selection.')


def recipe(pred, fonts, seed):
    rng=np.random.default_rng(seed)
    families={}
    for f in sorted(fonts.values(),key=lambda f:f['key']):families.setdefault(f['family_group'],f)
    queries=sorted(pred['queries'],key=lambda g:(-g['presence'],g['query_id']))[:pred['count']]
    queries.sort(key=lambda g:(g['box_xyxy'][0]+g['box_xyxy'][2])/2)
    pools=[string.digits,string.ascii_uppercase,string.ascii_lowercase]
    composition=pred['composition_probabilities']
    pool=''.join(p for p,v in zip(pools,composition) if v>=.5) or pools[int(np.argmax(composition))]
    chosen=[families[q['nearest_fonts'][0]['family']] for q in queries]
    text=''.join(str(rng.choice(list(''.join(c for c in pool if c in f['chars']) or f['chars']))) for f in chosen)
    candidates=[(name,pred['effect_probabilities'][R.REGISTRY[name]['head']]) for name in R.R.SAMPLED]
    effects=[];dropped=[]
    for name,p in sorted(candidates,key=lambda x:(-x[1],x[0])):
        if p<.5:continue
        groups=[('bold','thin','hollow'),('wave','morph','perspective','swirl'),('pixelate','resample')]
        if any(name in g and any(e['name'] in g for e in effects) for g in groups):dropped.append(name);continue
        level=int(np.clip(np.rint(10*pred['effect_intensity'][R.REGISTRY[name]['head']]),1,10))
        effects.append(dict(name=name,level=level))
    h,w=pred['geometry']['native_hw']
    return dict(schema='capri-decoder-preview-v1',render_seed=seed,size=[w,h],background=pred['background_rgb'],
        effects=effects,dropped_exclusive_effects=dropped,requested_count=pred['count'],
        text=dict(text=text,font_keys=[f['key'] for f in chosen],axes=[{a['tag']:a['default'] for a in f['axes']} for f in chosen],
                  colours=[q['colour_rgb'] for q in queries],target_boxes=[q['box_xyxy'] for q in queries],
                  query_ids=[q['query_id'] for q in queries],layout='predicted_boxes'),adapter=POLICY)


def draw_predicted(cv,spec,fonts,rng):
    """Render independent glyphs into predicted boxes, preserving instance records."""
    result=[]
    for i,char in enumerate(spec['text']):
        meta=fonts[spec['font_keys'][i]]
        b=np.asarray(spec['target_boxes'][i]);x0,y0=map(int,np.floor(b[:2]));x1,y1=map(int,np.ceil(b[2:]))
        # A model box outside the canvas remains a recorded limitation, not a fitted correction.
        width=max(1,x1-x0);height=max(1,y1-y0)
        mask,axes=R.glyph(meta,char,max(4,height),spec['axes'][i],1.,0.,0.,0)
        mask=np.asarray(Image.fromarray(mask.astype(np.float32),mode='F').resize((width,height),Image.Resampling.BILINEAR))
        coverage=np.zeros((cv.h,cv.w),np.float32)
        ax,ay=max(0,x0),max(0,y0);bx,by=min(cv.w,x1),min(cv.h,y1)
        if bx>ax and by>ay:coverage[ay:by,ax:bx]=mask[ay-y0:by-y0,ax-x0:bx-x0]
        item=dict(index=i,char=char,row=0,font_key=meta['key'],font_sha256=meta['sha256'],family=meta['family'],
            family_group=meta['family_group'],style=meta['style'],axes=axes,axis_application='success',angle=0.,shear=0.,stretch=1.,
            size_px=height,colour=spec['colours'][i],italic=meta['italic'],category=meta['category'],stroke_class=meta['stroke'],
            x_height=meta['x_height'],intrinsic_width=meta['width_rel'],position=[int(x0),int(y0)],
            clipped_fraction=float(1-coverage.sum()/max(1e-6,mask.sum())),fit_scale=1.,stroke_width=R.T.stroke_width_px(mask),
            preview_box_resize=[width,height],source_query_id=spec['query_ids'][i])
        item['id']=cv.blend(coverage,item['colour'],'text','text','text',params=copy.deepcopy(item));result.append(item)
    return result


def render_preview(value,fonts):
    original=R.draw_instances
    R.draw_instances=draw_predicted
    try:return R.render(value,fonts)
    finally:R.draw_instances=original
