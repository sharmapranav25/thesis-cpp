"""Manually controlled family simulations using captcha-effects, without a model.

Custom geometry and a fixed glyph palette use the renderer's labelled Canvas.
All changes are local to this demo; training code and renderer package are untouched.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RENDERER = Path('E:/pranav/Final sem/captcha-effects')
sys.path.insert(0, str(RENDERER))
from captcha_effects import render as R, styles as S, text as T
from captcha_effects.captcha import legibility

OUT = Path(__file__).resolve().parent
BANK = T.FontBank()


def font(family):
    return next(f['key'] for f in BANK.fonts
                if f['family'] == family and f['split'] == 'train'
                and not f.get('italic', False))


def rectangle(cv, bounds, color, *, head='bg_blocks', fill=True, width=1):
    x0, y0, x1, y1 = bounds
    mask = cv.mask(lambda d, sc: d.rectangle(
        [x0*sc, y0*sc, x1*sc, y1*sc],
        fill=255 if fill else None, outline=None if fill else 255,
        width=max(1, int(width*sc))), antialias=False)
    cv.blend(mask, color, 'manual_rectangle', head, 'under', 1.,
             dict(bounds=list(bounds), filled=fill, width=width))


def line(cv, points, color, width, head, antialias=True):
    mask = cv.mask(lambda d, sc: d.line(
        [(x*sc, y*sc) for x, y in points], fill=255,
        width=max(1, round(width*sc))), antialias=antialias)
    cv.blend(mask, color, 'manual_line', head, 'under', 1.,
             dict(width=width, points=[list(p) for p in points]))


def text_spec(text, families, **overrides):
    spec = dict(text=text, charset='mixed', layout='line',
        fonts=[font(f) for f in families], size=.8, size_jitter=.08,
        weight=.75, bold_stroke=0, x_scale=1., slant=0., rotation=15.,
        jitter=.08, spacing=.10, curve=0., multicolour=True,
        palette='manual_glyph_palette', min_contrast=45.)
    spec.update(overrides)
    return spec


def draw_glyphs(cv, spec, rng, palette):
    # Fixed per-character colors are a demo-only control absent from the public spec.
    original = S.colour
    state = [0]
    def controlled_color(r, name, background=(255, 255, 255), text_colour=None):
        if name == 'manual_glyph_palette':
            c = palette[state[0] % len(palette)]
            state[0] += 1
            return c
        return original(r, name, background, text_colour)
    S.colour = controlled_color
    try:
        return T.draw_text_spec(cv, spec, rng, BANK, colour=palette[0])
    finally:
        S.colour = original


def cryptbb(seed=202):
    rng = np.random.default_rng(seed)
    cv = R.new_canvas(200, 60, dict(background=[255,255,255]), rng)
    effects = dict(effects=[dict(name='blocks', level=4,
        style=dict(palette='pastel', layout='tiled', opacity=.48, speckle=True)),
        dict(name='dots', level=7, style=dict(palette='pastel'))])
    R.draw_under(cv, effects, rng)
    pastels = [(226,172,195), (153,209,206), (186,202,237), (193,218,164), (223,197,145)]
    for x in range(0,200,7):
        line(cv, [(x,0),(x,59)], pastels[int(rng.integers(len(pastels)))], .65, 'bg_grid')
    for y in range(0,60,7):
        line(cv, [(0,y),(199,y)], pastels[int(rng.integers(len(pastels)))], .65, 'bg_grid')
    for k in range(17):
        xs = np.linspace(-5,205,120)
        phase = rng.uniform(0,2*math.pi)
        ys = rng.uniform(8,52) + rng.uniform(7,17)*np.sin(xs/rng.uniform(9,25)+phase)
        line(cv, list(zip(xs,ys)), pastels[k % len(pastels)], .65, 'curves')
    spec = text_spec('G7kQm', ['Roboto Slab','Roboto','Merriweather'],
        size=.93, size_jitter=.14, weight=.90, bold_stroke=1,
        rotation=24., jitter=.12, spacing=.08)
    palette = [(89,104,34),(48,83,143),(148,44,92),(20,123,101),(102,56,138)]
    txt = draw_glyphs(cv, spec, rng, palette)
    rectangle(cv, (0,0,199,59), (35,35,35), head='border_frame', fill=False, width=1)
    return cv, txt, dict(seed=seed, canvas=[200,60], effects=effects,
        text=spec, glyph_palette=palette, grid_spacing=7, curve_count=17)


def darkbay(seed=304):
    rng = np.random.default_rng(seed)
    cv = R.new_canvas(200,50,dict(background=[127,114,137]),rng)
    for k in range(21):
        x = float(rng.uniform(-10,188)); w = float(rng.uniform(13,48))
        y = float(rng.uniform(-8,12)); h = float(rng.uniform(31,63))
        delta = int(rng.choice([-37,-25,-16,10,19,29]))
        color = tuple(int(np.clip(c+delta,0,255)) for c in cv.background)
        rectangle(cv,(x,y,x+w,y+h),color)
    for k in range(8):
        x = float(rng.uniform(0,178)); w = float(rng.uniform(14,37))
        y = float(rng.uniform(0,8)); h = float(rng.uniform(33,49))
        delta = int(rng.choice([-29,24]))
        color = tuple(int(np.clip(c+delta,0,255)) for c in cv.background)
        rectangle(cv,(x,y,x+w,y+h),color,fill=False,width=2)
    spec = text_spec('K7TM8X',['Roboto'], size=.87, size_jitter=.05,
        weight=.60, x_scale=1.10, rotation=17., jitter=.08, spacing=.12)
    palette = [(250,250,248),(20,19,23),(248,248,246),(19,18,22),(246,246,244),(22,21,25)]
    txt = draw_glyphs(cv,spec,rng,palette)
    return cv,txt,dict(seed=seed,canvas=[200,50],text=spec,
        glyph_palette=palette,background=[127,114,137],
        tall_rectangles=21,outlined_rectangles=8,background_lightness_offsets=[-37,-25,-16,10,19,29])


def city(seed=410):
    rng = np.random.default_rng(seed)
    cv = R.new_canvas(324,81,dict(background=[255,255,255]),rng)
    controls = [(10,15,5),(22,18,4),(34,48,5),(45,37,4),(58,68,5),(71,62,4)]
    for y0,y1,width in controls:
        line(cv,[(-4,y0),(328,y1)],(202,202,202),width,'bands',antialias=False)
    spec = text_spec('58316',['Roboto Mono'], size=.49, size_jitter=.07,
        weight=.65, bold_stroke=0, x_scale=.95, rotation=11.,
        jitter=.13, spacing=.95, multicolour=False)
    palette = [(43,43,43)]
    txt = draw_glyphs(cv,spec,rng,palette)
    # Nearest-neighbor upscaling keeps pixel-stepped edges rather than a soft blur.
    post = dict(effects=[dict(name='pixelate',level=1)])
    R.apply_post(cv,post,rng)
    return cv,txt,dict(seed=seed,canvas=[324,81],text=spec,
        glyph_palette=palette,bands=controls,band_color=[202,202,202],post=post)


def main():
    report = dict(method='Manual target-reference controls; no feature-model inference', families={})
    for name,draw in [('CryptBB',cryptbb),('DarkBay',darkbay),('City',city)]:
        cv,txt,recipe = draw()
        if txt is None:
            raise RuntimeError(f'{name}: glyph rendering failed')
        native = Image.fromarray(cv.image())
        native.save(OUT/f'{name}.png')
        native.resize((native.width*3,native.height*3),Image.Resampling.NEAREST).save(OUT/f'{name}_preview.png')
        guard = legibility(cv,txt)
        record = dict(recipe=recipe,legibility=guard,
            text_labels=T.text_labels(txt),drawing_records=cv.records,
            native_image=str(OUT/f'{name}.png'),preview_image=str(OUT/f'{name}_preview.png'))
        (OUT/f'{name}_recipe.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        report['families'][name]=dict(image=record['native_image'],legibility=guard)
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
