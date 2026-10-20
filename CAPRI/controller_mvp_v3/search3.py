from boot import ROOT,U
import copy,json
import numpy as np
import controls as C
from experiment import unique,expanded_candidates
import run as OLD

def expand_effect(start,name,level):
    p=copy.deepcopy(start);p['effects'][name]=level
    # v2's keep_only filter otherwise silently deletes newly proposed glyph/over/post effects.
    if p.get('keep_only') is not None and level>0 and name not in p['keep_only']:p['keep_only'].append(name)
    return p

def candidates(start,phase,colors,fonts,bgpool,rng):
    result=[copy.deepcopy(start)]
    def add(**kw):result.append({**copy.deepcopy(start),**kw})
    if phase=='all_effects':
        # All registered sampled effects, not just hybrid positives/under-effects.
        for name in C.R.R.SAMPLED:
            for level in [0,5,10]:result.append(expand_effect(start,name,level))
    elif phase=='background':
        glyphs=[n for n in C.R.R.GLYPH_EFFECTS if start['effects'].get(n,0)>0]
        for name in [None,*C.R.R.UNDER]:
            for level in ([0] if name is None else [2,6]):
                effects={n:start['effects'][n] for n in glyphs}
                if name:effects[name]=level
                add(keep_only=[*glyphs,*([name] if name else [])],effects=effects)
        for name in ['gradient','texture','pattern','blocks']:
            for pal in ['grey','shades','pastel','muted']:
                p=expand_effect(start,name,5);p['effect_styles']={**p['effect_styles'],name:{'palette':pal}};result.append(p)
    elif phase=='font':
        for f in fonts:add(font=f,font_pool=None)
    elif phase=='palette':
        result.extend(OLD.variants(start,'palette',colors,fonts,C.R.R.SAMPLED,rng)[1:])
        for mode in ['single','random','alternating']:
            for n in [2,6,16]:add(text_palette=colors['text'][:n],color_mode=mode)
        add(background_pool=bgpool)
    elif phase=='layout':
        result.extend(expanded_candidates(start,'layout',colors,bgpool,[fonts],C.R.R.SAMPLED,rng)[1:])
    else:
        for _ in range(40):
            p=copy.deepcopy(start);name=str(rng.choice(C.R.R.SAMPLED));p=expand_effect(p,name,int(rng.choice([0,2,5,10])))
            key=str(rng.choice(['size','width','spacing','weight']));lo,hi={'size':(.55,1.25),'width':(.55,1.4),'spacing':(-.4,.5),'weight':(-.4,.4)}[key]
            p[key]=float(np.clip(p[key]+rng.normal(0,.12*(hi-lo)),lo,hi));result.append(p)
    return unique(result)
