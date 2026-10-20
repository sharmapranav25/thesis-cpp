"""Close prediction -> recipe -> measured-label loop, using renderer labels only."""
import copy
import numpy as np
from captcha_effects import text as T
from captcha_effects.captcha import render_captcha

def measure_distance(out,goals):
    if out is None:return 1e6
    total=0.;terms=0
    for group in ('effects','text_features'):
        measured=out['effects'] if group=='effects' else out['text']['heads']
        for h,g in goals[group].items():
            value=measured.get(h,{'present':0,'level':0})
            if bool(value['present'])!=bool(g['present']):total+=2
            if g['present'] and g.get('level') is not None and value.get('level') is not None:
                total+=abs(int(value['level'])-int(g['level']))
            terms+=1
    return total/max(1,terms)

def map_recipe(bank,size,effects_spec,text_spec,goals,seed,passes=2):
    """Fixed RNG for candidates. Real pixels/answers never enter the objective."""
    es=copy.deepcopy(effects_spec);ts=copy.deepcopy(text_spec)
    history=[];calls=0
    def render(es,ts):
        nonlocal calls
        calls+=1
        return render_captcha(np.random.default_rng(seed),bank,size,effects_spec=es,text_spec=ts)
    first=render(es,ts);best=first;score=measure_distance(best,goals)
    # Do not overwrite the fitted font width here: x_scale must satisfy both width
    # and condensed/expanded targets; conflicts are reported by measured labels.
    controls={
      'per_char_rotation':('rotation',np.linspace(0,60,13)),
      'vertical_jitter':('jitter',np.linspace(0,.6,13)),
      'size_variation':('size_jitter',np.linspace(0,.8,13)),
      'text_on_curve':('curve',np.linspace(0,.8,13)),
      'slant':('slant',np.linspace(0,.4,11)),
      'overlap_tight_spacing':('spacing',np.linspace(-.9,0,13)),
      'wide_spacing':('spacing',np.linspace(0,1.4,15)),
      'condensed':('x_scale',np.linspace(.4,1,13)),
      'expanded':('x_scale',np.linspace(1,1.6,13))}
    for iteration in range(passes):
        changed=False
        for h,(key,values) in controls.items():
            g=goals['text_features'].get(h,{})
            if not g.get('present'):continue
            previous=ts[key];chosen=previous
            for value in values:
                candidate=copy.deepcopy(ts);candidate[key]=float(value)
                rendered=render(es,candidate);candidate_score=measure_distance(rendered,goals)
                if candidate_score<score-1e-9:
                    score=candidate_score;best=rendered;chosen=float(value)
            if chosen!=previous:
                ts[key]=chosen;changed=True;history.append(dict(pass_index=iteration,head=h,setting=key,before=previous,after=chosen))
        for i,effect in enumerate(es['effects']):
            # Same learned effect type; change its knob to reproduce the desired
            # visible level in context, rather than assuming requested==visible.
            previous=effect['level'];chosen=previous
            for value in range(1,11):
                candidate=copy.deepcopy(es);candidate['effects'][i]['level']=value
                rendered=render(candidate,ts);candidate_score=measure_distance(rendered,goals)
                if candidate_score<score-1e-9:
                    score=candidate_score;best=rendered;chosen=value
            if chosen!=previous:
                es['effects'][i]['level']=chosen;changed=True;history.append(dict(pass_index=iteration,effect=effect['name'],before=previous,after=chosen))
        if not changed:break
    # Replay the final recipe, so reported pixels and recipe agree exactly.
    final=render(es,ts)
    return dict(before=first,after=final,effects_spec=es,text_spec=ts,history=history,calls=calls,
                before_distance=measure_distance(first,goals),after_distance=measure_distance(final,goals))
