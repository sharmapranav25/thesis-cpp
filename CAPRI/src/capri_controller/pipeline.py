"""100 pixel inputs -> 100 cleaned hybrid recipes -> three controllers -> 300 entries."""
from pathlib import Path
import sys,copy,json,time,hashlib
from . import runtime as U
from .runtime import WORK,RUN_ROOT as ROOT
from .controllers import Mixed,Objective,key
from .rendering import renderer as R
from .artifacts import load_images,save_rows
from feature_bank.iter2.controller import initial_recipe
from captcha_effects.effects import REGISTRY
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from concurrent.futures import ProcessPoolExecutor

ARMS=['custom','dino','mixed']
METRICS=dict(custom='custom_objective',dino='dino_objective',mixed='objective')

def clean(description):
    d=copy.deepcopy(description);changes=[]
    groups=[['digits_only','uppercase_only','lowercase_only','mixed_case_alnum'],['overlap_tight_spacing','wide_spacing'],['condensed','expanded'],['two_layer','text_on_curve']]
    for names in groups:
        active=[n for n in names if d['text_features'][n]['present']]
        if len(active)>1:
            winner=max(active,key=lambda n:d['text_features'][n]['probability'])
            for name in active:
                if name!=winner:d['text_features'][name].update(present=False,level=0);changes.append(dict(rule='resolve_conflicting_heads',disabled=name,kept=winner))
    for names in [['gradient','texture','pattern'],['hollow','bold','thin'],['blur','pixelate','resample'],['wave','morph','perspective','swirl']]:
        heads=list(dict.fromkeys(REGISTRY[n]['head'] for n in names));active=[h for h in heads if d['effects'].get(h,{}).get('present')]
        if len(active)>1:
            winner=max(active,key=lambda h:d['effects'][h]['probability'])
            for h in active:
                if h!=winner:d['effects'][h].update(present=False,level=0);changes.append(dict(rule='resolve_exclusive_effects',disabled=h,kept=winner))
    for group in ['effects','text_features']:
        for name,g in d[group].items():
            old=g['level'];g['level']=int(np.clip(old,1,10)) if g['present'] else 0
            if old!=g['level']:changes.append(dict(rule='valid_level',head=name,before=old,after=g['level']))
    return d,changes

def recipe(description,seed,index):
    es,ts=initial_recipe(description,np.random.default_rng(seed))
    return dict(description=description,seed=seed,index=index,size=description['size'],effects_spec=es,text_spec=ts)

def draw_recipe(base,seed):
    r=copy.deepcopy(base);r['seed']=int(seed)
    r['text_spec']['text']=R.C.R.T.make_text(np.random.default_rng(seed+101),int(r['description']['length']),r['description']['content'])
    return r

def usable(row,length):return bool(row.get('complete') and row.get('usable') and len(row.get('text',''))==length)

def style_signature(entry):
    """Count parameter combinations without treating fresh strings as new styles."""
    text_spec=copy.deepcopy(entry['recipe']['text_spec']);text_spec.pop('text',None)
    return key(dict(recipe=text_spec,effects=entry['recipe']['effects_spec'],controls=entry['controls']))

def palette(im,mask):
    im=np.asarray(Image.fromarray(im).resize((256,64)))
    def reduce(where):
        x=im[where]
        if len(x)<16:x=im.reshape(-1,3)
        q=Image.fromarray(x.reshape(1,-1,3)).quantize(colors=5);p=q.getpalette()
        return [p[3*i:3*i+3] for _,i in sorted(q.getcolors(),reverse=True)]
    return reduce(mask>.5),reduce(mask<.2)

def slice_features(features,indices):return tuple({k:v[indices] for k,v in part.items()} for part in features)

def choose_lengths(counts,preview=False):
    """The documented length rule; fallback is enabled only for bounded previews."""
    assert sum(counts.values())==100
    winner=max(counts,key=counts.get)
    if counts[winner]>=60:
        return [int(winner)]*100,dict(counts=counts,threshold=.60,default_length=int(winner),votes=counts[winner],mode='majority')
    if not preview:raise RuntimeError('This run requires a model-derived majority; do not substitute a human length.')
    observed=sorted(int(n) for n,v in counts.items() if v)
    rng=np.random.default_rng(101073);assigned=[]
    while len(assigned)<100:assigned.extend(rng.permutation(observed).tolist())
    return assigned[:100],dict(counts=counts,threshold=.60,default_length=None,votes=counts[winner],mode='equal_observed_lengths_preview',
        probabilities={str(n):1/len(observed) for n in observed},
        assignment='Seed101073 shuffled cycles across observed lengths; each input fixes one length for all controller candidates/arms. Preview only; no corpus export.')


def main(preview=False):
    if ROOT.exists():raise FileExistsError('Use a new run directory; existing evidence is preserved: '+str(ROOT))
    ROOT.mkdir(parents=True);started=time.time();model=Mixed()
    refs=U.read(U.target_manifest())['fit'][:100]
    assert len(refs)==100 and len({r['pixel_sha256'] for r in refs})==100
    real=load_images(refs);U.dump(ROOT/'inputs.json',refs);raw=[];cleaned=[];raw_logits=[]
    hook=model.custom.hybrid.register_forward_hook(lambda mod,args,out:raw_logits.append({k:v[0].detach().cpu().tolist() for k,v in out.items() if isinstance(v,__import__('torch').Tensor) and not k.startswith('ctc')}))
    for i,im in enumerate(real):
        prediction=model.custom.describe([im])[0];d,changes=clean(prediction)
        raw.append(dict(index=i,source=refs[i],prediction=prediction,logits=raw_logits[-1]))
        cleaned.append(dict(index=i,description=d,changes=changes))
    hook.remove();counts={str(n):sum(r['prediction']['length']==n for r in raw) for n in range(2,9)}
    U.dump(ROOT/'raw_hybrid_predictions.json',raw)
    lengths,decision=choose_lengths(counts,preview)
    U.dump(ROOT/'length_decision.json',dict(**decision,assigned_lengths=lengths,model_sha256=U.sha(U.HYBRID),source='Raw hybrid argmax predictions on these100 pixels; not human labels or filename parsing.',note='Vote agreement is not calibrated accuracy.'))
    fit_count=6 if preview else 100
    recipes=[]
    for i,r in enumerate(cleaned):
        length=lengths[i];d=r['description'];old=d['length'];d['length']=length
        r['changes'].append(dict(rule='model_majority_length' if decision['mode']=='majority' else 'equal_observed_lengths_preview',before=old,after=length,votes=decision['votes']))
        rr=recipe(d,1100003+i*7919,i);before=copy.deepcopy(rr['text_spec'])
        for k,lo,hi in [('rotation',0,45),('jitter',0,.2),('size_jitter',0,.3),('spacing',-.35,.4),('x_scale',.4,1.6),('curve',0,.3)]:rr['text_spec'][k]=float(np.clip(rr['text_spec'][k],lo,hi))
        r['renderer_before_clamping']=before;r['renderer_after_clamping']=rr['text_spec'];recipes.append(rr)
    U.dump(ROOT/'refined_predictions.json',cleaned);U.dump(ROOT/'refined_recipes.json',recipes)
    target=model.extract(real);entries=[];pilot_rows=[];histories=[]
    U.dump(ROOT/'protocol.json',dict(inputs=100,refined_inputs=fit_count,preview_only=preview,controllers=ARMS,length_rule=decision,
        refinement='Highest probability resolves incompatible heads; inactive levels zero; clamp generic renderer parameter ranges; apply observed majority length. Preserve every raw prediction and per-image recipe.',
        comparison='50% corresponding image appearance descriptors +50% full100-image distribution. Custom and DINO metrics differ; mixed uses calibrated50/50 components. No character-string matching.',
        candidates='Per-image neutral, pixel-derived colors, generic geometry/effect variants, then separate mutations around each controller leader. Same model-derived length throughout.',
        selection='Per-controller shortlist validated on3 fresh synthetic seeds against the same unlabeled reference pixels; require all3 complete and usable. This is recipe reliability testing, not an unseen-target evaluation.',
        diversity=f'{fit_count} entries per controller, including shared choices. Do not select one globally best controller. Track exact duplicate settings and pixel hashes.',
        corpus='Disabled: visual preview only.' if preview else '100000 images; balanced333/334 allocations over300 entries, seeded remainder assignment. Fresh strings and render seeds; retry within the same entry for complete, usable outputs of derived length. Log every rejection.',
        source_hashes={k:v['sha256'] for k,v in U.registry()['models'].items() if v['status']=='active'},vlm=False,target_labels=False,code_sha256=U.sha(Path(__file__))))
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:
        # Shared calibration uses only the100 hybrid-derived recipes.
        calrows=list(pool.map(R.safe_render,[(r,R.expand(dict(length=r['description']['length']))) for r in recipes[:24]]))
        calibration=model.extract([r.get('image',np.full((64,256,3),255,np.uint8)) for r in calrows]);obj=Objective(target,calibration)
        for index,base in enumerate(recipes[:fit_count]):
            length=lengths[index];neutral=R.expand(dict(length=length))
            local=slice_features(target,slice(index,index+1));colors,bgs=palette(real[index],target[0]['mask'][index]);bg=bgs[0]
            # Remove colors nearly identical to the inferred background; this is pixel arithmetic.
            ink=[c for c in colors if np.linalg.norm(np.array(c)-np.array(bg))>=60] or [max(colors,key=lambda c:np.linalg.norm(np.array(c)-np.array(bg)))]
            start=R.expand(dict(length=length,background=bg,text_palette=ink,effect_palette=ink,color_mode='alternating' if len(ink)>1 else 'single'))
            options=[neutral,start,{**start,'color_mode':'random'},{**start,'background_pool':bgs,'background':None}]
            for size in [.5,.65,.8]:options.append({**start,'size_absolute':size})
            for width in [.75,1.1]:options.append({**start,'width_absolute':width})
            options.extend([{**start,'effects':{n:0 for n in R.C.R.R.SAMPLED}},{**start,'rotation_absolute':0.,'jitter_absolute':0.},{**start,'glyph_outline':1}])
            records=[];seen=set();fit_recipes=[draw_recipe(base,1200003+index*1009+j*104729) for j in range(2)]
            def evaluate(options,phase):
                unique=[]
                for p in options:
                    p=R.expand(p);assert p['length']==length
                    if key(p) not in seen:seen.add(key(p));unique.append(p)
                if not unique:return
                rows=list(pool.map(R.safe_render,[(r,p) for p in unique for r in fit_recipes]));images=[r.get('image',np.full((64,256,3),255,np.uint8)) for r in rows];f=model.extract(images)
                for j,p in enumerate(unique):
                    rr=rows[j*2:j*2+2];ff=slice_features(f,slice(j*2,j*2+2));global_score=obj.score(ff,target,rr);local_score=obj.score(ff,local,rr)
                    scores={a:.5*(global_score[m]+local_score[m]) for a,m in METRICS.items()}
                    records.append(dict(index=len(records),phase=phase,controls=p,scores=scores,usable=sum(usable(r,length) for r in rr)))
            evaluate(options,'initial')
            proposals=[]
            for arm in ARMS:
                p=min(records,key=lambda r:r['scores'][arm])['controls']
                proposals.extend([{**p,'rotation_absolute':v} for v in [5.,20.]])
                proposals.extend([{**p,'spacing_absolute':v} for v in [0.,.15]])
                proposals.extend([{**p,'size_absolute':.55,'jitter_absolute':.05},{**p,'color_mode':'single','text_palette':[ink[0]]}])
            evaluate(proposals,'controller_mutation')
            shortlist={}
            for arm in ARMS:
                for r in sorted(records,key=lambda r:r['scores'][arm])[:3]:shortlist[r['index']]=r
            validated=[];vr=[draw_recipe(base,1300003+index*1009+j*104729) for j in range(3)]
            for rec in shortlist.values():
                rows=list(pool.map(R.safe_render,[(r,rec['controls']) for r in vr]))
                if not all(usable(r,length) for r in rows):continue
                ff=model.extract([r['image'] for r in rows]);g=obj.score(ff,target,rows);l=obj.score(ff,local,rows)
                validated.append(dict(**rec,validation={a:.5*(g[m]+l[m]) for a,m in METRICS.items()}))
            if not validated:
                # Logged generic recovery, not a hidden replacement with a target template.
                for size in [.5,.4]:
                    p=R.expand(dict(length=length,size_absolute=size,width_absolute=.8,spacing_absolute=.1,rotation_absolute=0.,jitter_absolute=0.,background=bg,text_palette=[ink[0]],color_mode='single',effects={n:0 for n in R.C.R.R.SAMPLED}))
                    rr=list(pool.map(R.safe_render,[(r,p) for r in vr]))
                    if all(usable(r,length) for r in rr):
                        ff=model.extract([r['image'] for r in rr]);g=obj.score(ff,target,rr);l=obj.score(ff,local,rr)
                        validated=[dict(index=-1,phase='generic_recovery',controls=p,validation={a:.5*(g[m]+l[m]) for a,m in METRICS.items()})];break
            if not validated:raise RuntimeError('No usable recipe for input '+str(index)+'; no silent dropping or length change.')
            for arm in ARMS:
                chosen=min(validated,key=lambda r:r['validation'][arm]);p=chosen['controls'];sample=None
                for attempt in range(20):
                    r=draw_recipe(base,1400003+index*1009+attempt*10000019);out=list(pool.map(R.safe_render,[(r,p)]))[0]
                    if usable(out,length):sample=out;break
                if sample is None:raise RuntimeError('Pilot failed for '+arm+str(index))
                entry=dict(entry_id=f'{arm}_{index:03d}',controller=arm,input_index=index,recipe=base,controls=p,selected_phase=chosen['phase'],validation_score=chosen['validation'][arm],pilot_attempts=attempt+1)
                entries.append(entry);pilot_rows.append(sample)
            histories.append(dict(input_index=index,fit=records,validation=validated));U.dump(ROOT/'entries.json',entries)
            U.dump(ROOT/'status.json',dict(stage='controllers',processed=index+1,total=fit_count,seconds=time.time()-started))
            if preview or (index+1)%10==0:print('Controllers',index+1,'/',fit_count,flush=True)
    save_rows(ROOT,'pilot',pilot_rows);U.dump(ROOT/'controller_history.json',histories)
    for arm in ARMS:
        rows=[pilot_rows[i] for i,e in enumerate(entries) if e['controller']==arm];save_rows(ROOT,arm,rows)
    U.dump(ROOT/'pilot_summary.json',dict(entries=len(entries),per_controller={a:sum(e['controller']==a for e in entries) for a in ARMS},
        unique_parameter_signatures=len({style_signature(e) for e in entries}),
        generic_recovery_entries=sum(e['selected_phase']=='generic_recovery' for e in entries),complete=sum(r['complete'] for r in pilot_rows),usable=sum(r['usable'] for r in pilot_rows),lengths=sorted({len(r['text']) for r in pilot_rows}),seconds=time.time()-started))
    # Stable first6 input pairs, no visual selection.
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);sheet=Image.new('RGB',(1200,680),'#f4f6f8');d=ImageDraw.Draw(sheet)
    for j,title in enumerate(['Target original','Custom','DINO','Mixed']):d.text((12+300*j,12),title,font=font,fill='#182033')
    for i in range(6):
        for j,im in enumerate([real[i],*[pilot_rows[i*3+k]['image'] for k in range(3)]]):
            tile=ImageOps.contain(Image.fromarray(im),(286,84),Image.Resampling.NEAREST);sheet.paste(tile,(300*j+(300-tile.width)//2,60+i*100))
    sheet.save(ROOT/'pilot.png');U.dump(ROOT/'status.json',dict(stage='controllers',state='complete',processed=fit_count,total=fit_count,seconds=time.time()-started));print('Pilot complete',flush=True)

if __name__=='__main__':main()
