"""Frozen simple-hybrid feature utility study. No training or target answers."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys, json, copy, time, hashlib, itertools
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageDraw

WORK=Path('C:/Users/pranav/Documents/ChatGPT/thesis')
sys.path.insert(0,str(WORK/'metasim_controller_pilot_work/launch'))
import pilot as A
import pilot_render as P
from captcha_effects import text as T
from feature_bank.iter2.balanced import transformed_weight
OUT=WORK/'feature_ablation_20261004/run_03'
SEED=41062026

BG='bg_gradient bg_image_texture bg_blocks bg_grid bg_circles bg_decoy_chars bg_pattern border_frame'.split()
GLYPH='glyph_shadow extrude_3d glyph_outline hollow_outline_glyphs stroke_bold stroke_thin glyph_fill_texture translucent_text'.split()
INTER='straight_lines bands curves scribbles dot_noise blob_noise strike_through noise_over_text occlusion glyph_cuts'.split()
DIST='wave_warp morph_distortion perspective local_warp blur inverted_regions pixel_noise compression pixelation low_resolution colour_quantization'.split()
LAYOUT='per_char_rotation vertical_jitter size_variation text_on_curve slant overlap_tight_spacing wide_spacing condensed expanded two_layer'.split()
FONT='category stroke weight width x_height slant italic'.split()
GROUPS=dict(typography=['font.'+h for h in FONT]+['effect.'+h for h in GLYPH]+['text.font_mixture'],
            layout=['text.'+h for h in LAYOUT],
            background=['effect.'+h for h in BG]+['text.multicolor_text'],
            interference=['effect.'+h for h in INTER],distortion=['effect.'+h for h in DIST])
ALL=sum(GROUPS.values(),[])
FONT_DEFAULT=dict(category='Sans Serif',stroke='Sans Serif',weight=5,width=5,x_height=5,slant=0,italic=False)
START=time.time()
def dump(p,v):A.dump(p,v)
def event(stage,**kw):
    v=dict(stage=stage,seconds=round(time.time()-START,1),**kw)
    dump(OUT/'status.json',v)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(v,default=A.default)+'\n')
    print(json.dumps(v,default=A.default),flush=True)

def font_edit(r,h,value):
    ts=r['text_spec']; goal=copy.deepcopy(r['description']['font'])
    target=FONT_DEFAULT[h] if value=='off' else value
    if target==goal[h]:return
    goal[h]=target
    r['description']['font'][h]=target
    if h=='slant':ts['slant']=.04*float(target);return
    old=P.bank().by_key[ts['fonts'][0]]
    if h=='width':ts['x_scale']=float(np.clip((.38+.077*target)/(old.get('width_rel') or .8),.4,1.8));return
    if h in ('category','stroke','x_height','italic'):
        pool=P.bank().pool('train',category=goal['category'],caseless=not any(c.islower() for c in ts['text']))
        def rank(f):
            vals=dict(category=f['category'],stroke=f['stroke'],italic=bool(f.get('italic')),x_height=T.x_height_level(f))
            error=(abs((vals[h] if vals[h] is not None else 5)-target) if h=='x_height' else int(vals[h]!=target))
            return (error,int(vals['italic']!=bool(goal['italic'])),int(vals['stroke']!=goal['stroke']),abs((vals['x_height'] or 5)-goal['x_height']),f['key'])
        if not pool:return
        old=min(pool,key=rank);ts['fonts']=[old['key']]
        ts['x_scale']=float(np.clip((.38+.077*goal['width'])/(old.get('width_rel') or .8),.4,1.8))
    fit=transformed_weight(old,ts['text'],r['size'][1]*ts['size'],goal['weight'],ts['x_scale'],ts['slant'])
    if fit is not None:ts['weight'],ts['bold_stroke']=fit[1],fit[2]

def edit(recipe,ops):
    r=copy.deepcopy(recipe);es,ts=r['effects_spec'],r['text_spec']
    for key,value in ops.items():
        kind,h=key.split('.',1)
        if kind=='effect':
            existing=[e for e in es['effects'] if P.E.REGISTRY[e['name']]['head']==h]
            if value=='off':es['effects']=[e for e in es['effects'] if P.E.REGISTRY[e['name']]['head']!=h]
            elif existing:
                for e in existing:e['level']=int(value)
            else:es['effects'].append(dict(name=next(n for n in P.R.SAMPLED if P.E.REGISTRY[n]['head']==h),level=int(value)))
        elif kind=='font':font_edit(r,h,value)
        elif kind=='extra':
            if h=='white_background':es['background']=[255,255,255]
            elif h=='gray_effects':
                for e in es['effects']:e.setdefault('style',{})['palette']='grey'
        else:
            off=value=='off';v=0 if off else float(value)
            # Removing one side of a paired control must preserve the opposite side.
            if off and ((h=='text_on_curve' and ts['layout']!='curve') or
                        (h=='two_layer' and ts['layout']!='two_layer') or
                        (h=='overlap_tight_spacing' and ts['spacing']>=0) or
                        (h=='wide_spacing' and ts['spacing']<=.02) or
                        (h=='condensed' and ts['x_scale']>=1) or
                        (h=='expanded' and ts['x_scale']<=1)):
                continue
            if h=='per_char_rotation':ts['rotation']=v*6
            elif h=='vertical_jitter':ts['jitter']=v*.035
            elif h=='size_variation':ts['size_jitter']=v*.04
            elif h=='text_on_curve':ts.update(layout='line' if off else 'curve',curve=v*.03)
            elif h=='slant':ts['slant']=v*.04
            elif h=='overlap_tight_spacing':ts['spacing']=.02 if off else -.08*v
            elif h=='wide_spacing':ts['spacing']=.02 if off else .1*v
            elif h=='condensed':ts['x_scale']=1. if off else 1.-.06*v
            elif h=='expanded':ts['x_scale']=1. if off else 1.+.06*v
            elif h=='two_layer':ts['layout']='line' if off else 'two_layer'
            elif h=='multicolor_text':ts['multicolour']=not off
            elif h=='font_mixture':
                ts['fonts']=ts['fonts'][:1]
                if not off:
                    f=P.bank().by_key[ts['fonts'][0]]
                    pool=P.bank().pool('train',category=f['category'],caseless=not any(c.islower() for c in ts['text']))
                    other=next((f['key'] for f in pool if f['key']!=ts['fonts'][0]),None)
                    if other:ts['fonts'].append(other)
    assert ts['text']==recipe['text_spec']['text']
    return r

def render_task(item):
    recipe,ops=item
    r=edit(recipe,ops)
    row=P.render((r,P.NEUTRAL,[]))
    row.setdefault('pixel_sha256',hashlib.sha256(row['image'].tobytes()).hexdigest())
    row.setdefault('seed',recipe['seed'])
    row['recipe_changed']=bool(r['text_spec']!=recipe['text_spec'] or r['effects_spec']!=recipe['effects_spec'])
    row['block']=recipe['block']
    return row

def shape_features(images):
    """Unsupervised dark-ink geometry proxy, never called font accuracy."""
    from scipy import ndimage as ndi
    result=[]
    for image in images:
        rgb=np.asarray(Image.fromarray(image).resize((192,48)),float)/255
        gray=rgb@np.array([.299,.587,.114])
        # Three thresholds reduce dependence on one arbitrary foreground cutoff.
        vec=[]
        for threshold in (.25,.45,.65):
            m=gray<threshold
            labels,n=ndi.label(m);objects=ndi.find_objects(labels)
            shapes=[]
            for label,sl in enumerate(objects,1):
                if sl is None:continue
                hh,ww=sl[0].stop-sl[0].start,sl[1].stop-sl[1].start
                area=int((labels[sl]==label).sum())
                if hh>=6 and 2<=ww<=64 and area>=8:shapes.append((hh/48,ww/192,area/(hh*ww)))
            stats=np.median(shapes,axis=0) if shapes else np.zeros(3)
            vec.extend([m.mean(),min(len(shapes),20)/20,*stats,*m.mean(1).reshape(8,6).mean(1),*m.mean(0).reshape(16,12).mean(1)])
        result.append(vec)
    return np.asarray(result)

class Judges:
    def __init__(self,model,real):
        self.model=model;self.fb=A.Feedback(real)
        self.mean=real['shape'].mean(0);self.std=np.maximum(real['shape'].std(0),.04)
        self.bw=A.bandwidth((real['shape']-self.mean)/self.std)
    def extract(self,rows):
        images=[r['image'] for r in rows];f=self.model(images);f['shape']=shape_features(images);return f
    def score(self,f,real,rows):
        s=self.fb.score(f,real,rows)
        x=(f['shape']-self.mean)/self.std;y=(real['shape']-self.mean)/self.std
        s.update(ink_geometry=float(A.kernel(x,x,self.bw).mean()+A.kernel(y,y,self.bw).mean()-2*A.kernel(x,y,self.bw).mean()),
                 audit=self.fb.audit(f,real),complete=float(np.mean([r['text_complete'] for r in rows])),
                 changed=float(np.mean([r['recipe_changed'] for r in rows])))
        return s

def candidates(key):
    kind,h=key.split('.')
    if kind=='font':
        return dict(category=['Sans Serif','Serif','Display','Handwriting','Monospace'],stroke=['Sans Serif','Serif','Slab Serif'],
                    weight=[2,5,8],width=[2,5,8],x_height=[3,5,7],slant=[0,3,6],italic=[False,True])[h]
    if h in ('font_mixture','multicolor_text','two_layer'):return ['off',1]
    return ['off',2,5,8]

def make_profile(folder,rows,family):
    from feature_bank.iter2.controller import build_profiles,FONT_KEYS
    dest=folder/'profile_input';dest.mkdir()
    clean=[dict(family=family,index=i,source=r['source'],length=r['length'],font={k:r['font'][k] for k in FONT_KEYS},effects=r['effects'],text_features=r['text_features']) for i,r in enumerate(rows)]
    (dest/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in clean))
    dump(dest/'manifest.json',dict(total=len(clean),counts={family:len(clean)}))
    build_profiles(dest/'predictions.jsonl',dest/'manifest.json',dest/'profiles')
    return A.read(dest/'profiles'/f'{family}.json')

def prepare_recipes(pool,profile,n,seed):
    result=[]
    for block in range(3):
        recipes=A.recipes(pool,profile,n//3,seed+block*100003)
        for r in recipes:r['block']=block;r['index']=len(result);result.append(r)
    return result

def sub(f,indices):return {k:v[indices] for k,v in f.items()}
def eligible(s,base):return s['complete']==1 and s['usable']>=base['usable']-.05 and s['strict']>=base['strict']-.05

def coded_gallery(folder,real,groups):
    names=list(groups);order=np.random.default_rng(SEED+987).permutation(len(names))
    order=[names[i] for i in order];mapping={chr(65+i):n for i,n in enumerate(order)}
    w=300;h=104;sheet=Image.new('RGB',(w*(len(order)+1),38+8*h),'white');d=ImageDraw.Draw(sheet)
    d.text((5,10),'Target references (unpaired)',fill='black')
    for j in range(len(order)):d.text(((j+1)*w+5,10),f'Panel {chr(65+j)}',fill='black')
    for i in range(8):
        for j,im in enumerate([real[i]]+[groups[n][i]['image'] for n in order]):
            pic=Image.fromarray(im);pic.thumbnail((w-12,h-20));sheet.paste(pic,(j*w+6,38+i*h+(h-20-pic.height)//2))
    sheet.save(folder/'coded_comparison.jpg',quality=94);dump(folder/'panel_key.json',mapping)

def main():
    OUT.mkdir(exist_ok=False)
    protocol=dict(baseline='frozen simple neural hybrid only; cached predictions, new empirical fitting profiles and fresh strings',
      groups=GROUPS,fit_renders=18,selection_renders=36,test_renders=72,recipe_seed_blocks=3,
      selection='0.6 DINO MMD2 + 0.4 stats MMD2 + .8 unusable + .15 strict failures; completeness required, usability and strict readability within 5 points of baseline',
      diagnostics='Independent grayscale/HOG sliced Wasserstein and dark-ink geometry MMD, neither used for selection; ink includes clutter and is not font accuracy',
      group_screen='Off plus 6 fixed-budget random joint settings per group; select on independent selection recipes, drill down into top two groups by selection improvement including removal effects',
      individual='All 56 style attributes neutralized individually. Tune all members of the top two groups using a fixed categorical/ordinal catalogue, select best two fit candidates against neutral on selection.',
      interactions='Joint top-three individually selected edits, font-width plus wide-spacing, and best group combinations. Select before test.',
      test_policy='City reuses original recorded test split; last 591 untouched. Other families pixel-deduplicated 128 fit/64 selection/256 test. No test results until family policies frozen.',
      ablation_semantics='Effects removed; layout reset to neutral; font attributes set to named neutral defaults. These are renderer interventions, not retraining ablations.',
      font_defaults=FONT_DEFAULT,limitations=['Shared RNG means changing an effect can shift incidental downstream draws.','Font substitution may alter several intrinsic attributes.','Condensed/expanded/font width and text/font slant share renderer controls.','No real font labels or solver accuracy.','Three recipe seeds are not three independent training/search runs.','Confidence intervals are descriptive and uncorrected for multiple comparisons.'],
      preserved='Text and character count are fixed within each paired recipe. Four content heads and length are excluded. No real answers/CTC strings used. No neural updates.',
      seed=SEED)
    sourcepaths=[Path(__file__),WORK/'metasim_controller_pilot_work/launch/pilot.py',WORK/'metasim_controller_pilot_work/launch/pilot_render.py',P.PROJECT/'feature_bank/iter2/controller.py',P.PROJECT/'feature_bank/iter2/mapping.py',P.PROJECT/'runs/iter2_baseline_15_20261003_04/baseline_training/epoch_15/ema.pt']
    sourcepaths+=list(Path('E:/pranav/Final sem/captcha-effects/captcha_effects').glob('*.py'))
    protocol['source_hashes']={str(p):A.sha(p) for p in sourcepaths};dump(OUT/'protocol.json',protocol)
    rows=[json.loads(line) for line in (WORK/'iter2_full_target_recreation_01/predictions.jsonl').read_text().splitlines()]
    model=A.Features();summary={}
    with ProcessPoolExecutor(max_workers=6,initializer=P.install) as pool:
      for family in ['City','CryptBB','DarkBay']:
        folder=OUT/family;folder.mkdir(exist_ok=False);event('family_start',family=family)
        fam=[r for r in rows if r['family']==family]
        lookup={r['source'].lower():r for r in fam}
        if family=='City':
            old=A.read(WORK/'metasim_controller_pilot_work/run_20261004_01/target_split.json')['splits']
            splits={s:[lookup[r['source'].lower()] for r in old[s]] for s in ('fit','selection','test')}
        else:
            unique={}
            for r in fam:
                with Image.open(r['source']) as im:h=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
                unique.setdefault(h,r)
            ordered=list(unique.values());order=np.random.default_rng(SEED).permutation(len(ordered))
            splits={s:[ordered[i] for i in ids] for s,ids in dict(fit=order[:128],selection=order[128:192],test=order[192:448]).items()}
        split_manifest={}
        for s,rr in splits.items():
            split_manifest[s]=[]
            for r in rr:
                with Image.open(r['source']) as im:h=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
                split_manifest[s].append(dict(source=r['source'],pixel_sha256=h))
        sets=[{r['pixel_sha256'] for r in split_manifest[s]} for s in ('fit','selection','test')]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
        dump(folder/'target_splits.json',split_manifest)
        profile=make_profile(folder,splits['fit'],family)
        real={}
        for s in ('fit','selection'):
            ims=A.real_images(splits[s]);real[s]=model(ims);real[s]['shape']=shape_features(ims)
        judge=Judges(model,real['fit']);recipes={}
        for s,n,offset in [('fit',18,10),('selection',36,20),('test',72,30)]:
            event('prepare_recipes',family=family,split=s,count=n)
            recipes[s]=prepare_recipes(pool,profile,n,SEED+offset)
            dump(folder/f'{s}_recipes.json',recipes[s])
        cache={};history=[];render_count=0
        def evaluate(stage,ops,save=False):
            nonlocal render_count
            key=(stage,json.dumps(ops,sort_keys=True))
            if key in cache:return cache[key]
            rr=list(pool.map(render_task,[(r,ops) for r in recipes[stage]]));render_count+=len(rr)
            f=judge.extract(rr);s=judge.score(f,real[stage],rr)
            val=dict(ops=ops,score=s,rows=rr,features=f);cache[key]=val
            history.append(dict(stage=stage,ops=ops,score=s))
            return val
        baseline={s:evaluate(s,{}) for s in ('fit','selection')}
        plain=P.plain(recipes['fit'][0]);assert np.array_equal(plain['image'],baseline['fit']['rows'][0]['image'])
        dump(folder/'neutral_replay.json',dict(pixel_exact=True))
        group_selected={};rng=np.random.default_rng(SEED)
        for name,keys in GROUPS.items():
            options=[{k:'off' for k in keys}]
            # Exactly six candidates per group, each edits a random subset of group attributes.
            for j in range(6):
                keys_chosen=[k for k in keys if rng.random()<.35]
                if not keys_chosen:keys_chosen=[keys[j%len(keys)]]
                options.append({k:candidates(k)[int(rng.integers(len(candidates(k))))] for k in keys_chosen})
            fitted=[evaluate('fit',op) for op in options]
            shortlist=sorted(fitted,key=lambda x:x['score']['objective'])[:2]
            vals=[baseline['selection']]+[evaluate('selection',v['ops']) for v in shortlist]
            ok=[v for v in vals if v is baseline['selection'] or eligible(v['score'],baseline['selection']['score'])]
            best=min(ok,key=lambda v:v['score']['objective'])
            off=evaluate('selection',options[0])
            group_selected[name]=dict(ops=best['ops'],score=best['score'],off_score=off['score'])
            event('group_screen',family=family,group=name,baseline=baseline['selection']['score']['appearance'],best=best['score']['appearance'],off=off['score']['appearance'])
        ranked=sorted(GROUPS,key=lambda g:max(baseline['selection']['score']['objective']-group_selected[g]['score']['objective'],abs(group_selected[g]['off_score']['appearance']-baseline['selection']['score']['appearance'])),reverse=True)
        drill=ranked[:2];selected={}
        for group in drill:
          for key in GROUPS[group]:
            fitted=[evaluate('fit',{key:v}) for v in candidates(key)]
            shortlist=sorted(fitted,key=lambda x:x['score']['objective'])[:2]
            vals=[baseline['selection']]+[evaluate('selection',v['ops']) for v in shortlist]
            ok=[v for v in vals if v is baseline['selection'] or eligible(v['score'],baseline['selection']['score'])]
            best=min(ok,key=lambda v:v['score']['objective'])
            selected[key]=dict(ops=best['ops'],score=best['score'])
            event('individual_tuning',family=family,feature=key,selected=best['ops'],remaining=len(sum([GROUPS[g] for g in drill],[]))-len(selected))
        winners=sorted([v for v in selected.values() if v['ops']],key=lambda v:v['score']['objective'])[:3]
        joint={k:v for x in winners for k,v in x['ops'].items()}
        width=evaluate('selection',{'font.width':2,'text.wide_spacing':2})
        jointops=[joint,{'font.width':2,'text.wide_spacing':2},{k:v for g in drill for k,v in group_selected[g]['ops'].items()}]
        combos=[evaluate('selection',o) for o in jointops]
        combo=min([baseline['selection']]+[v for v in combos if eligible(v['score'],baseline['selection']['score'])],key=lambda v:v['score']['objective'])
        # Palette is outside learned head inventory; diagnose the known missing-control issue separately.
        extra=evaluate('selection',{'extra.white_background':1,'extra.gray_effects':1})
        policies=dict(groups=group_selected,drill_groups=drill,individual=selected,combination=dict(ops=combo['ops'],score=combo['score']),
                      interaction_candidates=jointops,missing_palette_control=extra['ops'])
        dump(folder/'selected_policies.json',policies)
        event('policies_frozen_before_test',family=family,drill_groups=drill)
        testims=A.real_images(splits['test']);real['test']=model(testims);real['test']['shape']=shape_features(testims)
        base=evaluate('test',{});A.save_group(folder,'baseline',base['rows']);np.savez(folder/'baseline_features.npz',**base['features'])
        results=dict(baseline=base['score'],groups={},individual={},interactions={},drill_groups=drill)
        def assess(name,ops,save=False):
            v=evaluate('test',ops);s=v['score'];b=base['score']
            result=dict(ops=ops,metrics=s,appearance_gain_pct=100*(b['appearance']-s['appearance'])/b['appearance'],
                objective_gain=b['objective']-s['objective'],audit_gain_pct=100*(b['audit']-s['audit'])/b['audit'],
                guard_pass=eligible(s,b),**judge.fb.bootstrap(base['features'],v['features'],real['test'],n=200))
            seedgains=[]
            for block in range(3):
                ids=np.array([i for i,r in enumerate(v['rows']) if r['block']==block])
                bs=judge.fb.score(sub(base['features'],ids),real['test'],[base['rows'][i] for i in ids])
                vs=judge.fb.score(sub(v['features'],ids),real['test'],[v['rows'][i] for i in ids])
                seedgains.append(bs['appearance']-vs['appearance'])
            result['seed_block_appearance_gains']=seedgains
            result['pixel_changed_fraction']=float(np.mean([a['pixel_sha256']!=b['pixel_sha256'] for a,b in zip(v['rows'],base['rows'])]))
            if save:A.save_group(folder,name,v['rows']);np.savez(folder/f'{name}_features.npz',**v['features'])
            return result,v
        gallery={'baseline':base['rows']}
        for g,keys in GROUPS.items():
            off,_=assess('group_off_'+g,{k:'off' for k in keys})
            tuned,v=assess('group_tuned_'+g,group_selected[g]['ops'],save=True)
            results['groups'][g]=dict(off=off,tuned=tuned)
        for i,key in enumerate(ALL):
            off,_=assess('off_'+key,{key:'off'})
            tuned=None
            if key in selected:tuned,_=assess('tuned_'+key,selected[key]['ops'])
            results['individual'][key]=dict(off=off,tuned=tuned)
            if i%8==0:event('test_ablation',family=family,completed=i+1,total=len(ALL))
        for i,op in enumerate(jointops):results['interactions'][str(i)],_=assess('interaction_'+str(i),op)
        results['combination'],v=assess('combination',combo['ops'],save=True);gallery['combination']=v['rows']
        results['missing_palette_control'],v=assess('palette_diagnostic',extra['ops'],save=True);gallery['palette_diagnostic']=v['rows']
        # Gallery candidates are chosen on selection, not test performance.
        for g in drill:gallery['group_'+g]=evaluate('test',group_selected[g]['ops'])['rows']
        if winners:gallery['best_individual']=evaluate('test',winners[0]['ops'])['rows']
        coded_gallery(folder,testims,gallery)
        dump(folder/'results.json',results);dump(folder/'search_history.json',history)
        dump(folder/'render_counts.json',dict(direct_renders=render_count,mapper_renders=sum(r['mapping_calls'] for rs in recipes.values() for r in rs),cached_variants=len(cache)))
        summary[family]=dict(baseline=results['baseline'],combination=results['combination'],groups=results['groups'],drill_groups=drill)
        dump(OUT/'summary.json',summary)
        event('family_complete',family=family,combination_gain=results['combination']['appearance_gain_pct'],render_count=render_count)
    preserved={p:A.sha(Path(p))==h for p,h in protocol['source_hashes'].items()}
    assert all(preserved.values())
    dump(OUT/'verification.json',dict(source_preservation=preserved,no_neural_updates=True,no_real_answers=True,completed_families=list(summary)))
    event('complete',families=list(summary))

if __name__=='__main__':main()
