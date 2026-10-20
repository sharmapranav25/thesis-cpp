"""Run a pixels-only renderer controller: --targets folder OR --splits manifest.json.

The input manifest contains fit, selection and test lists of {source,pixel_sha256}.
Target filenames are opaque locations, never parsed for strings or attributes.
"""
import common as U
import argparse,copy,hashlib,time,html
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image,ImageDraw
import render_control as R
from comparison import Comparator,Objective,palette,WEIGHTS

def load_images(rows):
    images=[]
    for row in rows:
        im=np.asarray(Image.open(row['source']).convert('RGB'))
        if hashlib.sha256(im.tobytes()).hexdigest()!=row['pixel_sha256']:raise ValueError('Target pixels changed')
        images.append(im)
    return images

def make_split(folder,seed):
    entries={}
    for path in sorted(Path(folder).rglob('*')):
        if path.suffix.lower() not in ['.png','.jpg','.jpeg','.webp','.bmp']:continue
        im=np.asarray(Image.open(path).convert('RGB'));digest=hashlib.sha256(im.tobytes()).hexdigest()
        entries.setdefault(digest,dict(source=str(path.resolve()),pixel_sha256=digest))
    rows=list(entries.values());np.random.default_rng(seed).shuffle(rows)
    if len(rows)<60:raise ValueError('Provide at least 60 unique target images for three disjoint splits')
    fit=min(128,len(rows)//2);sel=min(64,(len(rows)-fit)//2)
    return dict(fit=rows[:fit],selection=rows[fit:fit+sel],test=rows[fit+sel:fit+sel+256])

def validate_split(split):
    sets=[]
    for key in ['fit','selection','test']:
        if not split[key]:raise ValueError('Empty '+key)
        hashes=[r['pixel_sha256'] for r in split[key]]
        if len(hashes)!=len(set(hashes)):raise ValueError('Duplicate '+key+' pixels')
        sets.append(set(hashes))
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Target split leakage')

def recipes(pool,descriptions,count,seed):
    rng=np.random.default_rng(seed)
    return list(pool.map(R.prepare,[(descriptions[int(rng.integers(len(descriptions)))],int(rng.integers(1,2**30)),i) for i in range(count)]))

def variants(current,phase,colors,font_choices,effect_names,rng):
    candidates=[copy.deepcopy(current)]
    def add(**kwargs):p=copy.deepcopy(current);p.update(kwargs);candidates.append(p)
    if phase=='palette':
        add(background=colors['background'][0])
        for mode in ['single','random','alternating']:
            add(background=colors['background'][0],text_palette=colors['text'],color_mode=mode)
            add(background=colors['background'][0],text_palette=colors['text'],effect_palette=colors['effect'],color_mode=mode)
        add(effect_palette=colors['effect'])
    elif phase=='geometry':
        for key,values in dict(size=[.65,.8,1.,1.15],width=[.7,.85,1.,1.15],spacing=[-.3,-.15,0,.15],weight=[-.25,0,.25],rotation=[0,.5,1,1.5],jitter=[0,.5,1,1.5]).items():
            for v in values:add(**{key:v})
    elif phase=='font':
        for key in font_choices:add(font=key)
    elif phase=='effects':
        for name in effect_names:
            for level in [0,2,5]:add(effects={**current['effects'],name:level})
    else:
        bounds=dict(size=(.55,1.25),width=(.55,1.4),spacing=(-.4,.4),weight=(-.4,.4),rotation=(0,1.8),jitter=(0,1.8))
        for _ in range(28):
            p=copy.deepcopy(current)
            for key in rng.choice(list(bounds),2,replace=False):
                lo,hi=bounds[key];p[key]=float(np.clip(p[key]+rng.normal(0,(hi-lo)*.13),lo,hi))
            if rng.random()<.5:
                key=str(rng.choice(effect_names));p['effects'][key]=int(rng.choice([0,1,2,3,5,7]))
            candidates.append(p)
    unique={}
    for p in candidates:unique.setdefault(__import__('json').dumps(p,sort_keys=True),p)
    return list(unique.values())

def save_rows(out,name,rows):
    folder=out/name;folder.mkdir(exist_ok=True);records=[]
    for i,row in enumerate(rows):
        image=folder/f'{i:04d}.png';mask=folder/f'{i:04d}_mask.png'
        Image.fromarray(row['image']).save(image);Image.fromarray(row['mask']).save(mask)
        records.append({k:v for k,v in row.items() if k not in ['image','mask']}|dict(image=str(image.relative_to(out)),mask=str(mask.relative_to(out))))
    U.dump(folder/'labels.json',records)

def gallery(out,real,baseline,refined,features):
    n=min(16,len(real),len(baseline));sheet=Image.new('RGB',(1024,24+90*n),'white');draw=ImageDraw.Draw(sheet)
    titles=['Held-out target','Hybrid start','Controller result','Controller text probability']
    for j,title in enumerate(titles):draw.text((256*j+5,5),title,fill='black')
    for i in range(n):
        a=np.asarray(Image.fromarray(real[i]).resize((256,64)));m=features['mask'][i]
        overlay=(a*.55+np.stack([m*255,np.zeros_like(m),np.zeros_like(m)],2)*.45).clip(0,255).astype(np.uint8)
        for j,im in enumerate([real[i],baseline[i]['image'],refined[i]['image'],overlay]):sheet.paste(Image.fromarray(im).resize((256,64)),(256*j,24+90*i))
    sheet.save(out/'comparison.jpg',quality=95)

def report(out,result):
    before=result['test']['baseline'];after=result['test']['refined']
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{before[k]:.4f}</td><td>{after[k]:.4f}</td></tr>' for k in ['text','background','appearance','audit','complete','usable','strict'])
    content=f'''<!doctype html><meta charset="utf-8"><title>Pixel controller MVP</title><style>body{{font:17px system-ui;max-width:1200px;margin:35px auto;padding:0 20px;background:#f8fafc;color:#182030}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:9px 20px;border-bottom:1px solid #ccc}}code{{background:#eee}}</style>
    <h1>Pixel controller MVP</h1><p>Hybrid starting recipe → internal soft text localization → text/background appearance comparison → renderer search.</p>
    <p><strong>{html.escape(result['conclusion'])}</strong></p><p>Lower text, background, appearance and audit distances are better. Completeness and readability are synthetic renderer checks, not target OCR accuracy.</p>
    <table><tr><th>Held-out measure</th><th>Hybrid start</th><th>Controller</th></tr>{rows}</table>
    <h2>Unselected first 16 held-out examples</h2><p>Fresh random synthetic strings. Rows do not share text with targets. Red overlay shows inferred text probability, not target ground truth.</p><img src="comparison.jpg">
    <h2>Scope and limitations</h2><p>Target inputs are pixels only. No target answers, character boxes, masks, texture labels, target fonts or family-specific presets. All target comparisons use predicted masks on both sides. Synthetic labels only supervise perception and check renderer validity.</p>
    <p>This MVP searches renderer controls; its update rule is not a trained neural policy. The internal mask encoder was trained on synthetic images and also had auxiliary synthetic attribute objectives. Its attribute predictions and the earlier learned adjustment network are not used here. Real mask accuracy is unknown.</p>
    <p>Text length, content group and canvas-size distribution remain the frozen hybrid estimates. The controller can change font, geometry, palettes and effect presence/levels. It cannot guarantee exact target-font identity or arbitrary texture recovery.</p>
    <p>See <a href="results.json">results</a>, <a href="selected_controls.json">frozen controls</a>, <a href="history.json">all candidates</a>, <a href="protocol.json">protocol</a>, and generated labels in <code>refined/labels.json</code>.</p>'''
    (out/'report.html').write_text(content,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__);input_group=parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--targets');input_group.add_argument('--splits');parser.add_argument('--out',required=True)
    parser.add_argument('--workers',type=int,default=12);parser.add_argument('--seed',type=int,default=74191)
    parser.add_argument('--recipe-cache',help='Optional previously prepared fit/selection recipes from identical fit descriptions and seed')
    parser.add_argument('--fit-renders',type=int,default=18);parser.add_argument('--selection-renders',type=int,default=36);parser.add_argument('--test-renders',type=int,default=96)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def event(stage,**values):
        row=dict(stage=stage,seconds=round(time.time()-start,1),**values);U.dump(out/'status.json',row);print(__import__('json').dumps(row),flush=True)
    split=U.read(args.splits) if args.splits else make_split(args.targets,args.seed);validate_split(split);U.dump(out/'target_splits.json',split)
    U.dump(out/'protocol.json',dict(seed=args.seed,fit_renders=args.fit_renders,selection_renders=args.selection_renders,test_renders=args.test_renders,
        target_counts={k:len(split[k]) for k in ['fit','selection','test']},weights=WEIGHTS,hybrid_sha256=U.sha(U.HYBRID),encoder_sha256=U.sha(U.ENCODER),
        phases=['palette','geometry','font','effects','refine','refine'],target_labels_used=False,
        policy='Fit-only candidate proposals. Selection chooses among the best three proposals per phase and incumbent; requires text/background <= 1.05 times selection baseline and usable >= baseline minus .03. Final test never selects controls. All candidates and failures saved.',
        mask='Predicted soft masks for both real and generated comparisons; synthetic truth used only for diagnostics.',test_policy='Freeze selected_controls.json before reading test images; no test-driven retry.'))
    model=Comparator();fit_images=load_images(split['fit']);selection_images=load_images(split['selection'])
    real=model.extract(fit_images);selection=model.extract(selection_images)
    descriptions=model.describe(fit_images);U.dump(out/'hybrid_descriptions.json',descriptions)
    colors={key:palette(fit_images,real,key) for key in ['background','text','effect']};U.dump(out/'pixel_palettes.json',colors)
    rng=np.random.default_rng(args.seed)
    # Generic train-font bank, stratified by category; never target font files.
    fonts=U.read(U.WORK/'learned_controller_20261004/fonts.json');font_choices=[]
    for category in sorted(set(f['category'] for f in fonts)):
        pool=[f['key'] for f in fonts if f['category']==category];font_choices.extend(pool[:6])
    effects=list(dict.fromkeys(e['name'] for d in descriptions for e in R.initial_recipe(d,np.random.default_rng(177))[0]['effects']))
    # All background/interference types remain reachable, even if initially absent.
    effects=[n for n in dict.fromkeys(effects+list(R.R.UNDER)) if n in R.R.SAMPLED]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=R.install) as pool:
        event('preparing_recipes')
        cache=Path(args.recipe_cache) if args.recipe_cache else None
        if cache:
            assert U.read(cache/'hybrid_descriptions.json')==descriptions
            prior=U.read(cache/'protocol.json')
            assert prior['seed']==args.seed and prior['fit_renders']==args.fit_renders and prior['selection_renders']==args.selection_renders
        fit_recipes=U.read(cache/'fit_recipes.json') if cache else recipes(pool,descriptions,args.fit_renders,args.seed+100)
        U.dump(out/'fit_recipes.json',fit_recipes);event('fit_recipes_ready')
        selection_recipes=U.read(cache/'selection_recipes.json') if cache else recipes(pool,descriptions,args.selection_renders,args.seed+200)
        U.dump(out/'selection_recipes.json',selection_recipes);event('selection_recipes_ready')
        def evaluate(controls,rr,reference,objective=None):
            rows=list(pool.map(R.render,[(r,controls) for r in rr]));features=model.extract([r['image'] for r in rows])
            return rows,features,objective.score(features,reference,rows) if objective else None
        base_rows,base_features,_=evaluate(R.NEUTRAL,fit_recipes,real);objective=Objective(real,base_features)
        sb_rows,sb_features,sb=evaluate(R.NEUTRAL,selection_recipes,selection,objective)
        current=copy.deepcopy(R.NEUTRAL);best_selection=sb;history=[];event('baseline_ready',selection=sb)
        # Exact neutral equivalence checked in a separate unhooked process/inspection.
        for phase in ['palette','geometry','font','effects','refine','refine']:
            candidates=variants(current,phase,colors,font_choices,effects,rng);ranked=[]
            for i,p in enumerate(candidates):
                try:
                    rows,f,score=evaluate(p,fit_recipes,real,objective);ranked.append((score['objective'],p,score))
                    history.append(dict(phase=phase,controls=p,fit=score))
                except Exception as exc:
                    history.append(dict(phase=phase,controls=p,error=repr(exc)))
                if (i+1)%12==0:event('search',phase=phase,candidates_done=i+1,candidates_total=len(candidates))
            ranked.sort(key=lambda x:x[0])
            for _,p,fit_score in ranked[:3]:
                rows,f,score=evaluate(p,selection_recipes,selection,objective)
                accepted=score['text']<=sb['text']*1.05 and score['background']<=sb['background']*1.05 and score['usable']>=sb['usable']-.03 and score['complete']>=sb['complete']
                history.append(dict(phase=phase,controls=p,selection=score,eligible=accepted))
                if accepted and score['objective']<best_selection['objective']:
                    current=copy.deepcopy(p);best_selection=score
            U.dump(out/'history.json',history);U.dump(out/'selected_controls.json',current)
            event('phase_complete',phase=phase,selection=best_selection)
        # Parameters are frozen before reading any locked test pixels.
        U.dump(out/'selected_controls.json',current);frozen_hash=U.sha(out/'selected_controls.json')
        event('frozen_before_test',controls_sha256=frozen_hash)
        test_images=load_images(split['test']);test=model.extract(test_images)
        test_recipes=recipes(pool,descriptions,args.test_renders,args.seed+300);U.dump(out/'test_recipes.json',test_recipes)
        baseline,bf,bs=evaluate(R.NEUTRAL,test_recipes,test,objective);refined,rf,rs=evaluate(current,test_recipes,test,objective)
        save_rows(out,'baseline',baseline);save_rows(out,'refined',refined);gallery(out,test_images,baseline,refined,test)
        np.savez_compressed(out/'test_features.npz',**{'target_'+k:v for k,v in test.items()},**{'baseline_'+k:v for k,v in bf.items()},**{'refined_'+k:v for k,v in rf.items()})
        # Three disjoint fresh-render seed blocks expose improvement instability.
        blocks=[]
        for ids in np.array_split(np.arange(len(baseline)),3):
            blocks.append(dict(baseline=objective.score({k:v[ids] for k,v in bf.items()},test,[baseline[i] for i in ids]),refined=objective.score({k:v[ids] for k,v in rf.items()},test,[refined[i] for i in ids])))
        measurable=rs['text']<bs['text'] and rs['background']<bs['background'] and rs['audit']<bs['audit'] and rs['usable']>=bs['usable']-.03
        result=dict(test=dict(baseline=bs,refined=rs),seed_blocks=blocks,selection=dict(baseline=sb,refined=best_selection),
                    objective_scales=objective.scales,controls_sha256=frozen_hash,seconds=round(time.time()-start,1),
                    measured_joint_improvement=measurable,visual_review='pending',
                    conclusion='Held-out text, background and independent audit distances improved; visual review still required.' if measurable else 'MVP ran, but joint held-out improvement is not established.')
        assert U.sha(out/'selected_controls.json')==frozen_hash
        U.dump(out/'results.json',result);report(out,result);event('complete',results=result)
if __name__=='__main__':main()
