"""Whole-family predictions -> empirical profile -> fresh labelled CAPTCHAs.

python -m feature_bank.iter2.controller build --predictions predictions.jsonl --manifest manifest.json --out profiles
python -m feature_bank.iter2.controller generate --profiles profiles --out dataset --count 100 --workers 12
"""
import argparse,copy,csv,hashlib,json,math,sys,time
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from feature_bank.iter2.generate import CE_ROOT
sys.path.insert(0,str(CE_ROOT))
from captcha_effects import text as T,render as R
from captcha_effects.effects import REGISTRY
from feature_bank.iter2.mapping import map_recipe

BINARY={'font_mixture','multicolor_text','two_layer','digits_only','uppercase_only','lowercase_only','mixed_case_alnum'}
CONTENT_HEADS=('digits_only','uppercase_only','lowercase_only','mixed_case_alnum')
CONTENT=('digits','upper','lower','mixed')
FONT_KEYS=('category','stroke','weight','width','x_height','slant','italic')
_BANK=None

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def histogram(values):return {str(k):v for k,v in sorted(Counter(values).items(),key=lambda pair:str(pair[0]))}
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')

def build_profiles(predictions,manifest,out):
    predictions,manifest,out=Path(predictions),Path(manifest),Path(out)
    expected=json.loads(manifest.read_text());rows=[json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    actual=dict(Counter(r['family'] for r in rows))
    if len(rows)!=expected['total'] or actual!=expected['counts']:raise ValueError('Inference incomplete: counts do not match the manifest')
    if any('error' in r for r in rows):raise ValueError('Inference contains failed rows')
    if len({(r['family'],r['index']) for r in rows})!=len(rows):raise ValueError('Repeated inference row IDs')
    out.mkdir(exist_ok=False)
    inference_report=predictions.parent/'report.json'
    provenance=json.loads(inference_report.read_text()) if inference_report.exists() else {}
    index={}
    for family in sorted(actual):
        selected=[r for r in rows if r['family']==family];prototypes=Counter();clean=[]
        for row in selected:
            # Source header supplies canvas dimensions only. No text answers,
            # target font assets, real pixel fitting or previous render recipes.
            with Image.open(row['source']) as image:size=list(image.size)
            state=dict(size=size,length=int(row['length']),font={k:row['font'][k] for k in FONT_KEYS},
                       content=CONTENT[int(np.argmax([row['text_features'][h]['probability'] for h in CONTENT_HEADS]))])
            for group in ('effects','text_features'):
                state[group]={h:dict(present=bool(g['present']),level=(None if h in BINARY else int(g['level']) if g['present'] else 0)) for h,g in row[group].items()}
            prototypes[json.dumps(state,sort_keys=True)]+=1;clean.append(state)
        heads={}
        for group in ('effects','text_features'):
            heads[group]={}
            for head in clean[0][group]:
                active=[r[group][head]['level'] for r in clean if r[group][head]['present'] and r[group][head]['level'] is not None]
                heads[group][head]=dict(prevalence=sum(r[group][head]['present'] for r in clean)/len(clean),
                    mean_probability=float(np.mean([r[group][head]['probability'] for r in selected])),
                    level_distribution_when_present=histogram(active),mean_level_when_present=float(np.mean(active)) if active else None)
        profile=dict(version=1,family=family,n_predictions=len(clean),presence_threshold=.5,
            summaries=dict(**heads,length_distribution=histogram(r['length'] for r in clean),content_distribution=histogram(r['content'] for r in clean),
                canvas_distribution=histogram(f'{r["size"][0]}x{r["size"][1]}' for r in clean),
                font_distributions={key:histogram(r['font'][key] for r in clean) for key in FONT_KEYS}),
            joint_distribution=[dict(count=count,description=json.loads(state)) for state,count in sorted(prototypes.items())],
            provenance=dict(predictions_sha256=digest(predictions),manifest_sha256=digest(manifest),epoch=provenance.get('epoch'),checkpoint_sha256=provenance.get('checkpoint_sha256')),
            policy=dict(sampling='Weighted empirical joint descriptions: preserves co-occurrence instead of independently averaging features.',
                level_average='Only images where the feature is predicted present contribute to level summaries.',
                probability_usage='Uncalibrated probabilities are summaries, not prevalence weights. Presence uses the frozen inference threshold.',
                content='Highest-probability content group per image; category forced only for new synthetic text.',
                unsupported='Colours, glyph positions, signed directions, effect subtype and exact font identity use renderer defaults.',
                real_labels_used=False,real_text_used=False,target_fonts_used=False))
        name=family+'.json';dump(out/name,profile);index[family]=name
    dump(out/'index.json',dict(families=index,total_predictions=len(rows),state='complete'))
    print(json.dumps(dict(stage='profiles_complete',families=actual,out=str(out))),flush=True)
    return index

def bank():
    global _BANK
    if _BANK is None:_BANK=T.FontBank()
    return _BANK

def initial_recipe(description,rng):
    b=bank();font=description['font'];size=description['size'];content=description['content']
    text=T.make_text(rng,description['length'],content)
    if content=='mixed':
        chars=list(text);chars[0]=str(rng.choice(list(T.UPPER)));chars[1]=str(rng.choice(list(T.LOWER)));rng.shuffle(chars);text=''.join(chars)
    pool=b.pool('train',category=font['category'],caseless=content in ('digits','upper'))
    rank=lambda f:(int(bool(f.get('italic'))!=bool(font['italic'])),int(f['stroke']!=font['stroke']),abs((T.x_height_level(f) if T.x_height_level(f) is not None else 5)-font['x_height']))
    pool=sorted(pool,key=rank);nearest=[f for f in pool if rank(f)==rank(pool[0])]
    chosen=nearest[int(rng.integers(len(nearest)))];keys=[chosen['key']]
    if description['text_features']['font_mixture']['present']:
        others=[f['key'] for f in pool if f['key']!=chosen['key']]
        if others:keys.append(str(rng.choice(others)))
    xs=float(np.clip((.38+.077*font['width'])/(chosen.get('width_rel') or .8),.4,1.6))
    frac,bold,_=T.fit_weight(chosen,size[1]*.62,T.weight_target(font['weight']),text)
    ts=T.sample_text_spec(rng,b,length=description['length'],charset=content,layout='line',font_mix=1)
    ts.update(text=text,fonts=keys,size=.62,size_jitter=0.,rotation=0.,jitter=0.,spacing=.02,curve=0.,slant=.04*font['slant'],
              multicolour=description['text_features']['multicolor_text']['present'],palette='contrast',weight=frac,bold_stroke=bold,x_scale=xs)
    on=lambda h:description['text_features'][h]['present']
    lv=lambda h:max(1,description['text_features'][h]['level'] or 0)
    if on('per_char_rotation'):ts['rotation']=6*lv('per_char_rotation')
    if on('vertical_jitter'):ts['jitter']=.03*lv('vertical_jitter')*.62*math.sqrt(3)
    if on('size_variation'):ts['size_jitter']=.025*lv('size_variation')*math.sqrt(3)
    if on('text_on_curve'):ts.update(layout='curve',curve=.04*lv('text_on_curve')*.62)
    if on('two_layer'):ts.update(layout='two_layer',size=.45)
    if on('overlap_tight_spacing'):ts['spacing']=-.1*lv('overlap_tight_spacing')
    if on('wide_spacing'):ts['spacing']=.1*lv('wide_spacing')
    # Avoid overwriting the fitted intrinsic font width with a second arbitrary
    # x-scale. The feedback mapper resolves the two predictions jointly.
    es=dict(background='random',effects=[dict(name=next(n for n in R.SAMPLED if REGISTRY[n]['head']==h),level=max(1,g['level'])) for h,g in description['effects'].items() if g['present']])
    return es,ts

def generate_item(item):
    family,index,prototype_id,description,seed,out,attempts=item;output=Path(out)
    from feature_bank.iter2.balanced import readability
    failures=[]
    for attempt in range(attempts):
        render_seed=seed+index*1009+attempt*9176
        try:
            es,ts=initial_recipe(description,np.random.default_rng(render_seed))
            mapped=map_recipe(bank(),tuple(description['size']),es,ts,description,render_seed,passes=1)
            rendered=mapped['after']
            if rendered is None or not rendered['text']['usable']:
                failures.append('Renderer usable flag failed');continue
            actual=rendered['text']
            if len(actual['text'])!=description['length']:failures.append('Character missing');continue
            guard,guard_report=readability(rendered)
            path=output/family/'images'/f'{index:06d}.png';Image.fromarray(rendered['image']).save(path)
            return dict(family=family,index=index,image=str(path.relative_to(output)),prototype_id=prototype_id,seed=render_seed,attempt=attempt,
                requested=description,text=actual['text'],length=actual['length'],effects=rendered['effects'],text_features=actual['heads'],font_style=actual['style'],glyphs=actual['slots'],
                effects_spec=mapped['effects_spec'],text_spec=mapped['text_spec'],mapping_distance_before=mapped['before_distance'],mapping_distance_after=mapped['after_distance'],
                renderer_usable=True,strict_training_readability=guard,strict_guard_report=guard_report,pixel_sha256=hashlib.sha256(rendered['image'].tobytes()).hexdigest(),previous_failures=failures)
        except Exception as exc:failures.append(repr(exc))
    return dict(family=family,index=index,error='No usable render within attempt budget',failures=failures,prototype_id=prototype_id)

def generate_profiles(profiles,out,count,workers,seed,attempts):
    if count<1 or workers<1 or attempts<1:raise ValueError('Count, workers and attempts must be positive')
    profiles,out=Path(profiles),Path(out);profile_index=json.loads((profiles/'index.json').read_text())
    if profile_index['state']!='complete':raise ValueError('Profile stage is incomplete')
    out.mkdir(exist_ok=False);(out/'profiles').mkdir();items=[];loaded={};selection={}
    for family,filename in profile_index['families'].items():
        profile=json.loads((profiles/filename).read_text());loaded[family]=profile;dump(out/'profiles'/filename,profile)
        proto=profile['joint_distribution'];weights=np.asarray([p['count'] for p in proto],float);weights/=weights.sum()
        # Family-stable seed, independent of order or worker count.
        family_seed=seed+int.from_bytes(hashlib.sha256(family.encode()).digest()[:4],'little')
        chosen=np.random.default_rng(family_seed).choice(len(proto),count,p=weights)
        selection[family]=chosen.tolist();(out/family/'images').mkdir(parents=True)
        for i,j in enumerate(chosen):items.append((family,i,int(j),proto[int(j)]['description'],family_seed,str(out),attempts))
    dump(out/'sampling_plan.json',dict(seed=seed,count_per_family=count,prototype_ids=selection))
    results=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool,(out/'labels.jsonl').open('w',encoding='utf-8') as stream:
        for row in pool.map(generate_item,items,chunksize=1):
            results.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if len(results)%20==0 or len(results)==len(items):
                status=dict(stage='generating',completed=len(results),total=len(items),failed=sum('error' in r for r in results),seconds=round(time.perf_counter()-start,1))
                dump(out/'status.json',status);print(json.dumps(status),flush=True)
    audit={}
    for family in loaded:
        rows=[r for r in results if r['family']==family];valid=[r for r in rows if 'error' not in r]
        with (out/family/'labels.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['filename','label']);writer.writeheader()
            writer.writerows(dict(filename=str(Path(r['image']).relative_to(family)).replace('\\','/'),label=r['text']) for r in valid)
        if len({r['pixel_sha256'] for r in valid})!=len(valid):raise ValueError('Duplicate generated pixels detected')
        for row in valid:
            with Image.open(out/row['image']) as im:
                if hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()!=row['pixel_sha256']:raise ValueError('Saved image hash mismatch')
        def rates(key,group):return {h:sum(bool(r[key][group][h]['present']) if key=='requested' else bool(r[group][h]['present']) for r in valid)/len(valid) for h in loaded[family]['summaries']['effects']} if valid else {}
        audit[family]=dict(requested=count,generated=len(valid),failed=len(rows)-len(valid),unique_pixels=len(valid),
            sampled_lengths=histogram(r['requested']['length'] for r in valid),actual_lengths=histogram(r['length'] for r in valid),
            requested_effect_prevalence=rates('requested','effects'),actual_effect_prevalence={h:sum(r['effects'][h]['present'] for r in valid)/len(valid) for h in loaded[family]['summaries']['effects']} if valid else {},
            strict_training_readability_passed=sum(r['strict_training_readability'] for r in valid),
            mean_mapping_distance=float(np.mean([r['mapping_distance_after'] for r in valid])) if valid else None)
        sheet=Image.new('RGB',(1000,35+math.ceil(len(valid)/5)*85),'white');draw=ImageDraw.Draw(sheet);draw.text((8,8),f'{family}: family-profile generated dataset ({len(valid)} images)',fill='black')
        for j,r in enumerate(valid):
            with Image.open(out/r['image']) as im:
                im.thumbnail((190,60));x=(j%5)*200;y=35+(j//5)*85;sheet.paste(im,(x,y));draw.text((x,y+62),f'{r["index"]}: {r["text"]}',fill='black')
        sheet.save(out/f'{family}_generated.jpg',quality=93)
    failures=[r for r in results if 'error' in r]
    report=dict(stage='complete' if not failures else 'complete_with_failures',count_per_family=count,seed=seed,seconds=round(time.perf_counter()-start,1),audit=audit,failures=failures,
        real_answers_used=False,training_started=False,
        policy='Sample whole-set joint feature distribution; fresh text/fonts/seeds; corrected measured-label mapping. Failed retries keep the same prototype to avoid silently changing family proportions. Actual renderer labels are authoritative.',
        limitation='Renderer usability is required; stricter training-readability results are reported separately. Colours/positions are unspecified defaults. Family profiles inherit prediction errors. No claim that this preview dataset meets the 100k training corpus gates.')
    dump(out/'report.json',report);dump(out/'status.json',dict(stage=report['stage'],completed=len(results),total=len(items)))
    print(json.dumps(report),flush=True)
    if failures:raise RuntimeError(f'{len(failures)} renders failed; inspect report.json')

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('build');build.add_argument('--predictions',required=True);build.add_argument('--manifest',required=True);build.add_argument('--out',required=True)
    generate=sub.add_parser('generate');generate.add_argument('--profiles',required=True);generate.add_argument('--out',required=True);generate.add_argument('--count',type=int,default=100);generate.add_argument('--workers',type=int,default=12);generate.add_argument('--seed',type=int,default=42);generate.add_argument('--attempts',type=int,default=6)
    run=sub.add_parser('run',help='After inference completes: build all family profiles, then generate their datasets.')
    run.add_argument('--inference',required=True);run.add_argument('--out',required=True);run.add_argument('--count',type=int,default=100);run.add_argument('--workers',type=int,default=12);run.add_argument('--seed',type=int,default=42);run.add_argument('--attempts',type=int,default=6)
    args=parser.parse_args()
    if args.command=='build':build_profiles(args.predictions,args.manifest,args.out)
    elif args.command=='generate':generate_profiles(args.profiles,args.out,args.count,args.workers,args.seed,args.attempts)
    else:
        inference=Path(args.inference);report=json.loads((inference/'report.json').read_text())
        if report.get('state')!='complete':raise ValueError('Run requires completed whole-set inference')
        destination=Path(args.out);destination.mkdir(exist_ok=False)
        build_profiles(inference/'predictions.jsonl',inference/'manifest.json',destination/'profiles')
        generate_profiles(destination/'profiles',destination/'dataset',args.count,args.workers,args.seed,args.attempts)

if __name__=='__main__':main()
