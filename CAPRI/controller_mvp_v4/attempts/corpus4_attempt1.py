"""Matched synthetic transcripts and sampling budgets across all three arms."""
from setup4 import ROOT,V3,V1,U,FAMILIES,ARMS,CHARSET
import copy,hashlib,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
import controls as C

W,H=192,48
def compatible_font(font):
    meta=C.R.bank().by_key[font]
    if not meta['caps_only'] and all(c in meta['chars'] for c in CHARSET):return font
    eligible=[f for f in C.R.bank().pool('train',category=meta['category']) if not f['caps_only'] and all(c in f['chars'] for c in CHARSET)]
    if not eligible:raise RuntimeError('No full-alphabet font in category '+meta['category'])
    return sorted(eligible,key=lambda f:f['key'])[0]['key']

def clean_recipe(recipe,text,seed,font):
    r=copy.deepcopy(recipe);r['seed']=seed;r['size']=[256,64];r['description']['content']='mixed'
    r['effects_spec']=dict(background=[255,255,255],effects=[])
    r['text_spec'].update(text=text,charset='mixed',fonts=[font],layout='line',size=.7,size_jitter=0.,weight=.5,bold_stroke=0,x_scale=1.,slant=0.,rotation=0.,jitter=0.,spacing=.15,curve=0.,multicolour=False,palette='dark',font_assign='cyclic')
    return r

def render_job(job):
    index,split,family,recipe,text,seed,policies,anchor,font=job
    errors=[];recipe=copy.deepcopy(recipe);recipe['text_spec']['fonts']=[compatible_font(f) for f in recipe['text_spec']['fonts']]
    for attempt in range(30):
        r=copy.deepcopy(recipe);r['seed']=seed+attempt*100003;r['description']['content']='mixed'
        r['text_spec'].update(text=text,charset='mixed',layout='line',curve=0.)
        if anchor:r=clean_recipe(r,text,r['seed'],font)
        rendered=[];valid=True
        for arm,p in policies.items():
            p=copy.deepcopy(p);p['length']=None  # common transcript curriculum, independent of target answers
            if p['font']:p['font']=compatible_font(p['font'])
            if p['font_pool']:p['font_pool']=list(dict.fromkeys(compatible_font(f) for f in p['font_pool']))
            if anchor:p=C.expand(C.R.NEUTRAL)
            try:
                row=C.render((r,p))
                if not row['complete'] or not row['usable'] or row['text']!=text:valid=False;break
                array=np.asarray(Image.fromarray(row['image']).resize((W,H),Image.Resampling.BILINEAR))
                import json
                rendered.append((arm,array,dict(pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest(),native_sha256=row['pixel_sha256'],controls_sha256=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest())))
            except Exception as exc:errors.append(repr(exc));valid=False;break
        if valid:return index,rendered,dict(index=index,split=split,family=family,label=text,seed=r['seed'],attempts=attempt+1,anchor=anchor,errors=errors)
    raise RuntimeError(f'Cannot render matched readable sample {split}/{index}; no label or budget substitution')

def main():
    dest=ROOT/'corpus';dest.mkdir(exist_ok=False);rng=np.random.default_rng(42019);used=set();start=time.time()
    archives={f:U.read(ROOT/'collections'/f/'archive.json') for f in FAMILIES}
    singles={f:U.read(ROOT/'collections'/f/'single.json') for f in FAMILIES}
    templates={f:U.read(V1/('run_03' if f=='City' else f+'_run_01')/'fit_recipes.json') for f in FAMILIES}
    fonts=[f['key'] for f in U.read(U.WORK/'learned_controller_20261004/fonts.json')]
    U.dump(dest/'protocol.json',dict(train_per_arm=12800,common_validation=1600,shared_clean_fraction=.2,image_size=[W,H],strings='Same unique synthetic strings across arms; disjoint train/validation strings; uniform lengths 2..8 and generic alphanumeric alphabet, no target answer frequencies.',
        common_curriculum='Line layout and fresh full-alphabet strings shared across arms; length controls disabled for recognition training. Geometry/color/effect controls retained. 20% identical clean anchors ensure all arms see legible alphabet examples.',
        retry='Same new renderer seed for all arms until all are complete/usable with exact assigned transcript; maximum 30, fail rather than silently shrink an arm.',
        archives={f:U.sha(ROOT/'collections'/f/'archive.json') for f in FAMILIES},source_sha256={p.name:U.sha(p) for p in [ROOT/'corpus4.py',ROOT/'setup4.py']},
        font_support='Fonts lacking the complete generic alphabet or marked caps-only are replaced by the first compatible train font in the same category; applied uniformly to every arm.'))
    with ProcessPoolExecutor(max_workers=12,initializer=C.install) as pool:
        for split,n in [('train',12800),('validation',1600)]:
            arrays={a:np.lib.format.open_memmap(dest/(split+'_'+a+'.npy'),mode='w+',dtype=np.uint8,shape=(n,H,W,3)) for a in (ARMS if split=='train' else ['common'])}
            jobs=[];specs=[]
            for i in range(n):
                while True:
                    text=''.join(rng.choice(list(CHARSET),size=int(rng.integers(2,9))))
                    if text not in used:used.add(text);break
                anchor=i%5==0;local=i-i//5-1 if not anchor else i//5;family=FAMILIES[(local if not anchor else i//5)%5]
                # For non-anchors every family receives the same count and every configuration the same count.
                family_index=local//5 if not anchor else i//25;sid=family_index%16
                recipe=templates[family][family_index%len(templates[family])]
                policies=dict(hybrid=C.expand(C.R.NEUTRAL),single=singles[family],diverse=archives[family][sid])
                if split=='validation':policies={'common':policies[ARMS[local%3]]}
                seed=int(rng.integers(1000000,2**30));font=fonts[int(rng.integers(len(fonts)))]
                jobs.append((i,split,family,recipe,text,seed,policies,anchor,font));specs.append(dict(scheme=sid,anchor=anchor))
            records=[]
            for index,rendered,row in pool.map(render_job,jobs,chunksize=4):
                row.update(specs[index]);row['arms']={}
                for arm,pixels,meta in rendered:arrays[arm][index]=pixels;row['arms'][arm]=meta
                records.append(row)
                if len(records)%512==0:
                    U.dump(ROOT/'corpus_status.json',dict(split=split,done=len(records),total=n,seconds=time.time()-start));print(split,len(records),'/',n,flush=True)
            for array in arrays.values():array.flush()
            U.dump(dest/(split+'_labels.json'),records)
    train=U.read(dest/'train_labels.json');val=U.read(dest/'validation_labels.json')
    assert not {r['label'] for r in train}&{r['label'] for r in val}
    assert all(len({v['pixel_sha256'] for v in r['arms'].values()})==1 for r in train if r['anchor'])
    for arm in ARMS:
        hashes=[r['arms'][arm]['pixel_sha256'] for r in train]
        assert len(set(hashes))==len(hashes),'Duplicate synthetic training pixels'
        assert not set(hashes)&{r['arms']['common']['pixel_sha256'] for r in val},'Synthetic image split overlap'
    counts={f:{str(i):sum(not r['anchor'] and r['family']==f and r['scheme']==i for r in train) for i in range(16)} for f in FAMILIES}
    assert all(v==128 for f in counts.values() for v in f.values())
    U.dump(dest/'audit.json',dict(passed=True,train=12800,validation=1600,shared_labels=True,disjoint_strings=True,per_family_scheme_counts=counts,
        files={p.name:U.sha(p) for p in dest.glob('*.npy')},train_labels_sha256=U.sha(dest/'train_labels.json'),validation_labels_sha256=U.sha(dest/'validation_labels.json')))
    U.dump(ROOT/'corpus_status.json',dict(stage='complete',seconds=time.time()-start))

if __name__=='__main__':main()
