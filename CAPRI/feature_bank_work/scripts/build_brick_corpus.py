"""Synthetic-only candidate corpus. Never reads real targets or starts training."""
import argparse
import collections
import itertools
import json
import string
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,digest,pixel_hash,new_run,FEATURES as VOCABULARY
from feature_bank.labeled_bricks import FEATURES,POLICY,sample_operations,compose,evidence,crop_labels
from feature_bank.regions import regions,verify_crops,check_group_partitions

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',default='E:/pranav/Final sem/sourcefree_synth')
    parser.add_argument('--out',default='runs/feature_bank_corpus_candidate_01')
    parser.add_argument('--groups',type=int,default=1000)
    args=parser.parse_args()
    if args.groups<20 or args.groups%20:raise ValueError('Use a positive multiple of 20 groups')
    project=Path(args.project);out=new_run(ROOT/args.out)
    sys.path.insert(0,str(project))
    from sourcefree_synth.renderer import default_parameters,render_once,QualityError
    fonts=read(project/'runs/feature_bank_pilot_01/font_selection.json')[:3]
    for f in fonts:
        if digest(f['path'])!=f['sha256'] or digest(f['license_path'])!=f['license_sha256']:
            raise ValueError('Font or license hash mismatch')
    count=args.groups;held=count//10
    assignments=['train']*(count*60//100)+['selection']*(count*15//100)+['calibration']*(count*10//100)+['test']*(count*15//100)
    np.random.default_rng(77491).shuffle(assignments)
    assignments+=['withheld_font']*held+['withheld_combination']*held
    write(out/'protocol.json',dict(policy=POLICY,seed=77491,grid=[3,10],groups=len(assignments),
        main_groups=count,main_requested_composition={'clean':0.2,'single':0.4,'combination':0.4},
        split='60/15/10/15 whole base groups assigned before rendering',fonts=fonts,
        withheld_font=fonts[2]['family'],withheld_exact_combination=['straight_interference','dot_scatter'],
        dimensions=[[160,64],[200,80],[240,96],[300,100]],
        renderer='unchanged Python clean text plus instrumented Pillow overlays; no PHP renderer admitted',
        training_eligible=False,target_data_access='none',positive_support_gate=1000,negative_support_gate=1000,
        gate_unit='unique retained parent images with at least one known local label, not overlapping crop count',
        limits=['engineering labels require review','near-duplicate audit pending','nine unsupported features','withheld renderer unavailable']))
    for rel in ['scripts/build_brick_corpus.py','feature_bank/labeled_bricks.py','feature_bank/regions.py','tests/test_labeled_bricks.py']:
        p=out/'code_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
    start=time.perf_counter();parents=[];rows=[];failures=[];seen={};conflicts=[]
    for group,partition in enumerate(assignments):
        rng=np.random.default_rng(913000+group);gid=f'base-{group:05d}';folder=out/gid;folder.mkdir()
        w,h=[(160,64),(200,80),(240,96),(300,100)][int(rng.integers(4))]
        dark=bool(rng.integers(2));bg=tuple(map(int,rng.integers(8,61,3) if dark else rng.integers(215,256,3)))
        fg=tuple(map(int,rng.integers(185,246,3) if dark else rng.integers(10,81,3)))
        font=fonts[2] if partition=='withheld_font' else fonts[int(rng.integers(2))]
        text=['Il1','OO0Q','ijIl'][group%3] if group%5==0 else ''.join(rng.choice(list(string.ascii_letters+string.digits),int(rng.integers(3,8))))
        profile=dict(id='generic-synthetic-candidate',width=w,height=h,background=[list(bg)],foreground=[list(fg)])
        p=default_parameters(0);p.update(background='flat',dots=0,lines=0,rotation=0,jitter=0,outline=False,shadow=False,
            multicolor=False,wave_x=0,wave_y=0,blur=0,height=float(rng.uniform(.40,.65)),width=float(rng.uniform(.65,.86)))
        ops=sample_operations(w,h,35000+group,dark)
        # The same distribution for every effect includes faint/near-background
        # cases; these must abstain, never become positives just by being enabled.
        for op in ops:
            if rng.random()<.15:
                op['color']=np.clip(np.array(bg)+rng.integers(-18,19,3),0,255).astype(int).tolist()
        recipe=dict(group=gid,partition=partition,text=text,font=font,profile=profile,parameters=p,seed=17000+group,operations=ops)
        write(folder/'recipe.json',recipe)
        try:base,mask,glyphs,meta=render_once(text,p,profile,[font],17000+group)
        except QualityError as exc:
            failures.append(dict(base_group=gid,partition=partition,reason=str(exc),stage='base'));continue
        base.save(folder/'clean.png');mask.save(folder/'text_mask.png');write(folder/'base_metadata.json',meta)
        # One clean, two different singles, and two combinations per base.
        # Hold out the exact straight+dots pair from every main partition.
        settings=[[],[FEATURES[group%3]],[FEATURES[(group+1)%3]],
                  [FEATURES[0],FEATURES[1]] if group%2==0 else [FEATURES[1],FEATURES[2]],list(FEATURES)]
        if partition=='withheld_combination':settings=[[FEATURES[0],FEATURES[2]]]
        rs=regions(w,h,scales=((3,10),));background=Image.new('RGB',(w,h),bg)
        for variant,enabled in enumerate(settings):
            vid=f'{gid}/{variant}';final,ev=evidence(base,ops,enabled,background)
            if pixel_hash(final)!=pixel_hash(compose(base,ops,enabled)):raise ValueError('Replay failed')
            final.save(folder/f'{variant}.png');verify_crops(final,rs)
            reference=compose(background,ops,enabled)
            delta=np.max(np.abs(np.asarray(final).astype('int16')-np.asarray(reference).astype('int16')),axis=2)
            retention=[float((delta[np.asarray(g)>100]>=12).mean()) for g in glyphs]
            accepted=min(retention)>=.55
            if not accepted:failures.append(dict(id=vid,partition=partition,reason='final text visibility below .55',retention=retention))
            np.savez_compressed(folder/f'{variant}_evidence.npz',**{f+'_'+k:ev[f][k] for f in FEATURES for k in ('raw','visible')})
            parent=dict(id=vid,base_group=gid,partition=partition,file=f'{vid}.png',enabled=enabled,
                parent_pixel_sha256=pixel_hash(final),accepted=accepted,text_visibility=retention,font=font['family'],size=[w,h],dark=dark)
            parents.append(parent)
            for r in rs:
                labels,details=crop_labels(ops,ev,r['box'])
                labels={f:labels.get(f,'unknown') if accepted else 'uncertain' if f in FEATURES else 'unknown' for f in VOCABULARY}
                ph=pixel_hash(final.crop(r['box']))
                row=dict(id=vid+'/'+r['id'],base_group=gid,parent_id=vid,partition=partition,parent_pixel_sha256=parent['parent_pixel_sha256'],
                    crop_pixel_sha256=ph,box=r['box'],parent_file=parent['file'],labels=labels,evidence=details,
                    training_eligible=False,retained=accepted,exclusion=None if accepted else 'parent_quality')
                rows.append(row);seen.setdefault(ph,[]).append(row)
        if (group+1)%100==0:print(f'{group+1}/{len(assignments)} groups rendered',flush=True)
    # Exclude every occurrence of an exact cross-partition crop duplicate. Keep
    # one canonical occurrence within a split, never count duplicate support.
    for ph,bucket in seen.items():
        eligible=[r for r in bucket if r['retained']]
        known={f:{r['labels'][f] for r in eligible}&{'present','absent'} for f in FEATURES}
        contradictory=[f for f,s in known.items() if len(s)>1]
        if contradictory:conflicts.append(dict(hash=ph,features=contradictory,ids=[r['id'] for r in eligible]))
        across=len({r['partition'] for r in eligible})>1
        for i,row in enumerate(eligible):
            reason='conflicting_labels' if contradictory else 'cross_partition_duplicate' if across else 'within_partition_duplicate' if i else None
            if reason:row.update(retained=False,exclusion=reason)
    check_group_partitions([r for r in rows if r['retained']])
    support={}
    for part in sorted(set(assignments)):
        rr=[r for r in rows if r['partition']==part and r['retained']]
        support[part]={f:{state:dict(crops=sum(r['labels'][f]==state for r in rr),
            parents=len({r['parent_id'] for r in rr if r['labels'][f]==state}),
            base_groups=len({r['base_group'] for r in rr if r['labels'][f]==state}))
            for state in ('present','absent','uncertain')} for f in FEATURES}
    counts={part:dict(collections.Counter('clean' if not p['enabled'] else 'single' if len(p['enabled'])==1 else 'combination'
             for p in parents if p['partition']==part and p['accepted'])) for part in sorted(set(assignments))}
    gate={f:all(support['train'][f][s]['parents']>=1000 for s in ('present','absent')) for f in FEATURES}
    write(out/'parents.json',parents);write(out/'rejections.json',failures);write(out/'support.json',support)
    write(out/'duplicate_audit.json',dict(conflicts=conflicts,exclusions=dict(collections.Counter(r['exclusion'] for r in rows if r['exclusion']))))
    with (out/'bricks.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    status=dict(state='CANDIDATE_REVIEW_REQUIRED',parents=len(parents),bricks=len(rows),retained_bricks=sum(r['retained'] for r in rows),
        failures=len(failures),composition=counts,distinct_parent_support_check=gate,conflicting_hashes=len(conflicts),
        training_started=False,full_corpus_gate_passed=False,elapsed_seconds=time.perf_counter()-start,
        renderer_sha256=digest(project/'sourcefree_synth/renderer.py'))
    write(out/'status.json',status)
    # Deterministic gallery spans partitions, sizes and polarities without
    # selecting according to label quality or any model prediction.
    selected=[]
    for part in sorted(set(assignments)):
        candidates=[p for p in parents if p['partition']==part]
        selected+=candidates[::max(1,len(candidates)//8)][:8]
    canvas=Image.new('RGB',(1200,1450),'#edf1f5');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),'Candidate corpus: deterministic samples, not classifier predictions',fill='black')
    for i,row in enumerate(selected):
        x=10+(i%4)*300;y=32+(i//4)*116
        draw.text((x,y),row['id']+' '+row['partition'],fill='black')
        with Image.open(out/row['file']) as im:
            im.thumbnail((290,88));canvas.paste(im,(x,y+18))
    canvas.save(out/'gallery.png');print(json.dumps(status,indent=2))

if __name__=='__main__':main()
