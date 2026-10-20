"""Bounded diagnostic corpus with images, separate overlays, masks and coverage report."""
import collections
import hashlib
import io
import json
import time
from pathlib import Path
import concurrent.futures
import numpy as np
from PIL import Image, ImageDraw
from .common import read, write, now, sha
from .sampling import Sampler
from .renderer import render

_SAMPLER=None


def initialize(manifest):
    global _SAMPLER
    _SAMPLER=Sampler(manifest)


def one(task):
    split,index=task
    errors=[]
    for attempt in range(30):
        recipe=_SAMPLER.recipe(split,index,attempt)
        try:
            out=render(recipe,_SAMPLER.fonts)
            if not out['valid']:
                errors.append('visibility_or_clipping');continue
            arrays={k:out.pop(k) for k in ['image','geometry','visible','decoration','background_scene']}
            return dict(recipe=recipe,label=out,arrays=arrays,rejections=errors)
        except (ValueError,OSError) as exc:
            errors.append(type(exc).__name__+':'+str(exc)[:120])
    return dict(failed=True,split=split,index=index,rejections=errors)


def pair(task):
    split,group=task[:2]
    start_attempt=task[2] if len(task)>2 else 0
    errors=[]
    for attempt in range(start_attempt,start_attempt+80):
        results=[]
        try:
            for index in [group*2,group*2+1]:
                recipe=_SAMPLER.recipe(split,index,attempt)
                out=render(recipe,_SAMPLER.fonts)
                if not out['valid']:
                    raise ValueError('visibility_contrast_size_or_clipping')
                arrays={k:out.pop(k) for k in ['image','geometry','visible','decoration','background_scene']}
                results.append(dict(recipe=recipe,label=out,arrays=arrays,rejections=errors.copy()))
            return results
        except (ValueError,OSError) as exc:
            errors.append(type(exc).__name__+':'+str(exc)[:120])
    return [dict(failed=True,split=split,index=group*2+i,rejections=errors) for i in range(2)]


def generate(run,count=256,workers=8):
    if not 8<=count<=5000 or count%8:
        raise ValueError('Pilot must contain 8..5000 cases in multiples of 8; production is a separate gated stage')
    manifest=read(run/'fonts/font_manifest.json')
    dest=run/f'pilot_{count:05d}'
    if dest.exists():
        raise ValueError('Pilot already exists; preserve it and choose another count')
    dest.mkdir()
    start=time.monotonic()
    counts=collections.defaultdict(collections.Counter)
    failed=[];hashes=set();duplicates=0;gallery=[];bytes_total=0
    # Cover the whole bank, not only the first few contiguous quota families.
    sampler=Sampler(manifest)
    first_cycle=[(split, family_index*7+family_index%7)
                 for split in sampler.families for family_index in range(len(sampler.families[split]))]
    order=np.random.default_rng(811).permutation(len(first_cycle))
    tasks=[]
    for i in range(count//2):
        split,group=first_cycle[int(order[i%len(order)])]
        tasks.append((split,group+(i//len(order))*7*len(sampler.families[split])))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1,min(workers,12)),initializer=initialize,initargs=(manifest,)) as pool:
        import itertools
        for n,result in enumerate(itertools.chain.from_iterable(pool.map(pair,tasks)),1):
            if result.get('failed'):
                failed.append(result);continue
            recipe,label,arrays=result['recipe'],result['label'],result['arrays']
            stem=f"{recipe['split']}_{recipe['index']:06d}"
            image=Image.fromarray(arrays['image'])
            digest=hashlib.sha256(arrays['image'].tobytes()).hexdigest()
            duplicates+=int(digest in hashes);hashes.add(digest)
            image.save(dest/(stem+'.png'))
            np.savez_compressed(dest/(stem+'.masks.npz'),**{k:np.rint(arrays[k]*255).clip(0,255).astype(np.uint8) for k in ['geometry','visible','decoration']})
            label.update(recipe=recipe,pixel_sha256=digest,rejections=result['rejections'])
            write(dest/(stem+'.json'),label)
            counts['length'][str(label['count'])]+=1
            counts['layout'][recipe['text']['layout']]+=1
            counts['split'][recipe['split']]+=1
            counts['multicolour'][str(len({tuple(c) for c in recipe['text']['colours']})>1)]+=1
            counts['pair_kind'][str(recipe['pair_kind'])]+=1
            counts['difficulty'][label['quality']['difficulty']]+=1
            counts['rejections'].update(result['rejections'])
            for effect,values in label['effects'].items():
                if values['present']:
                    counts['effect_presence'][effect]+=1
                    counts['effect_levels'][effect+'/'+str(values['level'])]+=1
            counts['font_family'].update({g['family_group'] for g in label['glyphs']})
            if len(gallery)<48:
                overlay=image.copy();d=ImageDraw.Draw(overlay)
                for g in label['glyphs']:
                    d.rectangle(g['box_core'],outline=(255,0,0),width=1)
                gallery.append((stem,image,overlay))
            if n%32==0:
                print(json.dumps(dict(pilot_processed=n,total=count,unique=len(hashes))),flush=True)
    cellw,cellh=520,170
    sheet=Image.new('RGB',(cellw*2,cellh*len(gallery)),(238,238,238))
    draw=ImageDraw.Draw(sheet)
    for i,(stem,im,ov) in enumerate(gallery):
        for j,pic in enumerate([im,ov]):
            pic=pic.copy();pic.thumbnail((cellw-12,cellh-30))
            sheet.paste(pic,(j*cellw+6,i*cellh+24))
        draw.text((6,i*cellh+5),stem+' | raw image / annotation preview',fill='black')
    sheet.save(dest/'contact_sheet.png')
    bytes_total=sum(p.stat().st_size for p in dest.iterdir())
    elapsed=time.monotonic()-start
    report=dict(created_at=now(),requested=count,accepted=len(hashes)+duplicates,unique=len(hashes),duplicates=duplicates,
                failed=failed,seconds=elapsed,images_per_second=(len(hashes)+duplicates)/elapsed,
                bytes=bytes_total,estimated_500k_GiB=(bytes_total/max(1,len(hashes)))*500000/2**30,
                distributions={k:dict(v) for k,v in counts.items()},font_manifest_sha256=sha(run/'fonts/font_manifest.json'),
                source_hashes=data_source_hashes(),
                production_ready=False,note='Diagnostic pilot; distribution, masks, pair integrity and storage still require review')
    write(dest/'report.json',report)
    return {k:v for k,v in report.items() if k not in ['distributions','failed']}


def data_source_hashes():
    from .common import ROOT
    paths=[ROOT/'src/capri_controller/decoder'/name for name in ('renderer.py','sampling.py','pilot.py','common.py')]
    # The runtime loads external effects; record every imported package source too.
    from captcha_effects import __file__ as effects_file
    paths+=list(Path(effects_file).parent.rglob('*.py'))
    return {str(p.resolve()):sha(p) for p in paths}
