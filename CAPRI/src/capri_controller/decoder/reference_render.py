"""Bounded TRAIN reference rendering pilot; no embedding extraction/training."""
import collections
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from .common import read,write,sha,now
from .renderer import render,box
from .pilot import data_source_hashes

_FONTS=None
# Only layout size/spacing change on retry. Font, axes and characters are fixed.
SETTINGS=((512,.65,.18),(640,.75,.22),(768,.85,.25),
          (512,.5,.25),(768,.65,.35),(640,.9,.1))


def initialize(fonts):
    global _FONTS
    _FONTS=fonts


def recipe_for(record,attempt):
    if record['split']!='train':raise ValueError('Reference pilot is TRAIN-only')
    width,size,spacing=SETTINGS[attempt]
    text=record['text'];n=len(text)
    if n!=6:raise ValueError('Reference plan requires six known synthetic glyphs')
    return dict(schema='capri-font-reference-recipe-v1',split='train',attempt=attempt,
                specimen_id=record['specimen_id'],size=[width,128],background=[255,255,255],effects=[],
                render_seed=int(hashlib.sha256(record['specimen_id'].encode()).hexdigest()[:8],16),
                requested_count=n,text=dict(text=text,font_keys=[record['font_key']]*n,
                    axes=[dict(record['axes']) for _ in text],colours=[[20,20,20] for _ in text],
                    layout='line',size_fraction=size,size_jitter=0.,rotation=0.,stretch=1.,shear=0.,
                    stroke=0,spacing=spacing,x_position=.5,y_position=.5,jitter=0.,curve=0.))


def one(record):
    font=_FONTS[record['font_key']]
    if font['sha256']!=record['font_sha256'] or font['split']!='train' or font['split_group']!=record['family_group']:
        raise ValueError('Reference font provenance mismatch')
    if set(record['text'])-set(font['chars']):raise ValueError('Unsupported planned character')
    axes={a['tag']:a for a in font['axes']}
    if set(axes)!=set(record['axes']) or any(not a['min']<=record['axes'][k]<=a['max'] for k,a in axes.items()):
        raise ValueError('Invalid reference axes')
    errors=[]
    for attempt in range(len(SETTINGS)):
        recipe=recipe_for(record,attempt)
        try:
            out=render(recipe,_FONTS)
            if not out['valid']:
                errors.append(dict(attempt=attempt,reason='renderer_acceptance',quality=out['quality']))
                continue
            image=out.pop('image');background=out.pop('background_scene')
            arrays={k:np.rint(out.pop(k)*255).clip(0,255).astype(np.uint8) for k in ('geometry','visible','decoration')}
            assert out['count']==6 and out['rows']==1 and all(not e['present'] for e in out['effects'].values())
            for i,g in enumerate(out['glyphs']):
                if (g['char']!=record['text'][i] or g['font_sha256']!=record['font_sha256'] or g['axes']!=record['axes']
                    or g['box_core']!=box(arrays['geometry'][i].astype(np.float32)/255)
                    or g['box_visible']!=box(arrays['visible'][i].astype(np.float32)/255)):
                    raise AssertionError('Reference label/mask mismatch')
            pixels=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
            out.update(recipe=recipe,reference=record,pixel_sha256=pixels,rejections=errors)
            data={}
            for suffix,im in (('.png',image),('.background.png',background)):
                f=io.BytesIO();Image.fromarray(im).save(f,format='PNG');data[suffix]=f.getvalue()
            restored=np.asarray(Image.open(io.BytesIO(data['.png'])))
            assert np.array_equal(restored,image)
            f=io.BytesIO();np.savez_compressed(f,**arrays);data['.masks.npz']=f.getvalue()
            with np.load(io.BytesIO(data['.masks.npz']),allow_pickle=False) as again:
                assert all(np.array_equal(arrays[k],again[k]) for k in arrays)
            data['.json']=json.dumps(out,sort_keys=True,separators=(',',':')).encode()
            return dict(accepted=True,specimen_id=record['specimen_id'],attempt=attempt,quality=out['quality'],
                        pixel_sha256=pixels,payload=data)
        except (ValueError,OSError) as exc:
            errors.append(dict(attempt=attempt,reason=type(exc).__name__+':'+str(exc)[:200]))
    return dict(accepted=False,specimen_id=record['specimen_id'],rejections=errors)


def reference_pilot(run,count=96,workers=4):
    if not 8<=count<=512 or not 1<=workers<=8:raise ValueError('Bounded reference pilot: 8..512 images, 1..8 workers')
    validation=read(run/'font_reference_plans/validation.json')
    if not validation.get('passed'):raise ValueError('Validated reference plan required')
    matches=[p for p in (run/'font_reference_plans').glob('train_*.json') if sha(p)==validation['plan_sha256']]
    if len(matches)!=1:raise ValueError('Unique verified reference plan required')
    plan=read(matches[0]);manifest=run/'fonts/font_manifest.json';protocol=read(run/'corpus_protocol.json')
    if sha(manifest)!=plan['font_manifest_sha256'] or sha(manifest)!=protocol['font_manifest_sha256']:
        raise ValueError('Frozen font manifest mismatch')
    for path,digest in dict(protocol['source_hashes'],**protocol.get('data_asset_hashes',{})).items():
        if sha(path)!=digest:raise ValueError('Frozen rendering source mismatch: '+path)
    fonts={f['key']:f for f in read(manifest)['fonts'] if f['split']=='train'}
    records=plan['records'];chosen=set(np.linspace(0,len(records)-1,count,dtype=int).tolist())
    # Include the previously difficult math face if represented in TRAIN.
    difficult=next((i for i,r in enumerate(records) if 'jsmath-cmex10' in fonts[r['font_key']]['family'].lower()),None)
    if difficult is not None and difficult not in chosen:chosen.remove(max(chosen));chosen.add(difficult)
    tasks=[records[i] for i in sorted(chosen)]
    for key in {r['font_key'] for r in tasks}:
        if sha(fonts[key]['path'])!=fonts[key]['sha256']:raise ValueError('Reference font file hash mismatch')
    destination=run/('font_reference_pilot_'+str(time.time_ns()));destination.mkdir()
    provenance=dict(created_at=now(),plan_path=str(matches[0]),plan_sha256=sha(matches[0]),
                    sources=dict(data_source_hashes(),**{str(Path(__file__).resolve()):sha(__file__)}),
                    corpus_protocol_sha256=sha(run/'corpus_protocol.json'),specimens=[r['specimen_id'] for r in tasks])
    write(destination/'protocol.json',provenance)
    archive=destination/'references.zip.writing';rows=[];failures=[];previews=[];seen=set();duplicates=0
    start=time.monotonic();first=None
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(fonts,)) as pool:
            for index,result in enumerate(pool.map(one,tasks,chunksize=1)):
                if not result['accepted']:failures.append(result)
                else:
                    data=result.pop('payload');stem=f'reference_{index:05d}'
                    if first is None:first=(tasks[index],dict(data))
                    if len(previews)<16:previews.append((data['.png'],stem))
                    for suffix,content in data.items():z.writestr(stem+suffix,content)
                    duplicates+=int(result['pixel_sha256'] in seen);seen.add(result['pixel_sha256'])
                    rows.append(dict(result,stem=stem,files={stem+k:hashlib.sha256(v).hexdigest() for k,v in data.items()}))
                if (index+1)%16==0:
                    write(destination/'progress.json',dict(processed=index+1,accepted=len(rows),failed=len(failures),updated_at=now()))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Reference archive CRC failure')
        for row in rows:
            for name,digest in row['files'].items():
                if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Reference archive payload mismatch')
    initialize(fonts)
    if first:
        replay=one(first[0]);assert replay['accepted'] and replay['payload']==first[1]
    archive.rename(destination/'references.zip')
    sheet=Image.new('RGB',(1024,4*112),'white');draw=ImageDraw.Draw(sheet)
    for i,(content,stem) in enumerate(previews):
        im=Image.open(io.BytesIO(content)).convert('RGB');im.thumbnail((256,80))
        x=(i%4)*256;y=(i//4)*112;sheet.paste(im,(x,y+20));draw.text((x+4,y+3),stem,fill='black')
    sheet.save(destination/'preview.png')
    result=dict(created_at=now(),passed=not failures,diagnostic_only=True,requested=len(tasks),accepted=len(rows),
                failures=failures,duplicate_pixels=duplicates,deterministic_payload_replay=first is not None,
                attempt_counts=dict(collections.Counter(str(r['attempt']) for r in rows)),
                archive_sha256=sha(destination/'references.zip'),archive_bytes=(destination/'references.zip').stat().st_size,
                seconds=time.monotonic()-start,path=str(destination),trained_embedding_index=False,
                limitation='Bounded clean TRAIN library pilot; no model inference, font retrieval accuracy or full-reference coverage. Every rejected font/axis/string remains explicit; no substitution or quality-gate relaxation.')
    write(destination/'receipt.json',dict(result,records=rows));write(run/'font_reference_pilot.json',result)
    return result
