"""Clean DEV font-library pixels for selection-time retrieval evaluation only.

Never adds images to the 500k corpus or to TRAIN references. Known library
strings/boxes remain labels; subsequent embedding extraction must be pixel-only.
"""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import shutil
import string
import time
import zipfile
import numpy as np
from PIL import Image
from .common import read,write,sha,now
from .renderer import render,box

SETTINGS=((512,.65,.18),(640,.75,.22),(768,.85,.25),(512,.5,.25),(768,.65,.35),(640,.9,.1),
          (512,.4,.18),(512,.3,.18),(512,.2,.18))
_FONTS=None


def initialize(fonts):
    global _FONTS
    _FONTS=fonts


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def plan(fonts):
    alphabet=''.join((string.digits[i] if i<10 else '')+string.ascii_uppercase[i]+string.ascii_lowercase[i] for i in range(26))
    records=[]
    for font in sorted(fonts,key=lambda f:f['key']):
        if font['split']!='dev':raise ValueError('Only DEV fonts belong in this evaluation gallery')
        supported=''.join(c for c in alphabet if c in font['chars'])
        if len(supported)<2:raise ValueError('Insufficient DEV font coverage')
        defaults={a['tag']:a['default'] for a in font['axes']};variants=[('default',defaults)]
        for axis in font['axes']:
            if axis['tag'] not in ('wght','wdth','ital','slnt'):continue
            for endpoint in ('min','max'):
                if axis[endpoint]!=axis['default']:variants.append((axis['tag']+'_'+endpoint,dict(defaults,**{axis['tag']:axis[endpoint]})))
        for variant,axes in variants:
            for page in range(12):
                records.append(dict(specimen_id='dev/'+font['key']+'/'+variant+'/'+str(page),font_key=font['key'],
                    font_sha256=font['sha256'],family_group=font['split_group'],split='dev',role='selection_library',
                    style=font['style'] or 'unspecified',axes=axes,text=''.join(supported[(page*6+j)%len(supported)] for j in range(6))))
    if not records:raise ValueError('Empty DEV gallery')
    return records


def inputs(run):
    manifest=run/'fonts/font_manifest.json';corpus=read(run/'corpus_protocol.json')
    if sha(manifest)!=corpus['font_manifest_sha256']:raise ValueError('Frozen manifest mismatch')
    fonts={f['key']:f for f in read(manifest)['fonts'] if f['split']=='dev'}
    for f in fonts.values():
        if sha(f['path'])!=f['sha256']:raise ValueError('DEV font file changed')
    dependencies=dict(corpus['source_hashes'],**corpus.get('data_asset_hashes',{}))
    helper=str(Path(__file__).with_name('common.py').resolve())
    # Corpus stays immutable. Only the inspected storage-path helper changed
    # during the authorized E-drive cutover; this gallery binds that exact version.
    if dependencies[helper]!='75ca9373a60df2701609e2f7eb0a4a286214276a4e3ff2f7ff7b1de74f213e6b':
        raise ValueError('Unexpected original corpus helper')
    dependencies[helper]='6cf8189a9da344daa127e07cf957fff3629edf2fa86751e22b77e7c3fc7e35e8'
    for path,value in dependencies.items():
        if sha(path)!=value:raise ValueError('Frozen rendering dependency changed: '+path)
    records=plan(fonts.values())
    protocol=dict(schema='capri-dev-font-gallery-v1',split='dev',role='selection_library',font_manifest_sha256=sha(manifest),
        source_sha256=sha(__file__),dependencies=dependencies,original_corpus_protocol_sha256=sha(run/'corpus_protocol.json'),
        helper_amendment='Exact reviewed E-drive storage helper; original corpus protocol preserved.',
        records=records,fonts=len(fonts),families=len({f['split_group'] for f in fonts.values()}),
        shard_size=128,layout_attempts=SETTINGS,
        policy='DEV-only known library; never optimization. Reuse existing TRAIN references as evaluation distractors; no calibration/test access. No font/axes/text substitution; preserve ambiguity.',
        heldout_queries='Existing synthetic DEV corpus; no query text/count/family enters the visual forward or retrieval inputs.')
    return fonts,records,json.loads(json.dumps(protocol))


def one(record):
    font=_FONTS[record['font_key']]
    if (record['split']!='dev' or font['split']!='dev' or record['role']!='selection_library'
        or record['font_sha256']!=font['sha256'] or record['family_group']!=font['split_group']):
        raise ValueError('DEV gallery font/split provenance mismatch')
    if len(record['text'])!=6 or set(record['text'])-set(font['chars']):raise ValueError('Invalid known specimen text')
    axes={a['tag']:a for a in font['axes']}
    if set(axes)!=set(record['axes']) or any(not a['min']<=record['axes'][k]<=a['max'] for k,a in axes.items()):
        raise ValueError('Invalid DEV font axes')
    errors=[]
    for attempt,(width,size,spacing) in enumerate(SETTINGS):
        recipe=dict(schema='capri-dev-font-reference-v1',split='dev',role='selection_library',attempt=attempt,
            specimen_id=record['specimen_id'],size=[width,128],background=[255,255,255],effects=[],
            render_seed=int(hashlib.sha256(record['specimen_id'].encode()).hexdigest()[:8],16),requested_count=6,
            text=dict(text=record['text'],font_keys=[record['font_key']]*6,axes=[dict(record['axes']) for _ in range(6)],
                colours=[[20,20,20] for _ in range(6)],layout='line',size_fraction=size,size_jitter=0.,rotation=0.,stretch=1.,
                shear=0.,stroke=0,spacing=spacing,x_position=.5,y_position=.5,jitter=0.,curve=0.))
        try:
            out=render(recipe,_FONTS)
            if not out['valid']:errors.append(dict(attempt=attempt,reason='acceptance',quality=out['quality']));continue
            image=out.pop('image');out.pop('background_scene')
            arrays={k:np.rint(out.pop(k)*255).clip(0,255).astype(np.uint8) for k in ('geometry','visible','decoration')}
            assert out['count']==6 and out['rows']==1 and all(not e['present'] for e in out['effects'].values())
            for i,g in enumerate(out['glyphs']):
                if (g['char']!=record['text'][i] or g['font_sha256']!=record['font_sha256'] or g['axes']!=record['axes']
                    or g['box_core']!=box(arrays['geometry'][i].astype(np.float32)/255)
                    or g['box_visible']!=box(arrays['visible'][i].astype(np.float32)/255)):
                    raise AssertionError('DEV reference instance-label mismatch')
            pixel_sha=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
            out.update(recipe=recipe,reference=record,pixel_sha256=pixel_sha,rejections=errors)
            stream=io.BytesIO();Image.fromarray(image).save(stream,format='PNG');png=stream.getvalue()
            assert np.array_equal(np.asarray(Image.open(io.BytesIO(png))),image)
            return dict(accepted=True,specimen_id=record['specimen_id'],attempt=attempt,pixel_sha256=pixel_sha,
                payload={'.png':png,'.json':json.dumps(out,sort_keys=True,separators=(',',':')).encode()})
        except (ValueError,OSError) as exc:errors.append(dict(attempt=attempt,reason=repr(exc)))
    return dict(accepted=False,specimen_id=record['specimen_id'],rejections=errors)


def check_gallery(run,workers=2):
    if not 1<=workers<=4:raise ValueError('DEV pilot uses at most four CPU workers')
    fonts,records,protocol=inputs(run);destination=run/'dev_font_gallery_pilot';destination.mkdir(exist_ok=True)
    chosen=[records[i] for i in sorted(set(np.linspace(0,len(records)-1,32,dtype=int)))]
    initialize(fonts)
    try:plan([dict(next(iter(fonts.values())),split='train')])
    except ValueError:pass
    else:raise AssertionError('TRAIN fonts admitted to DEV gallery')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(fonts,)) as pool:
        results=list(pool.map(one,chosen,chunksize=1))
    failures=[{k:v for k,v in r.items() if k!='payload'} for r in results if not r['accepted']]
    if failures:
        write(destination/'failures.json',failures);raise RuntimeError('DEV reference pilot has rejected specimens')
    archive=destination/('references_'+str(time.time_ns())+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for i,result in enumerate(results):
            for suffix,value in result['payload'].items():z.writestr(f'dev_reference_{i:05d}'+suffix,value)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==2*len(results)
    assert one(chosen[0])==results[0]
    result=dict(created_at=now(),passed=True,protocol_hash=digest(protocol),source_sha256=sha(__file__),
        fonts=protocol['fonts'],families=protocol['families'],planned=len(records),pilot_accepted=len(results),
        strict_dev_split=True,instance_box_checks=True,png_roundtrip=True,deterministic_replay=True,
        archive=str(archive),archive_sha256=sha(archive),
        limitation='CPU rendering pilot only; no trained embeddings, DEV retrieval accuracy, optimization or corpus regeneration.')
    write(destination/'receipt.json',result);write(run/'dev_font_gallery_pilot.json',result);return result


def generate_gallery(run,workers=2):
    if not 1<=workers<=4:raise ValueError('DEV gallery uses at most four CPU workers')
    fonts,records,protocol=inputs(run);pilot=read(run/'dev_font_gallery_pilot.json')
    ph=digest(protocol)
    if not pilot['passed'] or pilot['protocol_hash']!=ph or sha(pilot['archive'])!=pilot['archive_sha256']:
        raise ValueError('Matching verified DEV pilot required')
    root=run/'font_gallery_dev_v1';root.mkdir(exist_ok=True)
    if (root/'protocol.json').exists() and read(root/'protocol.json')!=protocol:raise ValueError('DEV gallery protocol changed')
    write(root/'protocol.json',protocol);completed=0;shards=0;seen=set();duplicates=0
    for path in sorted(root.glob('shard_*.receipt.json')):
        r=read(path);archive=path.with_name(path.name.replace('.receipt.json','.zip'))
        if r['protocol_hash']!=ph or r['first']!=completed or sha(archive)!=r['sha256']:
            raise ValueError('DEV gallery checkpoint mismatch')
        if [x['specimen_id'] for x in r['records']]!=[x['specimen_id'] for x in records[completed:completed+len(r['records'])]]:
            raise ValueError('DEV gallery specimen coverage mismatch')
        for row in r['records']:duplicates+=int(row['pixel_sha256'] in seen);seen.add(row['pixel_sha256'])
        completed+=len(r['records']);shards+=1
    def progress(state,**extra):
        value=dict(updated_at=now(),state=state,split='dev',role='selection_library',accepted=completed,total=len(records),
            shards=shards,duplicate_pixels=duplicates,protocol_hash=ph,**extra)
        write(root/'progress.json',value);return value
    # Retain incompletely published archives; committed receipts alone advance.
    for archive in root.glob('shard_*.zip'):
        if not archive.with_name(archive.stem+'.receipt.json').exists():archive.rename(archive.with_name(archive.name+'.orphan.'+str(time.time_ns())))
    progress('running')
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(fonts,)) as pool:
            while completed<len(records):
                if shutil.disk_usage(root).free<10*1024**3:raise OSError('Preserve 10 GiB disk reserve')
                batch=records[completed:completed+protocol['shard_size']];rows=[]
                archive=root/f'shard_{shards:05d}.zip';temporary=archive.with_name(archive.name+'.writing.'+str(time.time_ns()))
                if archive.exists():raise FileExistsError('Preserve committed DEV archive')
                with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as z:
                    for i,result in enumerate(pool.map(one,batch,chunksize=1)):
                        if not result['accepted']:
                            write(root/('failure_'+str(time.time_ns())+'.json'),result);raise RuntimeError('Rejected DEV specimen: '+result['specimen_id'])
                        payload=result.pop('payload');stem=f'dev_reference_{completed+i:07d}'
                        for suffix,value in payload.items():z.writestr(stem+suffix,value)
                        rows.append(dict(result,stem=stem))
                with zipfile.ZipFile(temporary) as z:
                    if z.testzip() is not None or len(z.namelist())!=2*len(rows):raise ValueError('DEV gallery archive validation failed')
                receipt=dict(first=completed,records=rows,protocol_hash=ph,sha256=sha(temporary),bytes=temporary.stat().st_size)
                temporary.rename(archive);write(archive.with_name(archive.stem+'.receipt.json'),receipt)
                for row in rows:duplicates+=int(row['pixel_sha256'] in seen);seen.add(row['pixel_sha256'])
                completed+=len(rows);shards+=1;print(json.dumps(progress('running')),flush=True)
        result=progress('complete');write(root/'completion.json',result);return result
    except Exception as exc:progress('failed',error=repr(exc));raise
