"""Bounded, resumable ZIP shards with accepted-image uniqueness and integrity receipts."""
import collections
import concurrent.futures
import hashlib
import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read, write, sha, now
from .pilot import initialize, pair, data_source_hashes


def pixel_hash(array):
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def summarize(label):
    recipe = label['recipe']
    counts = collections.defaultdict(collections.Counter)
    for key, value in [('length',label['count']), ('layout',recipe['text']['layout']),
                       ('split',recipe['split']), ('difficulty',label['quality']['difficulty']),
                       ('multicolour',len({tuple(c) for c in recipe['text']['colours']})>1),
                       ('pair_kind',recipe['pair_kind']), ('quota_family',recipe['quota_family'])]:
        counts[key][str(value)] += 1
    counts['font_family'].update({g['family_group'] for g in label['glyphs']})
    counts['font_file'].update({g['font_key'] for g in label['glyphs']})
    counts['requested_effect'].update(e['name'] for e in recipe['effects'])
    counts['requested_level'].update(str(e['level']) for e in recipe['effects'])
    for name, effect in label['effects'].items():
        if effect['present']:
            counts['effect_presence'][name] += 1
            counts['effect_levels'][name+'/'+str(effect['level'])] += 1
    return {key:dict(value) for key,value in counts.items()}


def encoded_pair(task):
    results = pair(task)
    if results[0].get('failed'):
        raise RuntimeError('Quota could not be rendered; no font/length substitution: '+json.dumps(results[0]))
    encoded = []
    for result in results:
        recipe, label, arrays = result['recipe'], result['label'], result['arrays']
        stem = f"{recipe['split']}_{recipe['index']:07d}"
        digest = pixel_hash(arrays['image'])
        label.update(recipe=recipe,pixel_sha256=digest,rejections=result['rejections'])
        if label['count'] != recipe['requested_count'] or not label['valid']:
            raise ValueError('Count/validity mismatch')
        payloads = {}
        for suffix, array in [('.png',arrays['image']),('.background.png',arrays['background_scene'])]:
            buf=io.BytesIO(); Image.fromarray(array).save(buf,format='PNG')
            payloads[stem+suffix]=buf.getvalue()
        restored = np.asarray(Image.open(io.BytesIO(payloads[stem+'.png'])))
        if pixel_hash(restored)!=digest:
            raise ValueError('PNG round trip changed image')
        masks={name:np.rint(arrays[name]*255).clip(0,255).astype(np.uint8)
               for name in ['geometry','visible','decoration']}
        buf=io.BytesIO();np.savez_compressed(buf,**masks)
        payloads[stem+'.masks.npz']=buf.getvalue()
        # Every published label box is checked against its exact serialized mask.
        from .renderer import box
        for i,glyph in enumerate(label['glyphs']):
            if (glyph['box_core']!=box(masks['geometry'][i].astype(np.float32)/255) or
                    glyph['box_visible']!=box(masks['visible'][i].astype(np.float32)/255)):
                raise ValueError('Serialized mask/box mismatch')
        label['background_annotation']='Underlying background after under-layer effects and geometric warps; excludes over-layer and final photometric effects. Observed background palette can also be measured outside glyph support in raw RGB.'
        payloads[stem+'.json']=json.dumps(label,separators=(',',':')).encode()
        encoded.append(dict(stem=stem,pixel_sha256=digest,payloads=payloads,summary=summarize(label),
                            attempt=recipe['attempt'],rejections=result['rejections']))
    return encoded


def merge_counts(destination, source):
    for name, values in source.items():
        destination[name].update(values)


def bounded_results(pool, tasks, window):
    tasks=iter(tasks); pending=collections.deque()
    for _ in range(window):
        try:pending.append(pool.submit(encoded_pair,next(tasks)))
        except StopIteration:break
    while pending:
        yield pending.popleft().result()
        try:pending.append(pool.submit(encoded_pair,next(tasks)))
        except StopIteration:pass


def recover(destination, index, protocol_hash):
    """A receipt is authoritative only with a matching archive and protocol."""
    completed={};counts=collections.defaultdict(collections.Counter)
    for receipt_path in sorted(destination.glob('*/*.receipt.json')):
        receipt=read(receipt_path)
        archive=receipt_path.with_name(receipt_path.name.replace('.receipt.json','.zip'))
        if receipt['protocol_hash']!=protocol_hash or not archive.exists() or sha(archive)!=receipt['sha256']:
            raise ValueError('Corrupt or incompatible shard: '+str(receipt_path))
        key=(receipt['split'],receipt['shard'])
        if key in completed:
            raise ValueError('Duplicate shard receipt')
        for row in receipt['images']:
            old=index.get(row['pixel_sha256'])
            expected=(row['stem'],str(archive))
            if old is not None and tuple(old)!=expected:
                raise ValueError('Duplicate images across completed shards')
            index[row['pixel_sha256']]=expected
        completed[key]=receipt
        merge_counts(counts,receipt['distributions'])
    expected_total=sum(len(r['images']) for r in completed.values())
    if len(index)!=expected_total:
        raise ValueError('Index contains rows without committed shard receipts')
    return completed,counts


def generate_to(run,destination,protocol,workers=8,crash_after_first_receipt=False):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    manifest=read(run/'fonts/font_manifest.json')
    if protocol['font_manifest_sha256']!=sha(run/'fonts/font_manifest.json') or protocol['source_hashes']!=data_source_hashes():
        raise ValueError('Frozen data source/manifest changed; do not mix renderer versions')
    index={}  # Rebuilt from durable per-shard receipts; bounded to 500k hashes.
    start=time.monotonic();fresh=0
    try:
        completed,counts=recover(destination,index,protocol_hash)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1,min(workers,12)),initializer=initialize,initargs=(manifest,)) as pool:
            for split,size in protocol['split_sizes'].items():
                if size%2 or protocol['shard_images']%2:
                    raise ValueError('Sibling pair must not straddle a split/shard')
                split_dir=destination/split;split_dir.mkdir(exist_ok=True)
                for shard,first in enumerate(range(0,size,protocol['shard_images'])):
                    if (split,shard) in completed:continue
                    if shutil.disk_usage(destination).free < protocol['reserve_bytes']+512*2**20:
                        raise OSError('Disk reserve reached; preserving existing data/checkpoints')
                    final=split_dir/f'shard_{shard:04d}.zip'
                    if final.exists():
                        # A crash after ZIP publication and before the receipt: preserve for audit.
                        final.rename(final.with_name(final.name+'.orphan.'+str(time.time_ns())))
                    temp=final.with_name(final.name+'.writing.'+str(os.getpid())+'.'+str(time.time_ns()))
                    last=min(size,first+protocol['shard_images'])
                    tasks=[(split,index//2) for index in range(first,last,2)]
                    rows=[];stats=collections.defaultdict(collections.Counter);rejects=collections.Counter()
                    try:
                        with zipfile.ZipFile(temp,'w',allowZip64=True) as archive:
                            for task,batch in zip(tasks,bounded_results(pool,tasks,max(2,workers*2))):
                                for retry in range(12):
                                    digests=[r['pixel_sha256'] for r in batch]
                                    duplicate=len(set(digests))!=2 or any(d in index for d in digests)
                                    if not duplicate:break
                                    rejects['duplicate_pair']+=1
                                    batch=pool.submit(encoded_pair,(*task,max(r['attempt'] for r in batch)+1)).result()
                                else:raise ValueError('Duplicate retry limit; quota preserved for diagnosis')
                                for result in batch:
                                    for name,content in result['payloads'].items():
                                        archive.writestr(name,content,compress_type=zipfile.ZIP_DEFLATED if name.endswith('.json') else zipfile.ZIP_STORED,
                                                         compresslevel=6 if name.endswith('.json') else None)
                                    index[result['pixel_sha256']]=(result['stem'],str(final))
                                    rows.append(dict(stem=result['stem'],pixel_sha256=result['pixel_sha256']))
                                    merge_counts(stats,result['summary']);rejects.update(result['rejections'])
                        with temp.open('rb+') as f:os.fsync(f.fileno())
                        with zipfile.ZipFile(temp) as check:
                            bad=check.testzip()
                            if bad or len(check.namelist())!=4*len(rows):raise ValueError('Shard archive verification failed: '+str(bad))
                        receipt=dict(created_at=now(),protocol_hash=protocol_hash,split=split,shard=shard,
                                     first=first,last_exclusive=last,images=rows,sha256=sha(temp),bytes=temp.stat().st_size,
                                     distributions={k:dict(v) for k,v in stats.items()},rejections=dict(rejects))
                        temp.rename(final)
                        write(final.with_suffix('.receipt.json'),receipt)
                        if crash_after_first_receipt:raise RuntimeError('TEST injected interruption after receipt')
                    except Exception:
                        raise
                    completed[(split,shard)]=receipt;merge_counts(counts,receipt['distributions']);fresh+=len(rows)
                    total=sum(len(r['images']) for r in completed.values())
                    progress=dict(updated_at=now(),state='running',accepted_unique=total,requested=sum(protocol['split_sizes'].values()),
                                  shards=len(completed),new_images_per_second=fresh/max(1,time.monotonic()-start),
                                  archive_bytes=sum(r['bytes'] for r in completed.values()),
                                  distributions={k:dict(v) for k,v in counts.items()},protocol_hash=protocol_hash)
                    write(destination/'progress.json',progress)
                    print(json.dumps({k:v for k,v in progress.items() if k!='distributions'}),flush=True)
        total=sum(len(r['images']) for r in completed.values())
        if total!=sum(protocol['split_sizes'].values()):raise ValueError('Corpus size mismatch')
        report=dict(updated_at=now(),state='complete',accepted_unique=total,split_sizes=protocol['split_sizes'],shards=len(completed),
                    archive_bytes=sum(r['bytes'] for r in completed.values()),protocol_hash=protocol_hash,
                    distributions={k:dict(v) for k,v in counts.items()})
        write(destination/'progress.json',report);write(destination/'completion.json',report)
        return {k:v for k,v in report.items() if k!='distributions'}
    finally:pass


def generate(run,workers=8):
    protocol=read(run/'corpus_protocol.json')
    ready=read(run/'production_readiness.json')
    if not ready['passed'] or ready['protocol_sha256']!=sha(run/'corpus_protocol.json'):
        raise ValueError('Readiness receipt missing or stale')
    return generate_to(run,run/'corpus',protocol,workers)


def check_storage(run,workers=2):
    protocol=dict(font_manifest_sha256=sha(run/'fonts/font_manifest.json'),source_hashes=data_source_hashes(),
                  split_sizes={'train':16,'dev':8,'calibration':8,'test':8},shard_images=8,reserve_bytes=2**30)
    dest=run/('storage_smoke_'+str(time.time_ns()))
    try:generate_to(run,dest,protocol,workers,crash_after_first_receipt=True)
    except RuntimeError as exc:
        if 'TEST injected' not in str(exc):raise
    else:raise AssertionError('Interruption test did not run')
    report=generate_to(run,dest,protocol,workers)
    before={str(p):sha(p) for p in dest.glob('*/*.zip')}
    again=generate_to(run,dest,protocol,workers)
    assert report['accepted_unique']==again['accepted_unique']==40
    assert before=={str(p):sha(p) for p in dest.glob('*/*.zip')}
    result=dict(created_at=now(),passed=True,accepted_unique=40,recovered_after_receipt_interruption=True,
                completed_shards_unchanged_on_resume=True,exported_masks_and_png_checked=True,
                measured_bytes_per_image=report['archive_bytes']/40,path=str(dest),source_hashes=data_source_hashes())
    write(run/'storage_checks.json',result)
    return result
