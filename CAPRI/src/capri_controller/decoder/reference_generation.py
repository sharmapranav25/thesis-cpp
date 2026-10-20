"""Resumable clean font references, isolated from the completed decoder corpus."""
import concurrent.futures
import collections
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile
from .common import read,write,sha,now
from .reference_render import initialize,one


def bounded_results(pool,tasks,window):
    iterator=iter(tasks);pending=collections.deque()
    for _ in range(window):
        task=next(iterator,None)
        if task is None:break
        pending.append(pool.submit(one,task))
    while pending:
        yield pending.popleft().result()
        task=next(iterator,None)
        if task is not None:pending.append(pool.submit(one,task))


def inputs(run):
    validation=read(run/'font_reference_plans/validation.json')
    if not validation.get('passed'):raise ValueError('Reference plan validation required')
    plans=[p for p in (run/'font_reference_plans').glob('train_*.json') if sha(p)==validation['plan_sha256']]
    if len(plans)!=1:raise ValueError('Unique verified plan required')
    plan=read(plans[0]);manifest=run/'fonts/font_manifest.json';corpus=read(run/'corpus_protocol.json')
    if sha(manifest)!=plan['font_manifest_sha256'] or sha(manifest)!=corpus['font_manifest_sha256']:
        raise ValueError('Reference font manifest mismatch')
    for path,digest in dict(corpus['source_hashes'],**corpus.get('data_asset_hashes',{})).items():
        if sha(path)!=digest:raise ValueError('Frozen rendering dependency changed')
    fonts={f['key']:f for f in read(manifest)['fonts'] if f['split']=='train'}
    for f in fonts.values():
        if sha(f['path'])!=f['sha256']:raise ValueError('Font file hash mismatch')
    return plans[0],plan,fonts


def generate_set(destination,records,fonts,protocol,workers=4,shard_size=500,stop_after_shards=None):
    if not 1<=workers<=8 or not records:raise ValueError('Invalid reference generation request')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    pp=destination/'protocol.json'
    if pp.exists():
        if read(pp)!=protocol:raise ValueError('Reference resume protocol mismatch')
    else:write(pp,protocol)
    completed=0;seen=set();duplicates=0;shards=0
    for receipt_path in sorted(destination.glob('shard_*.receipt.json')):
        receipt=read(receipt_path);archive=receipt_path.with_name(receipt_path.name.replace('.receipt.json','.zip'))
        if receipt['protocol_hash']!=digest or receipt['first']!=completed or sha(archive)!=receipt['sha256']:
            raise ValueError('Reference shard identity/continuity/hash mismatch')
        expected=records[completed:completed+len(receipt['records'])]
        if not expected or [x['specimen_id'] for x in expected]!=[x['specimen_id'] for x in receipt['records']]:
            raise ValueError('Reference specimen order mismatch')
        for row in receipt['records']:
            duplicates+=int(row['pixel_sha256'] in seen);seen.add(row['pixel_sha256'])
        completed+=len(expected);shards+=1
    def progress(state,**extra):
        value=dict(updated_at=now(),state=state,accepted_references=completed,planned=len(records),shards=shards,
                   duplicate_reference_pixels=duplicates,protocol_hash=digest,trained_embedding_index=False,**extra)
        write(destination/'progress.json',value);return value
    # An archive without its receipt is incomplete publication, not a committed
    # shard. Preserve it under an orphan name before recreating that shard.
    for archive in destination.glob('shard_*.zip'):
        if not archive.with_name(archive.stem+'.receipt.json').exists():
            archive.rename(archive.with_name(archive.name+'.orphan.'+str(time.time_ns())))
    progress('running')
    try:
        if completed<len(records):
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(fonts,)) as pool:
                while completed<len(records):
                    if shutil.disk_usage(destination).free<30*2**30:raise OSError('Reference generation preserves 30 GiB free for training')
                    batch=records[completed:completed+shard_size];first=completed
                    archive=destination/f'shard_{shards:05d}.zip'
                    if archive.exists():raise FileExistsError('Preserve completed reference archive')
                    temporary=archive.with_name(archive.name+'.writing.'+str(time.time_ns()));rows=[]
                    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as z:
                        for index,result in enumerate(bounded_results(pool,batch,max(2,workers*2))):
                            if not result['accepted']:
                                write(destination/('failure_'+str(time.time_ns())+'.json'),result)
                                raise RuntimeError('Reference rejected without font/axis substitution: '+result['specimen_id'])
                            payload=result.pop('payload');stem=f'reference_{first+index:07d}'
                            for suffix,content in payload.items():z.writestr(stem+suffix,content)
                            rows.append(dict(result,stem=stem))
                    with zipfile.ZipFile(temporary) as z:
                        if z.testzip() is not None:raise ValueError('Reference archive CRC failure')
                        if len(z.namelist())!=4*len(rows):raise ValueError('Reference payload count mismatch')
                    receipt=dict(first=first,records=rows,protocol_hash=digest,sha256=sha(temporary),bytes=temporary.stat().st_size)
                    temporary.rename(archive);write(archive.with_name(archive.stem+'.receipt.json'),receipt)
                    for row in rows:
                        duplicates+=int(row['pixel_sha256'] in seen);seen.add(row['pixel_sha256'])
                    completed+=len(rows);shards+=1
                    print(json.dumps(progress('running')),flush=True)
                    if stop_after_shards is not None and shards>=stop_after_shards:return progress('diagnostic_stop')
        result=progress('complete');write(destination/'completion.json',result);return result
    except Exception as exc:
        progress('failed',error=repr(exc));raise


def generate_references(run,workers=4):
    plan_path,plan,fonts=inputs(run)
    pilot=read(run/'font_reference_pilot.json');storage=read(run/'reference_storage_checks.json')
    if not pilot['passed'] or pilot['accepted']<96 or not storage['passed']:raise ValueError('Reference pilot/storage readiness required')
    provenance=read(Path(pilot['path'])/'protocol.json')
    for path,digest in provenance['sources'].items():
        if sha(path)!=digest:raise ValueError('Reference pilot source changed')
    if sha(Path(pilot['path'])/'references.zip')!=pilot['archive_sha256']:raise ValueError('Reference pilot hash mismatch')
    if storage['generation_source_sha256']!=sha(__file__):raise ValueError('Reference storage tests are stale')
    protocol=dict(schema='capri-font-references-v1',plan_sha256=sha(plan_path),font_manifest_sha256=plan['font_manifest_sha256'],
                  source_sha256=sha(__file__),pilot_sources=provenance['sources'],total=len(plan['records']),shard_size=500,
                  split='train',role='Clean library references only; not added to the 500000-image corpus',
                  rejection_policy='Fail with explicit specimen identity; preserve partial data; never substitute font, axis or text')
    return generate_set(run/'font_references_train_v1',plan['records'],fonts,protocol,workers)


def check_reference_storage(run):
    plan_path,plan,fonts=inputs(run);records=plan['records'][:16]
    destination=run/('reference_storage_smoke_'+str(time.time_ns()))
    protocol=dict(diagnostic=True,plan_sha256=sha(plan_path),specimens=[r['specimen_id'] for r in records])
    first=generate_set(destination,records,fonts,protocol,workers=2,shard_size=8,stop_after_shards=1)
    assert first['accepted_references']==8
    before=sha(destination/'shard_00000.zip')
    orphan=destination/'shard_00001.zip.writing.interrupted';orphan.write_bytes(b'diagnostic partial archive')
    second=generate_set(destination,records,fonts,protocol,workers=2,shard_size=8)
    assert second['accepted_references']==16 and sha(destination/'shard_00000.zip')==before and orphan.exists()
    hashes={p.name:sha(p) for p in destination.glob('shard_*.zip')}
    third=generate_set(destination,records,fonts,protocol,workers=2,shard_size=8)
    assert third['accepted_references']==16 and hashes=={p.name:sha(p) for p in destination.glob('shard_*.zip')}
    result=dict(created_at=now(),passed=True,interrupted_resume=True,unchanged_completed_archives=True,
                partial_preserved=True,path=str(destination),generation_source_sha256=sha(__file__))
    write(run/'reference_storage_checks.json',result);return result
