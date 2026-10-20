"""Audit immutable published shards while generation continues; never alters data."""
import collections
import hashlib
import json
import math
import random
import shutil
import time
import zipfile
from pathlib import Path
import numpy as np
from .common import read,write,sha,now
from .pilot import data_source_hashes


def association(table):
    groups=sorted(table);counts=list(map(str,range(2,9)))
    a=np.array([[table[g].get(c,0) for c in counts] for g in groups],dtype=float)
    total=a.sum()
    if not total:return dict(n=0,mutual_information_bits=None)
    p=a/total;expected=p.sum(1,keepdims=True)*p.sum(0,keepdims=True)
    ok=p>0
    information=float((p[ok]*np.log2(p[ok]/expected[ok])).sum())
    return dict(n=int(total),mutual_information_bits=information,
                table={g:{c:int(a[i,j]) for j,c in enumerate(counts)} for i,g in enumerate(groups)},
                note='Descriptive association on a fixed train sample; finite-sample MI has upward bias. Not causal evidence or a trained count benchmark.')


def audit(run,per_shard=32):
    run=Path(run);start=time.monotonic();protocol=read(run/'corpus_protocol.json')
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if protocol['source_hashes']!=data_source_hashes():raise ValueError('Frozen renderer source changed')
    if protocol['font_manifest_sha256']!=sha(run/'fonts/font_manifest.json'):raise ValueError('Frozen font manifest changed')
    fontmap={f['key']:f for f in read(run/'fonts/font_manifest.json')['fonts']}
    receipts=sorted((run/'corpus').glob('*/*.receipt.json'))
    if not receipts:raise ValueError('No published shards')
    seen=set();stems=set();total_bytes=0;total=0;stats=collections.defaultdict(collections.Counter)
    split_stats={};rejections=collections.Counter();joint=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    numeric=collections.defaultdict(list);examples=0;sample_receipts=[]
    for path in receipts:
        receipt=read(path);split=receipt['split'];rows=receipt['images'];n=len(rows)
        archive=path.with_name(path.name.replace('.receipt.json','.zip'))
        if receipt['protocol_hash']!=digest or sha(archive)!=receipt['sha256']:raise ValueError('Published shard checksum/protocol mismatch: '+str(path))
        if n!=receipt['last_exclusive']-receipt['first'] or receipt['distributions']['split']!={split:n}:
            raise ValueError('Shard count metadata mismatch')
        split_record=split_stats.setdefault(split,dict(images=0,shards=[],lengths=collections.Counter(),families=set(),font_files=set()))
        split_record['images']+=n;split_record['shards'].append(receipt['shard'])
        split_record['lengths'].update(receipt['distributions']['length'])
        split_record['families'].update(receipt['distributions']['font_family'])
        split_record['font_files'].update(receipt['distributions']['font_file'])
        for key in receipt['distributions']['font_file']:
            if key not in fontmap or fontmap[key]['split']!=split:raise ValueError('Font split leakage in published data')
        for row in rows:
            if row['pixel_sha256'] in seen or row['stem'] in stems:raise ValueError('Cross-shard duplicate')
            seen.add(row['pixel_sha256']);stems.add(row['stem'])
        for key,values in receipt['distributions'].items():stats[key].update(values)
        rejections.update(receipt['rejections']);total+=n;total_bytes+=receipt['bytes']
        sample_receipts.append(dict(path=str(path),sha256=receipt['sha256']))
        # No held-out labels are sampled for exploratory analysis.
        if split!='train':continue
        rng=random.Random(931+receipt['shard']);sample=rng.sample(rows,min(per_shard,n))
        with zipfile.ZipFile(archive) as z:
            for row in sample:
                label=json.loads(z.read(row['stem']+'.json'));recipe=label['recipe'];count=str(label['count'])
                if label['count']!=recipe['requested_count'] or not label['valid']:raise ValueError('Invalid accepted label')
                w,h=recipe['size'];glyphs=label['glyphs']
                for g in glyphs:
                    f=fontmap[g['font_key']]
                    if f['split']!='train' or g['char'] not in f['chars']:raise ValueError('Sampled font coverage/split mismatch')
                    if (g['visible_fraction']<.25 or g['clipped_fraction']>.05 or g['touches_canvas_edge'] or
                        g['visible_pixels']<10 or g['local_contrast']<12):raise ValueError('Quality gate mismatch')
                for key,value in [('native_height',str(h)),('aspect_bin',str(min(5,math.floor(w/h)))),
                                  ('layout',recipe['text']['layout']),('difficulty',label['quality']['difficulty']),
                                  ('requested_effect_count',str(len(recipe['effects']))),
                                  ('multicolour',str(len({tuple(c) for c in recipe['text']['colours']})>1))]:
                    joint[key][value][count]+=1
                numeric['attempt_index'].append(recipe['attempt'])
                for name,value in label['quality'].items():
                    if name!='difficulty':numeric[name].append(value)
                numeric['rotation_requested'].append(recipe['text']['rotation'])
                numeric['x_position_requested'].append(recipe['text']['x_position'])
                numeric['y_position_requested'].append(recipe['text']['y_position'])
                examples+=1
    summaries={}
    for split,record in split_stats.items():
        expected_files={k for k,f in fontmap.items() if f['split']==split}
        expected_families={fontmap[k]['family_group'] for k in expected_files}
        assert record['shards']==list(range(len(record['shards']))),'Published shard gap'
        lengths=record['lengths'];spread=max(lengths.values())-min(lengths.values())
        assert set(lengths)==set(map(str,range(2,9))) and spread<=2,'Count anchor imbalance'
        summaries[split]=dict(images=record['images'],shards=len(record['shards']),lengths=dict(lengths),max_count_spread=spread,
                             families_seen=len(record['families']),families_available=len(expected_families),
                             font_files_seen=len(record['font_files']),font_files_available=len(expected_files),
                             missing_font_files=sorted(expected_files-record['font_files']),
                             missing_families=sorted(expected_families-record['families']))
    free=shutil.disk_usage(run).free;remaining=max(0,protocol['total_accepted_unique']-total)
    projected=remaining*total_bytes/total*1.2
    alerts=[]
    if free<projected+protocol['reserve_bytes']:alerts.append('Projected remaining corpus plus checkpoint reserve exceeds free disk')
    if total>=10000 and len(stats['effect_presence'])!=37:alerts.append('Missing effect-head coverage')
    report=dict(created_at=now(),protocol_hash=digest,published_shards=len(receipts),accepted_unique=total,
                independently_hashed_archive_bytes=total_bytes,split_coverage=summaries,train_sample_size=examples,
                train_count_associations={key:association(table) for key,table in joint.items()},
                sample_quantiles={key:{str(q):float(np.quantile(values,q)) for q in (0,.1,.5,.9,1)} for key,values in numeric.items()},
                distributions={key:dict(value) for key,value in stats.items() if key not in ('font_family','font_file','quota_family')},
                inherited_pair_rejection_mentions=dict(rejections),alerts=alerts,free_GiB=free/2**30,
                remaining_corpus_projection_with_20pct_margin_GiB=projected/2**30,checkpoint_reserve_GiB=protocol['reserve_bytes']/2**30,
                receipt_snapshot_sha256=hashlib.sha256(json.dumps(sample_receipts,sort_keys=True).encode()).hexdigest(),
                seconds=time.monotonic()-start,
                limitations=['Snapshot only; additional shards can finish during this audit.',
                             'Rejection mentions are copied into both siblings and are not unique rejected-recipe counts.',
                             'Quality/association samples use train labels only; all published archive hashes and receipt-level uniqueness/splits were checked.',
                             'Presence/intensity/layout acceptance is stochastic. This audit does not claim exact joint balancing or remove intrinsically ambiguous images.'])
    dest=run/'audits'/('corpus_'+str(time.time_ns())+'.json');write(dest,report)
    write(run/'corpus_audit_latest.json',dict(path=str(dest),accepted_unique=total,alerts=alerts,created_at=report['created_at']))
    return dict(path=str(dest),accepted_unique=total,shards=len(receipts),train_sample_size=examples,
                split_coverage=summaries,association_bits={key:value['mutual_information_bits'] for key,value in report['train_count_associations'].items()},
                alerts=alerts,seconds=report['seconds'])
