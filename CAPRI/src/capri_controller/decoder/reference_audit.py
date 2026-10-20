"""Independent published reference audit; no model or target answers."""
import collections
import hashlib
import io
import json
import time
import zipfile
import numpy as np
from PIL import Image
from .common import read,write,sha,now
from .reference_generation import inputs
from .renderer import box


def audit_references(run):
    plan_path,plan,fonts=inputs(run);root=run/'font_references_train_v1'
    protocol=read(root/'protocol.json');digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if sha(plan_path)!=protocol['plan_sha256']:raise ValueError('Reference audit plan mismatch')
    receipts=sorted(root.glob('shard_*.receipt.json'))
    seen={};files=collections.Counter();attempts=collections.Counter();cursor=0;sampled=0;archive_bytes=0
    expected=plan['records'];amendment=None
    if (run/'reference_recovery/amendment.json').exists():
        amendment=read(run/'reference_recovery/amendment.json')
        if not amendment['passed'] or amendment['original_protocol_sha256']!=sha(root/'protocol.json'):
            raise ValueError('Reference audit amendment mismatch')
    for rp in receipts:
        receipt=read(rp);archive=rp.with_name(rp.name.replace('.receipt.json','.zip'))
        if receipt['protocol_hash']!=digest or receipt['first']!=cursor or sha(archive)!=receipt['sha256']:
            raise ValueError('Reference archive integrity/continuity mismatch')
        archive_bytes+=archive.stat().st_size
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None or len(z.namelist())!=4*len(receipt['records']):raise ValueError('Reference payload CRC/count mismatch')
            for i,row in enumerate(receipt['records']):
                specification=expected[cursor];cursor+=1
                label=json.loads(z.read(row['stem']+'.json'));g=label['glyphs']
                if (label['reference']!=specification or row['specimen_id']!=specification['specimen_id']
                    or label['pixel_sha256']!=row['pixel_sha256'] or not label['valid'] or label['count']!=6 or len(g)!=6):
                    raise ValueError('Reference specimen/label identity mismatch')
                recipe=label['recipe'];attempt=recipe['attempt'];attempts[str(attempt)]+=1
                if recipe['effects'] or recipe['split']!='train' or ''.join(x['char'] for x in g)!=specification['text']:
                    raise ValueError('Reference text/effect/split mismatch')
                if attempt>=6 and (amendment is None or attempt>=6+len(amendment['extra_settings'])):
                    raise ValueError('Reference fallback without verified amendment')
                for x in g:
                    if (x['font_sha256']!=specification['font_sha256'] or x['font_key']!=specification['font_key']
                        or x['axes']!=specification['axes'] or x['clipped_fraction']>.05 or x['touches_canvas_edge']
                        or x['visible_pixels']<10 or x['visible_fraction']<.25 or x['local_contrast']<12):
                        raise ValueError('Reference glyph provenance/acceptance mismatch')
                files[specification['font_key']]+=1
                seen.setdefault(row['pixel_sha256'],set()).add(specification['family_group'])
                # A deterministic sample checks decoded pixels and serialized
                # instance geometry, including every fallback specimen.
                if i%32==0 or attempt>=6:
                    image=np.asarray(Image.open(io.BytesIO(z.read(row['stem']+'.png'))).convert('RGB'))
                    if hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()!=row['pixel_sha256']:
                        raise ValueError('Reference decoded pixel hash mismatch')
                    with np.load(io.BytesIO(z.read(row['stem']+'.masks.npz')),allow_pickle=False) as masks:
                        if any(masks[k].shape!=(6,*image.shape[:2]) or masks[k].dtype!=np.uint8 for k in ('geometry','visible','decoration')):
                            raise ValueError('Reference serialized mask shape/type mismatch')
                        for k,x in enumerate(g):
                            if x['box_core']!=box(masks['geometry'][k].astype(np.float32)/255) or x['box_visible']!=box(masks['visible'][k].astype(np.float32)/255):
                                raise ValueError('Reference serialized mask/box mismatch')
                    sampled+=1
    complete=cursor==len(expected)
    if complete:
        done=read(root/'completion.json')
        if done['state']!='complete' or done['accepted_references']!=cursor or done['protocol_hash']!=digest:
            raise ValueError('Reference completion receipt mismatch')
        planned=collections.Counter(x['font_key'] for x in expected)
        if files!=planned or set(files)!=set(fonts):raise ValueError('Incomplete font coverage')
    stamp=str(time.time_ns())
    ambiguity_path=run/'audits'/('font_references_'+stamp+'_ambiguities.json')
    write(ambiguity_path,dict(pixel_hash_groups=[dict(pixel_sha256=k,family_groups=sorted(v)) for k,v in seen.items() if len(v)>1],
                             meaning='Identical clean reference images can match multiple nonidentical full-font families; retain alternatives rather than claiming unique identity. This does not merge entire families or change corpus splits.'))
    result=dict(created_at=now(),passed=True,full_plan_completed=complete,accepted_references=cursor,
                planned=len(expected),archives=len(receipts),archive_bytes=archive_bytes,font_files=len(files),
                alias_families=len({fonts[k]['split_group'] for k in files}),attempt_counts=dict(attempts),
                receipt_unique_pixel_hashes=len(seen),duplicate_references=cursor-len(seen),
                pixel_hash_groups_spanning_distinct_alias_families=sum(len(v)>1 for v in seen.values()),
                decoded_pixel_and_mask_samples=sampled,protocol_hash=digest,source_sha256=sha(__file__),
                ambiguity_report=dict(path=str(ambiguity_path),sha256=sha(ambiguity_path)),
                limitation='All archive hashes/CRCs and all reference/font/axis/text/acceptance labels checked; decoded pixels/masks sampled deterministically. Duplicate counts use renderer-provided pixel hashes; references are not a unique training corpus or trained embedding index.')
    destination=run/'audits'/('font_references_'+stamp+'.json');write(destination,result)
    write(run/'font_reference_audit_latest.json',dict(path=str(destination),sha256=sha(destination),result=result))
    return result
