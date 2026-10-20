"""Diagnose reference layout exhaustion without changing glyphs or gates."""
from pathlib import Path
from PIL import Image
from .common import read,write,now,sha
from .reference_generation import inputs
from .reference_render import recipe_for
from .renderer import render

EXTRA_SETTINGS=((512,.4,.18),(512,.3,.18),(512,.2,.18))


def initialize_extended(fonts):
    """Process-local initializer: original attempts precede bounded additions."""
    from . import reference_render as rr
    rr.initialize(fonts)
    if len(rr.SETTINGS)!=6:raise ValueError('Unexpected reference retry settings')
    rr.SETTINGS=rr.SETTINGS+EXTRA_SETTINGS


def check_recovery(run):
    import json
    import zipfile
    from . import reference_render as rr,reference_generation as generation
    _,plan,fonts=inputs(run)
    protocol=read(run/'font_references_train_v1/protocol.json')
    if sha(generation.__file__)!=protocol['source_sha256']:raise ValueError('Original reference writer changed')
    for path,digest in protocol['pilot_sources'].items():
        if sha(path)!=digest:raise ValueError('Original reference source changed')
    diagnosis=read(run/'reference_recovery/diagnosis.json')
    failed=next(r for r in plan['records'] if r['specimen_id']==diagnosis['specimen']['specimen_id'])
    original=rr.SETTINGS;rr.initialize(fonts)
    assert len(original)==6 and not rr.one(failed)['accepted']
    pilot=read(run/'font_reference_pilot.json');checked=0
    try:
        rr.SETTINGS=original+EXTRA_SETTINGS
        with zipfile.ZipFile(Path(pilot['path'])/'references.zip') as z:
            for row in read(Path(pilot['path'])/'receipt.json')['records']:
                stored=json.loads(z.read(row['stem']+'.json'));result=rr.one(stored['reference'])
                assert result['accepted']
                assert all(content==z.read(row['stem']+suffix) for suffix,content in result['payload'].items())
                checked+=1
        fixed=rr.one(failed);assert fixed['accepted'] and fixed['attempt']==6
        label=json.loads(fixed['payload']['.json'])
        assert label['count']==6 and label['valid'] and label['reference']==failed
        assert label['recipe']['text']['size_fraction']==.4
        destination=run/'reference_recovery'
        for suffix,content in fixed['payload'].items():(destination/('recovered'+suffix)).write_bytes(content)
    finally:rr.SETTINGS=original
    result=dict(created_at=now(),passed=True,original_generation_sha256=sha(generation.__file__),
                original_reference_render_sha256=sha(rr.__file__),recovery_source_sha256=sha(__file__),
                original_protocol_sha256=sha(run/'font_references_train_v1/protocol.json'),
                original_settings=list(original),extra_settings=list(EXTRA_SETTINGS),
                identical_pilot_payloads=checked,recovered_specimen=failed['specimen_id'],accepted_attempt=6,
                scope='Bounded smaller-size fallback after the original six attempts; unchanged font/text/axes, renderer/gates and completed archives')
    write(run/'reference_recovery/amendment.json',result);return result


def resume(run,workers=4):
    from . import reference_generation as generation,reference_render as rr
    amendment=read(run/'reference_recovery/amendment.json')
    protocol_path=run/'font_references_train_v1/protocol.json';protocol=read(protocol_path)
    if (not amendment['passed'] or amendment['recovery_source_sha256']!=sha(__file__)
        or amendment['original_generation_sha256']!=sha(generation.__file__)
        or amendment['original_reference_render_sha256']!=sha(rr.__file__)
        or amendment['original_protocol_sha256']!=sha(protocol_path)):
        raise ValueError('Unverified reference retry amendment')
    plan_path,plan,fonts=inputs(run)
    if sha(plan_path)!=protocol['plan_sha256']:raise ValueError('Reference plan changed')
    for path,digest in protocol['pilot_sources'].items():
        if sha(path)!=digest:raise ValueError('Original reference source changed')
    previous=generation.initialize
    generation.initialize=initialize_extended
    try:return generation.generate_set(run/'font_references_train_v1',plan['records'],fonts,protocol,workers)
    finally:generation.initialize=previous


def diagnose(run):
    _,plan,fonts=inputs(run)
    failures=sorted((run/'font_references_train_v1').glob('failure_*.json'))
    if not failures:raise ValueError('No reference failure to diagnose')
    failure=read(failures[-1]);record=next(r for r in plan['records'] if r['specimen_id']==failure['specimen_id'])
    attempts=[];destination=run/'reference_recovery';destination.mkdir(exist_ok=True)
    for index,size in enumerate([.65,.4,.3,.2,.15,.1]):
        recipe=recipe_for(record,0);recipe['text']['size_fraction']=size
        out=render(recipe,fonts)
        issues=[]
        for g in out['glyphs']:
            if g['clipped_fraction']>.05 or g['touches_canvas_edge'] or g['visible_pixels']<10 or g['box_core'][3]-g['box_core'][1]<7 or g['local_contrast']<12:
                issues.append({k:g[k] for k in ('index','char','clipped_fraction','touches_canvas_edge','visible_pixels','box_core','size_px','local_contrast')})
        attempts.append(dict(size_fraction=size,valid=out['valid'],quality=out['quality'],issues=issues))
        Image.fromarray(out['image']).save(destination/f'diagnosis_{index}.png')
    result=dict(created_at=now(),specimen=record,attempts=attempts,source_sha256=sha(__file__))
    write(destination/'diagnosis.json',result);return result
