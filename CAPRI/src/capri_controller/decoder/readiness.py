"""Freeze the 500k corpus only after broad font, pair, mask and storage checks."""
import collections
import hashlib
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read, write, sha, now
from .pilot import data_source_hashes
from .renderer import box


def prepare(run,count=4096):
    if (run/'corpus_protocol.json').exists():
        raise FileExistsError('Preserve the frozen corpus protocol; inspect existing readiness rather than replacing it')
    pilot=run/f'pilot_{count:05d}'
    report=read(pilot/'report.json');manifest=read(run/'fonts/font_manifest.json')
    checks=read(run/'renderer_checks.json');storage=read(run/'storage_checks.json')
    source=data_source_hashes()
    assert count>=3000 and report['requested']==report['accepted']==report['unique']==count
    assert not report['duplicates'] and not report['failed'] and checks['passed'] and storage['passed']
    assert report['source_hashes']==storage['source_hashes']==source
    assert report['font_manifest_sha256']==sha(run/'fonts/font_manifest.json')
    # This is a saved human/agent visual review of synthetic examples, not a VLM call.
    visual=read(run/'pilot_visual_review.json')
    assert visual['passed'] and visual['pilot']==str(pilot.name) and visual['report_sha256']==sha(pilot/'report.json')
    fontmap={f['key']:f for f in manifest['fonts']}
    for font in fontmap.values():
        if sha(font['path'])!=font['sha256']:raise ValueError('Font asset changed: '+font['path'])
    labels={};families=set();splitgroups={};hashes=set()
    for font in fontmap.values():
        splitgroups.setdefault(font['split_group'],set()).add(font['split'])
    assert all(len(v)==1 for v in splitgroups.values())
    for path in sorted(pilot.glob('*.json')):
        if path.name=='report.json':continue
        label=read(path);recipe=label['recipe'];n=label['count']
        im=np.asarray(Image.open(path.with_suffix('.png')))
        digest=hashlib.sha256(im.tobytes()).hexdigest()
        assert digest==label['pixel_sha256'] and digest not in hashes
        hashes.add(digest)
        masks=np.load(path.with_suffix('.masks.npz'))
        assert len(label['glyphs'])==recipe['requested_count']==n
        assert all(masks[k].shape==(n,*im.shape[:2]) for k in ('geometry','visible','decoration'))
        for i,g in enumerate(label['glyphs']):
            font=fontmap[g['font_key']]
            assert font['split']==recipe['split'] and g['char'] in font['chars']
            assert g['box_core']==box(masks['geometry'][i].astype(np.float32)/255)
            assert g['box_visible']==box(masks['visible'][i].astype(np.float32)/255)
            families.add(font['family_group'])
        masks.close()
        labels[(recipe['split'],recipe['index'])]=label
    assert families=={f['family_group'] for f in fontmap.values()},'Pilot did not cover all font families'
    pairs=collections.Counter()
    for (split,index),label in labels.items():
        if index%2:continue
        a=label['recipe'];b=labels[(split,index+1)]['recipe']
        assert a['attempt']==b['attempt']
        kind=a['pair_kind'];pairs[str(kind)]+=1
        if kind=='clean_degraded':
            assert a['text']==b['text'] and a['size']==b['size'] and a['render_seed']==b['render_seed']
            assert not a['effects'] and b['effects']
        elif kind=='same_fonts_different_content':
            assert a['text']['font_keys']==b['text']['font_keys'] and a['text']['text']!=b['text']['text']
        elif kind=='same_content_different_fonts':
            assert a['text']['text']==b['text']['text']
            assert fontmap[a['text']['font_keys'][0]]['split_group']!=fontmap[b['text']['font_keys'][0]]['split_group']
        elif kind is not None:raise ValueError('Unfulfilled pair relationship')
    assert set(report['distributions']['length'])==set(map(str,range(2,9)))
    assert len(report['distributions']['effect_presence'])==37
    assert min(report['distributions']['effect_presence'].values())>=25
    free=shutil.disk_usage(run).free
    estimate=int(max(report['bytes']/count,storage['measured_bytes_per_image'])*500000*1.3)
    reserve=30*2**30
    if free<estimate+reserve:raise OSError('Conservative corpus budget plus checkpoint reserve does not fit')
    protocol=dict(schema='capri-decoder-corpus-v1',created_at=now(),total_accepted_unique=500000,
                  split_sizes={'train':450000,'dev':20000,'calibration':10000,'test':20000},
                  split_boundary='Canonical font family plus exact ASCII-outline aliases; sibling pairs stay within split. No real target labels.',
                  font_manifest_sha256=sha(run/'fonts/font_manifest.json'),source_hashes=source,
                  writer_sha256=sha(Path(__file__).with_name('production.py')),shard_images=1000,reserve_bytes=reserve,
                  sampling='Balanced count 2..8 and cyclic font-family/count anchors; stochastic supported face, axes, colours, layout, effects and intensity. Pair-level retries keep font/count anchor. Measured accepted distributions retained per shard.',
                  pair_ratios={'clean_degraded':.1,'same_fonts_different_content':.1,'same_content_different_fonts':.1,'independent':.7},
                  quality_gates={'min_visible_fraction':.25,'max_initial_clipped_fraction':.05,'min_visible_pixels':10,
                                 'min_core_height':7,'min_local_contrast':12,'allow_final_canvas_edge':False},
                  labels='Per-instance core/visible/decorative coverage uint8, matching boxes, count, realized layout/axes/colour, effect presence and realized strength, full recipe/provenance and underlying background RGB.',
                  evaluation='Dev selects synthetic checkpoints; calibration fits uncertainty/thresholds after selection; test labels reserved for final evaluation. No training/affinity on held-out splits.',
                  limitations=['Stochastic effect/level acceptance is audited rather than exact joint balancing. Family and count anchors survive rejection.',
                               'Exact outline aliases grouped; nonidentical near-duplicates may remain.',
                               'Soft coverage after photometric/JPEG effects is annotation convention, not unique physical alpha.',
                               'Font filename matches do not prove byte identity with historical unavailable files.'])
    snapshot=run/'frozen_data_sources';snapshot.mkdir(exist_ok=True)
    for number,(path,digest) in enumerate(source.items()):
        shutil.copyfile(path,snapshot/f'{number:02d}_{Path(path).name}')
    shutil.copyfile(run/'fonts/font_manifest.json',snapshot/'font_manifest.json')
    write(run/'corpus_protocol.json',protocol)
    result=dict(created_at=now(),passed=True,protocol_sha256=sha(run/'corpus_protocol.json'),pilot_count=count,
                family_coverage=len(families),pairs=dict(pairs),all_exported_masks_checked=True,all_font_hashes_checked=True,
                estimate_GiB=estimate/2**30,free_GiB=free/2**30,checkpoint_reserve_GiB=30,
                font_manifest_sha256=protocol['font_manifest_sha256'],source_hashes=source)
    write(run/'production_readiness.json',result)
    return {k:v for k,v in result.items() if k!='source_hashes'}
