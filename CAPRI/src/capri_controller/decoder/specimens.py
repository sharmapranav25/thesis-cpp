"""Deterministic TRAIN-only library specimens, separate from the frozen corpus.

This produces a plan, not images or trained embeddings. Held-out gallery builds
must have their own declared evaluation protocol; this API cannot plan them.
"""
import hashlib
import json
from pathlib import Path
import string
from .common import read,sha,write


def plan_specimens(fonts):
    # Every file sees the same interleaved digit/upper/lower sequence, restricted
    # only by recorded coverage. Never seed textual content with family identity.
    alphabet=''.join((string.digits[i] if i<10 else '')+string.ascii_uppercase[i]+string.ascii_lowercase[i] for i in range(26))
    records=[]
    for font in sorted(fonts,key=lambda f:f['key']):
        if font['split']!='train': raise ValueError('Reference planning is TRAIN-only')
        supported=''.join(c for c in alphabet if c in font['chars'])
        if len(supported)<2: raise ValueError('Insufficient library glyph coverage')
        defaults={a['tag']:a['default'] for a in font['axes']}
        variants=[('default',defaults)]
        # One axis at a time; avoid an exponential product and retain actual
        # axis coordinates instead of relabelling all variants as the default.
        for axis in font['axes']:
            if axis['tag'] not in ('wght','wdth','ital','slnt'):continue
            for endpoint in ('min','max'):
                if axis[endpoint]!=axis['default']:
                    variants.append((axis['tag']+'_'+endpoint,dict(defaults,**{axis['tag']:axis[endpoint]})))
        for variant,axes in variants:
            for page in range(12):
                text=''.join(supported[(page*6+j)%len(supported)] for j in range(6))
                records.append(dict(specimen_id=font['key']+'/'+variant+'/'+str(page),font_key=font['key'],
                                    font_sha256=font['sha256'],family_group=font['split_group'],split='train',
                                    style=font['style'] or 'unspecified',axes=axes,text=text))
    return records


def prepare_train_references(run):
    path=Path(run)/'fonts/font_manifest.json';protocol=read(Path(run)/'corpus_protocol.json')
    digest=sha(path)
    if digest!=protocol['font_manifest_sha256']:raise ValueError('Frozen font manifest mismatch')
    fonts=[f for f in read(path)['fonts'] if f['split']=='train']
    for f in fonts:
        if sha(f['path'])!=f['sha256']:raise ValueError('Font file differs from manifest: '+f['key'])
    plan=dict(schema='capri-train-font-specimens-v1',font_manifest_sha256=digest,
              source_sha256=sha(__file__),fonts=len(fonts),records=plan_specimens(fonts),
              purpose='TRAIN reference planning only; not generated images or an embedding index',
              policy='12 six-character strings per default/one-axis endpoint variant; fixed coverage-based content; clean rendering and predicted-region extraction must be validated before gallery publication')
    plan_hash=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    destination=Path(run)/'font_reference_plans'/('train_'+plan_hash+'.json')
    if destination.exists():
        if read(destination)!=plan:raise ValueError('Existing reference plan differs')
    else:write(destination,plan)
    return dict(path=str(destination),sha256=sha(destination),fonts=len(fonts),specimens=len(plan['records']),
                planned_glyphs=6*len(plan['records']),generated_images=0,trained_embedding_index=False)
