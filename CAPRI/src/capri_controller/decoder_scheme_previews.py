"""Repair preview metadata export, preserving all frozen predictions and pixels."""
import html
import json
from . import decoder_scheme_evaluation as E
from .decoder.common import ROOT,read,write,sha,now


def main():
    source=E.OUT
    freeze=read(source/'prediction_freeze.json');assert sha(source/'predictions.json')==freeze['sha256']
    sets=read(source/'image_manifests.json');records=read(source/'predictions.json')
    output=source/'preview_verified'
    if output.exists():raise FileExistsError('Preserving preview export')
    output.mkdir()
    E.OUT=output
    try:reports=E.render_panels(sets,records)
    finally:E.OUT=source
    assert all('error' not in v for rows in reports.values() for row in rows for v in row['variants'])
    files=list((output/'renders').glob('*.png'));assert len(files)==24
    # Only Python integer serialization changed. Pixel data and recipes must match
    # the artifacts produced before any real labels were opened.
    assert all(sha(p)==sha(source/'renders'/p.name) for p in files)
    recipes=list((output/'renders').glob('*.recipe.json'))
    assert all(read(p)==read(source/'renders'/p.name) for p in recipes)
    audit=dict(completed_at=now(),prediction_sha256=freeze['sha256'],rendered_images=24,
        identical_to_prelabel_pixels=True,identical_to_prelabel_recipes=True,
        valid_images=sum(v['valid'] for rows in reports.values() for row in rows for v in row['variants']),
        code_sha256=sha(__file__),adapter_sha256=sha(ROOT/'src/capri_controller/decoder_preview_render.py'),
        files={p.name:sha(p) for p in (output/'renders').glob('*')})
    write(output/'audit.json',audit)
    summary=read(source/'summary.json')
    page='<html><meta charset="utf-8"><title>CAPRI comparisons: 100 per dataset</title><style>body{font:17px system-ui;max-width:1260px;margin:24px auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>CAPRI: 100 inputs per dataset</h1><p>Left: real input. Right: fresh random text using predicted count, boxes, nearest font families, colour and effects. Two fixed seeds for the first four hash-ranked inputs per dataset. Unrefined procedural previews; no VLM, controller fitting or new training. Font/style resemblance is unverified.</p><p>CryptBB count: 62/100. All 100 saved annotations have five characters (posthoc constant-five baseline: 100/100). City and SimpleCaptcha have predictions here, without matched verified scoring annotations. No real font/localization accuracy claim.</p>'
    for name in records:page+='<h2>'+html.escape(name)+'</h2><img src="'+name+'_comparison.png">'
    page+='<p>Renderer acceptance checks passed for '+str(audit['valid_images'])+'/24 images. These checks concern clipping, visibility and contrast, not similarity to the target.</p><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre></html>'
    (output/'report.html').write_text(page,encoding='utf8')
    print(json.dumps({k:v for k,v in audit.items() if k!='files'},indent=2),flush=True)
