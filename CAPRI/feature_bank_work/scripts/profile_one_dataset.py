"""Run one authorized dataset with frozen models and synthetic thresholds."""
import argparse,collections,json,shutil,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from profile_five_datasets import ROOT,PROJECT,FEATURES,initialize,analyze,generate
from feature_bank.core import read,write,new_run,load,digest,pixel_hash
from feature_bank.prepare import resolve_image
from feature_bank.views import image_views

def main():
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['city','courier','darkbay','outlined_text_captcha']);args=parser.parse_args()
    name=args.dataset;start=time.perf_counter();out=new_run(ROOT/f'runs/feature_bank_{name}_reference_01')
    previous=ROOT/'runs/feature_bank_full_reference_01';cuts=read(previous/'thresholds.json')
    for f in FEATURES:
        assert digest(ROOT/'runs/feature_bank_exploratory_01'/f'{f}_LR.joblib')==cuts[f]['model_sha256']
    shutil.copy2(previous/'thresholds.json',out/'thresholds.json')
    path=PROJECT/'runs'/('external_pixels_outlined_01' if name=='outlined_text_captcha' else 'pixel_statistics_01')/name/'split.json'
    split=read(path);members=split['fit']+split['validation'];expected={r['pixel_sha256']:r for r in members}
    assert len(expected)==len(members)
    cache={};ignored=0
    if (previous/(name+'.jsonl')).exists():
        for line in (previous/(name+'.jsonl')).read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:ignored+=1;continue
            if r['pixel_sha256'] in expected:cache[r['pixel_sha256']]=r
    for ph,r in cache.items():
        source=resolve_image(PROJECT,name,expected[ph])
        assert digest(source)==expected[ph]['file_sha256'] and pixel_hash(load(source))==ph
    items=[(name,resolve_image(PROJECT,name,r),r['pixel_sha256'],r['file_sha256']) for r in members if r['pixel_sha256'] not in cache]
    write(out/'protocol.json',dict(dataset=name,images=len(members),reused=len(cache),ignored_incomplete_cache_lines=ignored,
        source_manifest=str(path),source_sha256=digest(path),exclusions=split['excluded_directories'],
        purpose='Whole-reference predicted distributions and limited generation, not accuracy evaluation',
        real_labels_accessed=False,other_datasets_started=False,thresholds_retuned=False,
        aggregation='any positive brick => positive image; all absent => absent; otherwise uncertain',
        reference_data_not_untouched=True,workers=4))
    for rel in ['scripts/profile_one_dataset.py','scripts/profile_five_datasets.py','feature_bank/descriptors.py','feature_bank/views.py','feature_bank/regions.py']:
        p=out/'code_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
    rows=list(cache.values());print(f'{name}: {len(rows)} cached, {len(items)} remaining',flush=True)
    with (out/(name+'.jsonl')).open('w') as stream:
        for r in rows:stream.write(json.dumps(r)+'\n')
        with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(cuts,)) as pool:
            for result in pool.map(analyze,items,chunksize=4):
                rows.append(result);stream.write(json.dumps(result)+'\n')
                if len(rows)%100==0:
                    stream.flush();write(out/'progress.json',dict(images=len(rows),total=len(members),elapsed_seconds=time.perf_counter()-start))
                    print(f'{name}: {len(rows)}/{len(members)}',flush=True)
    assert len(rows)==len(expected) and {r['pixel_sha256'] for r in rows}==set(expected)
    summary=dict(images=len(rows),features={},combinations={})
    for f in FEATURES:
        summary['features'][f]={s:sum(r['predictions'][f]['image_state']==s for r in rows) for s in ('present','absent','uncertain')}
        summary['features'][f]['positive_brick_fraction']=float(np.mean([s=='present' for r in rows for s in r['predictions'][f]['states']]))
    for r in rows:
        key='+'.join(f for f in FEATURES if r['predictions'][f]['image_state']=='present') or 'no_confident_positive'
        summary['combinations'][key]=summary['combinations'].get(key,0)+1
    write(out/'summaries.json',{name:summary})
    canvas=Image.new('RGB',(1200,1080),'#eef1f5');draw=ImageDraw.Draw(canvas)
    for j,title in enumerate(['Original','Feature pop','Line regions','Curve regions','Dot regions']):draw.text((j*240+8,8),title,fill='black')
    for i,r in enumerate(sorted(rows,key=lambda r:r['pixel_sha256'])[:8]):
        im=load(r['path']);views=[im,Image.fromarray(image_views(im)['feature_pop']).convert('RGB')]
        for f in FEATURES:
            overlay=im.copy();d=ImageDraw.Draw(overlay)
            for region,state in zip(r['regions'],r['predictions'][f]['states']):
                if state=='present':d.rectangle(region['core_box'],outline=(255,65,0),width=1)
            views.append(overlay)
        for j,view in enumerate(views):
            view.thumbnail((228,98));x=j*240+6;y=34+i*128;canvas.paste(view,(x,y));draw.text((x,y+100),r['pixel_sha256'][:10],fill='black')
    canvas.save(out/'predictions.png');generate(out,{name:summary})
    write(out/'status.json',dict(state='DESCRIPTIVE_COMPLETE',dataset=name,images=len(rows),bricks=len(rows)*30,
        accuracy_measured=False,style_recreation_verified=False,generated_attempts=32,
        generation_failures=len(read(out/'generation_failures.json')),elapsed_seconds=time.perf_counter()-start))
    lines=[f'# {name.title()} whole-reference analysis','',f'**OBSERVED:** Processed all {len(rows)} registered unique references ({len(rows)*30} overlapping bricks). Existing exclusions were preserved. Only this dataset was run. Frozen models and synthetic calibration thresholds were reused without target-label tuning.','',
        f'[Prediction gallery](../../runs/{out.name}/predictions.png) · [Generated examples](../../runs/{out.name}/{name}_generated.png) · [Summary](../../runs/{out.name}/summaries.json)','',
        '| Feature | Present images | Absent | Uncertain | Positive brick fraction |','|---|---:|---:|---:|---:|']
    for f in FEATURES:
        s=summary['features'][f];lines.append(f"| {f} | {s['present']} | {s['absent']} | {s['uncertain']} | {s['positive_brick_fraction']:.2%} |")
    lines+=['','These are predictions, not audited feature prevalence or accuracy. Any positive brick makes an image positive; this image aggregation is not calibrated. Overlapping brick counts are not area/density estimates.',
        '', 'Generated 32 attempts using the existing Python text renderer, audited library fonts and Pillow effect adapters. Only positive feature combinations, native dimensions and median background RGB guide the generation. Font style, texture, deformation, density and tilt remain generic or unsupported. PASS labels mean text visibility only, not visual matching. This batch does not use PHP renderers.',
        '',f"Text-visibility/rendering failures: {len(read(out/'generation_failures.json'))}. Failed attempts remain recorded. Reference data now guides generation and is not untouched evaluation data. Earlier experiments remain intact. Artifacts are in the local staging workspace."]
    (ROOT/f'docs/feature_bank/{name.upper()}_REFERENCE_01.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2));print('COMPLETE',flush=True)

if __name__=='__main__':main()
