"""Run the unchanged sample inference/render mapping over every target image."""
import json,time,hashlib,html
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE=Path(__file__).parent
OUT=Path('C:/Users/pranav/Documents/ChatGPT/thesis/iter2_full_target_recreation_01')

def build_worker():
    source=(HERE/'iter2_target_recreate.py').read_text()
    prefix=source[:source.index('families=')]
    prefix=prefix.replace("iter2_target_recreation_01",'iter2_full_target_recreation_01').replace('out.mkdir(exist_ok=False)','out.mkdir(exist_ok=True)')
    begin=source.index('        path=paths[int(index)]')
    end=source.index('        rows.append(row)',begin)
    body=source[begin:end]
    body='\n'.join(line[4:] if line.startswith('    ') else line for line in body.splitlines())
    body=body.replace('path=paths[int(index)]','path=Path(source_path)').replace('1700+len(rows)','1700+global_index')
    worker=prefix+'''\ndef run_item(item):
    family,j,global_index,source_path=item
    try:
        return render_item(family,j,global_index,source_path)
    except Exception as exc:
        return dict(family=family,index=j,source=source_path,error=repr(exc))

def render_item(family,j,global_index,source_path):
'''+body+'\n    row["index"]=j\n    return row\n'
    (HERE/'iter2_full_target_worker.py').write_text(worker)

def main():
    OUT.mkdir(exist_ok=False)
    build_worker()
    from iter2_full_target_worker import run_item,checkpoint
    directories={
      'CryptBB':[Path('E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023')],
      'City':[Path('E:/pranav/City (live)-20260503T045450Z-3-001/City (live)')],
      'DarkBay':[Path('E:/pranav/cryptbb_sourcefree/data/darkbay/pool'),Path('E:/pranav/cryptbb_sourcefree/data/darkbay/eval')]}
    items=[];counts={}
    for family,dirs in directories.items():
        paths=[p for d in dirs for p in sorted(d.iterdir()) if p.suffix.lower() in ('.png','.jpg','.jpeg')]
        counts[family]=len(paths)
        for j,path in enumerate(paths):items.append((family,j,len(items),str(path)))
    (OUT/'manifest.json').write_text(json.dumps(dict(epoch=15,counts=counts,total=len(items),sources={k:[str(d) for d in v] for k,v in directories.items()},real_answers_used=False),indent=2))
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=12) as pool,(OUT/'predictions.jsonl').open('w') as stream:
        for row in pool.map(run_item,items,chunksize=8):
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if len(rows)%100==0 or len(rows)==len(items):
                status=dict(completed=len(rows),total=len(items),errors=sum('error' in r for r in rows),seconds=round(time.perf_counter()-start,1))
                (OUT/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    summary={}
    for family in directories:
        selected=[r for r in rows if r['family']==family];valid=[r for r in selected if 'error' not in r]
        summary[family]=dict(images=len(selected),recreated=len(valid),errors=len(selected)-len(valid),
          predicted_lengths=dict(sorted(Counter(r['length'] for r in valid).items())),
          font_categories=dict(Counter(r['font']['category'] for r in valid)),
          effect_positive_counts=dict(Counter(h for r in valid for h,v in r['effects'].items() if v['present'])),
          render_usable=sum(r['render_usable'] for r in valid))
        sections=[]
        for r in selected:
            title=html.escape(Path(r['source']).name)
            if 'error' in r:sections.append(f'<article><h3>{title}</h3><p>{html.escape(r["error"])}</p></article>');continue
            filename=Path(r['comparison']).name
            sections.append(f'<article id="image-{r["index"]}"><h3>{r["index"]+1}. {title}</h3><img loading="lazy" src="{html.escape(filename)}" alt="Original above; inferred recreation below"><p>Predicted length {r["length"]}; font {html.escape(r["font"]["category"])}; fresh text {html.escape(r["rendered_text"])}</p></article>')
        page=f'<!doctype html><meta charset="utf-8"><title>{family}: all target recreations</title><style>body{{font:16px system-ui;background:#eceef2;margin:24px}}article{{background:white;padding:16px;margin:24px auto;max-width:1100px}}img{{max-width:100%;height:auto}}h1,p{{max-width:1100px;margin:16px auto}}</style><h1>{family}: all {len(selected)} target images</h1><p>Epoch 15 EMA. Original above; generated recreation below. Fresh text, own font bank, renderer default colours and positions. No answer labels or fitting to real images.</p>'+''.join(sections)
        (OUT/f'{family}_gallery.html').write_text(page,encoding='utf-8')
    report=dict(state='complete',total=len(items),epoch=15,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),seconds=round(time.perf_counter()-start,1),summary=summary,
       mapping='Same frozen per-image mapping as the six-example test; .5 presence threshold; predicted length and content; own font bank; renderer default colours/positions; no real answers, model fitting, or family overrides.',errors=[r for r in rows if 'error' in r])
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    (OUT/'status.json').write_text(json.dumps(dict(state='complete',completed=len(rows),total=len(items),seconds=report['seconds']),indent=2))
    index='<!doctype html><meta charset="utf-8"><title>Full target recreation</title><style>body{font:18px system-ui;margin:40px;max-width:900px}li{margin:20px}</style><h1>Full target inference and recreation</h1><p>Every target image, in source order. Original above, recreation below. Epoch 15 EMA; fresh strings from predicted lengths; colours and positions use defaults.</p><ul>'+''.join(f'<li><a href="{f}_gallery.html">{f}: {counts[f]} images</a></li>' for f in counts)+'</ul><p><a href="report.json">Summary report</a> | <a href="predictions.jsonl">Every prediction and recipe</a></p>'
    (OUT/'index.html').write_text(index,encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
