"""Balanced, resumable CAPRI corpus. Each shard preserves one controller entry."""
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from .pipeline import ROOT,U,R,ARMS,draw_recipe,usable,key
import argparse,collections,hashlib,io,json,time,zipfile,copy
from concurrent.futures import ProcessPoolExecutor,as_completed
from PIL import Image
import numpy as np

OUT=ROOT/'corpus'
SEED_BASE=int(os.environ.get('CAPRI_CORPUS_SEED_BASE','2000000000'))

def init():
    R.install()

def generate(task):
    entry,count,position,length,entries_sha=task
    name=entry['entry_id'];dest=OUT/(name+'.zip');summary=OUT/(name+'.json')
    if dest.exists() and summary.exists():
        s=U.read(summary)
        if s['count']==count and s['entries_sha256']==entries_sha and s.get('seed_base',2000000000)==SEED_BASE and U.sha(dest)==s['zip_sha256']:return s
        raise RuntimeError('Existing shard does not match run: '+name)
    started=time.time();labels=[];rejections=[];hashes=set();tmp=OUT/(name+'.partial')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED) as z:
        for j in range(count):
            for attempt in range(100):
                # Disjoint seed intervals per entry and output; retries cannot collide.
                seed=SEED_BASE+position*10000000+j*10000+attempt
                row=R.safe_render((draw_recipe(entry['recipe'],seed),entry['controls']))
                reason=None
                if not usable(row,length):reason=row.get('error','renderer completeness/usability/length guard')
                if reason is None:
                    digest=hashlib.sha256(row['image'].tobytes()).hexdigest()
                    if digest in hashes:reason='duplicate RGB pixels within entry'
                if reason is not None:
                    rejections.append(dict(sample=j,attempt=attempt,seed=seed,reason=reason));continue
                hashes.add(digest);break
            else:raise RuntimeError('100 retries exhausted for '+name+' sample '+str(j))
            filename=f'{j:05d}.png';buf=io.BytesIO();Image.fromarray(row['image']).save(buf,format='PNG')
            z.writestr(filename,buf.getvalue())
            labels.append(dict(image=filename,text=row['text'],seed=seed,entry_id=name,controller=entry['controller'],input_index=entry['input_index'],
                pixel_sha256=digest,complete=bool(row['complete']),usable=bool(row['usable']),strict=bool(row['strict']),attempts=attempt+1,
                size=list(Image.fromarray(row['image']).size),text_spec=row['text_spec'],effects_spec=row['effects_spec']))
        z.writestr('labels.jsonl','\n'.join(json.dumps(r,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()) for r in labels)+'\n')
        z.writestr('rejections.jsonl','\n'.join(json.dumps(r) for r in rejections)+ ('\n' if rejections else ''))
    tmp.replace(dest)
    s=dict(entry_id=name,controller=entry['controller'],count=count,rejections=len(rejections),strict=sum(r['strict'] for r in labels),
        seconds=time.time()-started,bytes=dest.stat().st_size,zip_sha256=U.sha(dest),entries_sha256=entries_sha,seed_base=SEED_BASE)
    U.dump(summary,s);return s

def plan():
    entries=U.read(ROOT/'entries.json');assert len(entries)==300
    rng=np.random.default_rng(1052026);extra=set()
    for arm,n in zip(ARMS,[34,33,33]):
        ids=[e['entry_id'] for e in entries if e['controller']==arm];assert len(ids)==100
        extra.update(rng.choice(ids,n,replace=False).tolist())
    return entries,{e['entry_id']:333+int(e['entry_id'] in extra) for e in entries}

def audit():
    entries,counts=plan();length=U.read(ROOT/'length_decision.json')['default_length'];target={r['pixel_sha256'] for r in U.read(ROOT/'inputs.json')}
    entries_sha=U.sha(ROOT/'entries.json')
    seen=set();seeds=set();controllers=collections.Counter();lengths=collections.Counter();strict=0;rejections=0
    for e in entries:
        path=OUT/(e['entry_id']+'.zip');s=U.read(path.with_suffix('.json'));assert U.sha(path)==s['zip_sha256'] and s['entries_sha256']==entries_sha
        with zipfile.ZipFile(path) as z:
            rows=[json.loads(l) for l in z.read('labels.jsonl').decode().splitlines()];assert len(rows)==counts[e['entry_id']]
            assert len([n for n in z.namelist() if n.endswith('.png')])==len(rows)
            for r in rows:
                im=np.asarray(Image.open(io.BytesIO(z.read(r['image']))).convert('RGB'));h=hashlib.sha256(im.tobytes()).hexdigest()
                assert h==r['pixel_sha256'] and h not in seen and h not in target
                assert r['seed'] not in seeds and r['entry_id']==e['entry_id'] and r['controller']==e['controller']
                assert r['complete'] and r['usable'] and len(r['text'])==length
                assert r['text_spec']['text']==r['text'] and r['size']==[im.shape[1],im.shape[0]]
                seen.add(h);seeds.add(r['seed']);controllers[r['controller']]+=1;lengths[len(r['text'])]+=1;strict+=int(r['strict'])
            rejections+=s['rejections']
    assert len(seen)==100000
    result=dict(images=len(seen),unique_pixel_hashes=len(seen),unique_seeds=len(seeds),target_pixel_overlap=0,controllers=dict(controllers),lengths=dict(lengths),
        strict=strict,rejections=rejections,entries=300,shards=300,all_pngs_decoded=True,all_lengths_model_derived=True,
        limitation='Renderer usability is not human legibility or solver accuracy; target fidelity and transfer are not established.')
    U.dump(ROOT/'corpus_audit.json',result);print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.audit:audit();return
    OUT.mkdir(exist_ok=True);entries,counts=plan();length=U.read(ROOT/'length_decision.json')['default_length'];sha=U.sha(ROOT/'entries.json')
    U.dump(ROOT/'allocation.json',counts);tasks=[(e,counts[e['entry_id']],i,length,sha) for i,e in enumerate(entries)]
    if a.pilot:tasks=[t for t in tasks if t[0]['input_index'] in [0,24,49,74,99]]
    started=time.time();summaries=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init) as pool:
        futures=[pool.submit(generate,t) for t in tasks]
        for future in as_completed(futures):
            summaries.append(future.result());done=sum(s['count'] for s in summaries)
            U.dump(ROOT/'corpus_status.json',dict(completed_images=done,completed_shards=len(summaries),requested_shards=len(tasks),seconds=time.time()-started))
            print(f'{len(summaries)}/{len(tasks)} shards, {done} images, {time.time()-started:.1f}s',flush=True)
    U.dump(ROOT/('corpus_pilot_summary.json' if a.pilot else 'corpus_summary.json'),summaries)
    if not a.pilot:audit()

if __name__=='__main__':main()
