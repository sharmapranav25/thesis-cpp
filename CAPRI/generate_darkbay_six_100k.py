"""Finite generation and full audit of the separately requested six-char corpus."""
from pathlib import Path
import os,sys,json,csv,time,hashlib,subprocess,concurrent.futures,collections
from PIL import Image
WORK=Path(__file__).resolve().parent
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
PKG=PROJECT/'releases/CAPRI_typography_v3'
OUT=Path('E:/pranav/Final sem/synth_data/darkbay_capri_v3_6char_100k')
def write(p,obj):
 temp=p.with_name(p.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(obj,indent=2))
 for retry in range(100):
  try:temp.replace(p);return
  except PermissionError:
   if retry==99:raise
   time.sleep(.2)
def main():
 OUT.mkdir(exist_ok=False);(OUT/'shards').mkdir();(OUT/'logs').mkdir()
 source=(PKG/'generate.py').read_text();source=source.replace('ROOT=Path(__file__).resolve().parent','ROOT=Path('+repr(str(PKG))+')')
 needle="   if i<42:previews.append((image.copy(),text+' | '+('mixed' if mixed else 'single')))"
 assert needle in source
 source=source.replace(needle,needle+"\n   if (i+1)%500==0:\n    mf.flush();cf.flush();(a.out/'progress.json').write_text(json.dumps(dict(generated=i+1)));print('generated',i+1,flush=True)")
 worker=OUT/'worker_snapshot.py';worker.write_text(source);compile(source,str(worker),'exec')
 shards=[dict(id=i,count=9999 if i<9 else 10009,seed=202609285000+i) for i in range(10)]
 protocol=dict(family='darkbay',images=100000,length=6,length_basis='User-selected from interim recognizer results:59%correct-length,mean6.28,mode6; not automatic count-model success',evaluation_exposure='The200-image real evaluation summary informed this choice. Subsequent evaluation on it is development-exposed, not an untouched final comparison.',font_policy='35-font uniform shuffled decks; mixed fraction0.5; bounded per-glyph variation',refinement='off as defined by new Phase2 typography release',buckets=['numeric','alphabetic','alphanumeric'],partition='train',shards=shards,workers=6,package=str(PKG),worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),package_manifest_sha256=hashlib.sha256((PKG/'INTEGRITY.json').read_bytes()).hexdigest(),comparison_limit='Changes both length and typography versus old100k; not a length-only ablation',recognizer_training=False)
 write(OUT/'protocol.json',protocol)
 def status(state,**extra):
  rows=[]
  for sh in shards:
   d=OUT/'shards'/f"{sh['id']:02d}";p=d/'audit.json';q=d/'progress.json';r={}
   try:r=json.loads(p.read_text()) if p.exists() else (json.loads(q.read_text()) if q.exists() else {})
   except (OSError,ValueError):pass
   rows.append(dict(shard=sh['id'],generated=r.get('images',r.get('generated',0)),target=sh['count'],state=r.get('state','running_or_queued')))
  row=dict(state=state,pid=os.getpid(),time=time.time(),generated=sum(r['generated'] for r in rows),target=100000,shards=rows,**extra);write(OUT/'status.json',row);write(PROJECT/'monitor/darkbay_6char_100k_status.json',row)
 def run(sh):
  with (OUT/'logs'/f"{sh['id']:02d}.log").open('w') as f:
   args=[sys.executable,str(worker),'--base',str(PROJECT/'releases/CAPRI_v2'),'--family','darkbay','--count',str(sh['count']),'--seed',str(sh['seed']),'--partition','train','--lengths','6','--mixed-fraction','.5','--out',str(OUT/'shards'/f"{sh['id']:02d}")];result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f"Shard{sh['id']}failed; see its log")
 status('generating')
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  pending={pool.submit(run,s) for s in shards}
  while pending:
   done,pending=concurrent.futures.wait(pending,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:f.result()
   status('generating')
 status('auditing');seen=set();cells=collections.Counter();fonts=collections.Counter();modes=collections.Counter();total=0
 with (OUT/'labels.csv').open('w',newline='') as cf,(OUT/'manifest.jsonl').open('w') as mf:
  writer=csv.writer(cf);writer.writerow(['filename','label'])
  for sh in shards:
   d=OUT/'shards'/f"{sh['id']:02d}";num=0
   with (d/'manifest.jsonl').open() as stream:
    for line in stream:
     r=json.loads(line);path=d/r['filename'];im=Image.open(path).convert('RGB');h=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest();assert h==r['pixel_sha256'] and h not in seen;seen.add(h)
     assert len(r['label'])==r['length']==6;assert len(r['font_ids'])==6;assert ''.join(g['character'] for g in r['recipe']['glyphs'])==r['label'];assert all(g['visible_fraction']>=.55 for g in r['recipe']['glyphs'])
     if r['mixed']:assert len(set(r['font_ids']))>=2
     else:assert len(set(r['font_ids']))==1
     k=int(hashlib.sha256(('phase12-text-v1:'+r['label']).encode()).hexdigest()[:8],16)%20;assert k<18 and r['partition']=='train'
     t=r['label'];bucket=r['bucket'];assert t.isascii()
     assert t.isdigit() if bucket=='numeric' else t.isalpha() if bucket=='alphabetic' else t.isalnum() and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)
     r['filename']=str(path.relative_to(OUT)).replace('\\','/');writer.writerow([r['filename'],t]);mf.write(json.dumps(r)+'\n');cells[bucket]+=1;fonts.update(r['font_families']);modes['mixed' if r['mixed'] else 'single']+=1;num+=1;total+=1
     if total%5000==0:status('auditing',audited=total)
   assert num==sh['count']
 assert total==100000 and max(cells.values())-min(cells.values())<=1
 write(OUT/'audit.json',dict(passed=True,images=total,unique_pixels=len(seen),length=6,cells=dict(cells),font_glyph_counts=dict(fonts),modes=dict(modes),all_pixels_rehashed=True,real_accuracy_verified=False))
 (OUT/'README.md').write_text('# DarkBay second corpus —100,000 six-character images\n\nUse labels.csv (relative paths) and manifest.jsonl (glyph/font/recipe details). Balanced numeric/alphabetic/alphanumeric buckets. New35-font Phase2 with mixed-font images; optionalrefinementoff. Original100k corpus preserved. Six is user-selected based on recognizer evaluation summaries, not verified automatic length inference. This is a combined length+typography experiment, not an isolated length ablation. Read protocol.json and audit.json. No recogniser training launched.\n')
 status('complete',audited=total)
 for name in ['STYLE_HANDOFF.md','STYLE_DECISIONS.md']:
  with (PROJECT/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\nSecond DarkBay corpus complete: '+str(OUT)+'.100000six-character images audited;35fonts/mixed typography; old corpus preserved. See protocol for user-selected length/evaluation-exposure and combined-ablation limitation.\n')
 print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if OUT.exists():write(OUT/'failure.json',dict(error=repr(e),time=time.time()));write(OUT/'status.json',dict(state='failed',error=repr(e),time=time.time()))
  raise
