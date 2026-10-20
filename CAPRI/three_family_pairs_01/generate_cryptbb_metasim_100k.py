"""Finite generation and full audit of the separately requested variable-length CryptBB corpus."""
from pathlib import Path
import os,sys,json,csv,time,hashlib,subprocess,concurrent.futures,collections
from PIL import Image
WORK=Path(__file__).resolve().parent
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
PKG=PROJECT/'releases/CAPRI_typography_v3'
SOURCE=Path('E:\\pranav\\Final sem\\synth_data\\cryptbb_capri_v3_variable_2to8_100k')
OUT=Path('E:\\pranav\\Final sem\\synth_data\\cryptbb_capri_v3_variable_2to8_100k_metasim')
def write(p,obj):
 temp=p.with_name(p.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(obj,indent=2))
 for retry in range(100):
  try:temp.replace(p);return
  except PermissionError:
   if retry==99:raise
   time.sleep(.2)
def main():
 OUT.mkdir(exist_ok=False);(OUT/'shards').mkdir();(OUT/'logs').mkdir()
 assert json.loads((SOURCE/'audit.json').read_text())['passed']
 source=(WORK/'cryptbb_metasim_worker.py').read_text();worker=OUT/'worker_snapshot.py';worker.write_text(source)
 (OUT/'cryptbb_metasim_pair.py').write_bytes((WORK/'cryptbb_metasim_pair.py').read_bytes())
 shards=[dict(id=i,count=9996 if i<9 else 10036,seed=202609297000+i) for i in range(10)]
 protocol=dict(family='cryptbb',images=100000,lengths=list(range(2,9)),length_basis='Broad first-stage 2-8 support; no target length assumed',evaluation_exposure='Uses existing frozen CryptBB style profile; historical reference exposure remains unchanged. No new real labels consumed.',font_policy='35-font uniform shuffled decks; mixed fraction0.5; bounded per-glyph variation',refinement='frozen CryptBB adapter paired replay; no refit',baseline=str(SOURCE),adapter_sha256=hashlib.sha256((PROJECT/'releases/CAPRI_v2/adapters/cryptbb/adapter.pt').read_bytes()).hexdigest(),pair_code_sha256=hashlib.sha256((OUT/'cryptbb_metasim_pair.py').read_bytes()).hexdigest(),buckets=['numeric','alphabetic','alphanumeric'],partition='train',shards=shards,workers=4,package=str(PKG),worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),package_manifest_sha256=hashlib.sha256((PKG/'INTEGRITY.json').read_bytes()).hexdigest(),comparison_limit='Broad-font variable-length generation; renderer still has count-dependent layout, so count-shortcut robustness is not established',recognizer_training=False)
 write(OUT/'protocol.json',protocol)
 def status(state,**extra):
  rows=[]
  for sh in shards:
   d=OUT/'shards'/f"{sh['id']:02d}";p=d/'audit.json';q=d/'progress.json';r={}
   try:r=json.loads(p.read_text()) if p.exists() else (json.loads(q.read_text()) if q.exists() else {})
   except (OSError,ValueError):pass
   rows.append(dict(shard=sh['id'],generated=r.get('images',r.get('generated',0)),target=sh['count'],state=r.get('state','running_or_queued')))
  row=dict(state=state,pid=os.getpid(),time=time.time(),generated=sum(r['generated'] for r in rows),target=100000,shards=rows,**extra);write(OUT/'status.json',row);write(PROJECT/'monitor/cryptbb_variable_100k_metasim_status.json',row)
 def run(sh):
  with (OUT/'logs'/f"{sh['id']:02d}.log").open('w') as f:
   args=[sys.executable,str(worker),'--base',str(PROJECT/'releases/CAPRI_v2'),'--family','cryptbb','--count',str(sh['count']),'--seed',str(sh['seed']),'--partition','train','--lengths','2','3','4','5','6','7','8','--mixed-fraction','.5','--out',str(OUT/'shards'/f"{sh['id']:02d}")];result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f"Shard{sh['id']}failed; see its log")
 status('generating')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  pending={pool.submit(run,s) for s in shards}
  while pending:
   done,pending=concurrent.futures.wait(pending,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:f.result()
   status('generating')
 status('auditing');seen=set();cells=collections.Counter();fonts=collections.Counter();modes=collections.Counter();total=0;changed=0;fallbacks=0
 with (OUT/'labels.csv').open('w',newline='') as cf,(OUT/'manifest.jsonl').open('w') as mf:
  writer=csv.writer(cf);writer.writerow(['filename','label'])
  for sh in shards:
   d=OUT/'shards'/f"{sh['id']:02d}";num=0
   originals=(SOURCE/'shards'/f"{sh['id']:02d}"/'manifest.jsonl').open()
   with (d/'manifest.jsonl').open() as stream:
    for line in stream:
     r=json.loads(line);path=d/r['filename'];im=Image.open(path).convert('RGB');h=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest();assert h==r['pixel_sha256'] and h not in seen;seen.add(h)
     old=json.loads(next(originals));assert r['baseline_pixel_sha256']==old['pixel_sha256']
     assert all(r[k]==old[k] for k in ['label','length','font_ids','seed','profile_entry','mixed','bucket'])
     assert [{k:v for k,v in g.items() if k!='visible_fraction'} for g in r['recipe']['glyphs']]==[{k:v for k,v in g.items() if k!='visible_fraction'} for g in old['recipe']['glyphs']]
     ref=r['recipe']['refinement'];assert ref['exact_glyph_geometry_and_rgb_preserved'];changed+=int(h!=old['pixel_sha256']);fallbacks+=int(ref['fallback'])
     assert len(r['label'])==r['length'] and 2<=r['length']<=8;assert len(r['font_ids'])==r['length'];assert ''.join(g['character'] for g in r['recipe']['glyphs'])==r['label'];assert all(g['visible_fraction']>=.55 for g in r['recipe']['glyphs'])
     if r['mixed']:assert len(set(r['font_ids']))>=2
     else:assert len(set(r['font_ids']))==1
     k=int(hashlib.sha256(('phase12-text-v1:'+r['label']).encode()).hexdigest()[:8],16)%20;assert k<18 and r['partition']=='train'
     t=r['label'];bucket=r['bucket'];assert t.isascii()
     assert t.isdigit() if bucket=='numeric' else t.isalpha() if bucket=='alphabetic' else t.isalnum() and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)
     r['filename']=str(path.relative_to(OUT)).replace('\\','/');writer.writerow([r['filename'],t]);mf.write(json.dumps(r)+'\n');cells[bucket+'_'+str(r['length'])]+=1;fonts.update(r['font_families']);modes['mixed' if r['mixed'] else 'single']+=1;num+=1;total+=1
     if total%5000==0:status('auditing',audited=total)
   assert num==sh['count']
 assert total==100000 and max(cells.values())-min(cells.values())<=1
 write(OUT/'audit.json',dict(passed=True,images=total,unique_pixels=len(seen),lengths=list(range(2,9)),cells=dict(cells),font_glyph_counts=dict(fonts),modes=dict(modes),all_pixels_rehashed=True,real_accuracy_verified=False,paired_glyph_checks=total,changed_images=changed,fallbacks=fallbacks))
 (OUT/'README.md').write_text('# CryptBB paired Meta-Sim-inspired corpus\n\n100,000 existing CryptBB samples replayed with frozen CryptBB adapter. Same labels/fonts/seeds/glyph geometry/RGB; background recipe gains applied. Broad typography adapter transfer experiment, no refit or recognition-benefit claim. Source corpus preserved; baseline_filename and baseline_pixel_sha256 identify pairs. Read audit.json for changed and fallback counts; labels.csv for training paths. No recognizer training.\n')
 status('complete',audited=total)
 for name in ['STYLE_HANDOFF.md','STYLE_DECISIONS.md']:
  with (PROJECT/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\nCryptBB paired Meta-Sim corpus complete: '+str(OUT)+'.100000 images, lengths2-8, 35fonts/mixed typography, paired glyph preservation audited; source preserved. Frozen old-font CryptBB adapter, no refit; recognition benefit unverified. See protocol for historical profile exposure and count-dependent layout limitation.\n')
 print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if OUT.exists():write(OUT/'failure.json',dict(error=repr(e),time=time.time()));write(OUT/'status.json',dict(state='failed',error=repr(e),time=time.time()))
  raise
