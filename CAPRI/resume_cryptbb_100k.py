"""Finite parallel CAPRI dataset generation with a full post-generation audit."""
import concurrent.futures,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from PIL import Image
PKG=Path(r'E:\pranav\Final sem\sourcefree_synth\releases\CAPRI_v2')
OUT=Path(r'E:\pranav\Final sem\synth_data\cryptbb_capri_v2_100k')
PROJECT=Path(r'E:\pranav\Final sem\sourcefree_synth')
def write(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');tmp.replace(path)
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'shards').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(PKG/'capri.py'),'verify'],check=True)
 shards=[dict(id=i,count=10080 if i<9 else 9280,seed=202610000+i) for i in range(10)]
 protocol=dict(family='cryptbb',count=100000,partition='train',lengths=list(range(2,9)),buckets=['numeric','alphabetic','alphanumeric'],font_mode='similar',font_top_k=3,refinement='on',architecture='conformer',workers=6,shards=shards,package=str(PKG),package_manifest_sha256=hashlib.sha256((PKG/'INTEGRITY.json').read_bytes()).hexdigest(),real_labels_used=False,training=False,limits='Synthetic train corpus; real OCR performance unverified. Font variety + adapter combination has smoke validation only. Repeated text allowed inside train; generated pixels must be unique. Stable text hash reserves other strings for future validation/test.')
 write(OUT/'protocol.json',protocol)
 def status(state,**kw):
  progress=[]
  for s in shards:
   p=OUT/'shards'/f"{s['id']:02d}"/'status.json'
   try:r=json.loads(p.read_text()) if p.exists() else {}
   except (OSError,ValueError):r={}
   progress.append(dict(shard=s['id'],target=s['count'],generated=r.get('generated',0),state=r.get('state','pending_or_starting')))
  row=dict(state=state,pid=os.getpid(),time=time.time(),generated=sum(r['generated'] for r in progress),target=100000,shards=progress,**kw);write(OUT/'status.json',row);write(PROJECT/'monitor/cryptbb_100k_status.json',row)
 def worker(s):
  done=OUT/'shards'/f"{s['id']:02d}"/'status.json'
  if done.exists() and json.loads(done.read_text()).get('state')=='complete':return s['id']
  cmd=[sys.executable,str(Path('E:\\pranav\\CAPRI\\capri_safe_generation.py')),'generate','--family','cryptbb','--kind','conformer','--count',str(s['count']),'--seed',str(s['seed']),'--partition','train','--font-mode','similar','--font-top-k','3','--refinement','on','--out',str(OUT/'shards'/f"{s['id']:02d}")]
  with (OUT/'logs'/f"{s['id']:02d}.log").open('w') as log:
   result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f"Shard {s['id']} failed; see logs/{s['id']:02d}.log")
  return s['id']
 status('generating')
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  pending={pool.submit(worker,s) for s in shards}
  while pending:
   done,pending=concurrent.futures.wait(pending,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:print('completed shard',f.result(),flush=True)
   status('generating')
 status('auditing');seen=set();counts={};font_counts={};fallbacks=0;total=0
 sys.path.insert(0,str(PKG/'base'));from pipeline import partition_for
 with (OUT/'labels.csv').open('w',newline='',encoding='utf8') as cf,(OUT/'manifest.jsonl').open('w',encoding='utf8') as mf:
  writer=csv.writer(cf);writer.writerow(['filename','label'])
  for s in shards:
   folder=OUT/'shards'/f"{s['id']:02d}";rows=0
   with (folder/'manifest.jsonl').open() as stream:
    for line in stream:
     r=json.loads(line);path=folder/r['filename'];im=Image.open(path).convert('RGB');h=hashlib.sha256(str(im.size).encode('ascii'));h.update(im.tobytes());digest=h.hexdigest();assert digest==r['pixel_sha256'],'Pixel hash mismatch';assert digest not in seen,'Duplicate generated pixels';seen.add(digest)
     text=r['label'];assert partition_for(text)=='train';assert len(text)==r['length'] and 2<=len(text)<=8
     if r['bucket']=='numeric':assert text.isascii() and text.isdigit()
     elif r['bucket']=='alphabetic':assert text.isascii() and text.isalpha()
     else:assert text.isascii() and text.isalnum() and any(c.isalpha() for c in text) and any(c.isdigit() for c in text)
     assert r['font_index'] in r['font_policy']['indices'];assert r['refinement']['method']=='Meta-Sim-inspired';assert r['minimum_glyph_visibility']>=.55
     r['filename']=str(path.relative_to(OUT)).replace('\\','/');writer.writerow([r['filename'],text]);mf.write(json.dumps(r)+'\n');key=r['bucket']+'_'+str(r['length']);counts[key]=counts.get(key,0)+1;font_counts[r['font_family']]=font_counts.get(r['font_family'],0)+1;fallbacks+=int(r['refinement']['fallback']);rows+=1;total+=1
     if total%5000==0:status('auditing',audited=total)
   assert rows==s['count']
 assert total==100000 and len(counts)==21 and max(counts.values())-min(counts.values())<=1
 assert hashlib.sha256((PKG/'INTEGRITY.json').read_bytes()).hexdigest()==protocol['package_manifest_sha256']
 write(OUT/'audit.json',dict(passed=True,images=total,unique_pixels=len(seen),all_saved_pixels_rehashed=True,cells=counts,font_counts=font_counts,refinement_fallbacks=fallbacks,real_accuracy_verified=False))
 (OUT/'README.md').write_text('# cryptbb CAPRI v2 —100,000synthetic training images\n\nCompleted and fully audited. Images are in shards/00/images through shards/09/images. Top-level labels.csv uses paths relative to this folder; manifest.jsonl contains text, font, seed, source-profile index, pixel hash and refinement decisions. Exactly100,000images;21buckets each4761or4762. Known synthetic labels, no real text labels. Fonts sampled from top3predicted matches, frozen cryptbb Meta-Sim-inspired adapter on. Read protocol.json and audit.json for limits and fallback counts. Preview PNGs are inside each shard.\n\nUse labels.csv directly to build your recognition training loader; no PARSeq training or LMDB conversion was performed. Future validation/test generation must use the corresponding partition flag to keep text strings disjoint. Do not use these same images as validation.\n',encoding='utf8')
 status('complete',audited=total,refinement_fallbacks=fallbacks);print('COMPLETE',OUT,flush=True)
 for name in ['STYLE_HANDOFF.md','STYLE_DECISIONS.md']:
  with (PROJECT/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\ncryptbb100k CAPRI_v2 dataset complete at '+str(OUT)+'. All100000decoded pixel hashes/unique pixels/labels/length-bucket counts/train-text partition audited. Similar top3font mixture + matched frozen adapter. See audit/protocol; no OCR training or real-performance claim.\n')
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if OUT.exists():
   write(OUT/'failure.json',dict(error=repr(exc),time=time.time()))
   write(OUT/'status.json',dict(state='failed',pid=os.getpid(),error=repr(exc),time=time.time(),generated=0,target=100000))
  raise
