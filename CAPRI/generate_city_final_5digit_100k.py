"""User-selected Phase3 output: exactly five digits, broad35-font CAPRI typography."""
from pathlib import Path
import os,sys,json,csv,time,hashlib,subprocess,concurrent.futures,collections,shutil,traceback
from PIL import Image
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
PKG=PROJECT/'releases/CAPRI_typography_v3'
RUN=PROJECT/'runs/capri_phase3_city_02'
OUT=Path('E:/pranav/Final sem/synth_data/city_capri_phase3_5digit_100k')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):
 temp=p.with_name(p.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(obj,indent=2),encoding='utf8')
 for retry in range(100):
  try:temp.replace(p);return
  except PermissionError:
   if retry==99:raise
   time.sleep(.2)
def record(message):
 for name in ['STYLE_HANDOFF.md','STYLE_DECISIONS.md','THESIS_PAPERS.md']:
  with (PROJECT/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\n'+message+'\n')

def main():
 assert not OUT.exists(),'Existing dataset preserved; inspect instead of relaunching'
 prior=Path('E:/pranav/Final sem/synth_data/city_capri_v3_variable_2to8_100k')
 assert read(prior/'audit.json')['passed'] and read(prior/'status.json')['state']=='complete'
 assert sha(PKG/'INTEGRITY.json')==read(prior/'protocol.json')['package_manifest_sha256']
 for rel,digest in read(PKG/'INTEGRITY.json').items():assert sha(PKG/rel)==digest,rel
 OUT.mkdir();(OUT/'shards').mkdir();(OUT/'logs').mkdir();(OUT/'provenance').mkdir();shutil.copy2(__file__,OUT/'coordinator_snapshot.py')
 source=(PKG/'generate.py').read_text();needle='ROOT=Path(__file__).resolve().parent';assert source.count(needle)==1;source=source.replace(needle,'ROOT=Path('+repr(str(PKG))+')')
 needle="cells=[(b,n) for b in ('numeric','alphabetic','alphanumeric') for n in sorted(set(a.lengths))]";assert source.count(needle)==1;source=source.replace(needle,"cells=[('numeric',5)]")
 helper="""def make_phase3_text(basegen,rng,bucket,n,partition):
 for attempt in range(10000):
  text,sub=basegen.make_text(rng,bucket,n,partition)
  k=int(hashlib.sha256(('capri-phase3-city-v1:'+text).encode()).hexdigest()[:8],16)%100
  if k<80:return text,sub
 raise RuntimeError('No permitted training text')

"""
 source=source.replace('def main():',helper+'def main():',1)
 needle='text,sub=basegen.make_text(rng,bucket,n,a.partition)';assert source.count(needle)==1;source=source.replace(needle,'text,sub=make_phase3_text(basegen,rng,bucket,n,a.partition)')
 needle="   if i<42:previews.append((image.copy(),text+' | '+('mixed' if mixed else 'single')))";assert needle in source
 source=source.replace(needle,needle+"\n   if (i+1)%500==0:\n    mf.flush();cf.flush();(a.out/'progress.json').write_text(json.dumps(dict(generated=i+1)));print('generated',i+1,flush=True)")
 worker=OUT/'worker_snapshot.py';compile(source,str(worker),'exec');worker.write_text(source)
 checkpoint=next((RUN/'interim_real_checkpoints/crnn_aux').glob('epoch_008_*'))
 assert sha(checkpoint/'frozen.pt')==read(checkpoint/'summary.json')['checkpoint_sha256']
 for name in ['summary.json','synthetic_calibration.json','synthetic_calibration_metrics.json','fit_predictions.jsonl']:
  shutil.copy2(checkpoint/name,OUT/'provenance'/('epoch8_'+name))
 shutil.copy2(RUN/'data/audit.json',OUT/'provenance/reader_data_audit.json');shutil.copy2(RUN/'data/partitions.jsonl',OUT/'provenance/reader_partitions.jsonl')
 scheme=dict(lengths=[5],alphabet='0123456789',groups=['digits'],source='Explicit user instruction after interim model predictions and assistant visual review',user_instruction='creat the final dataset 100k Length 5 and numeric only',model_evidence=read(checkpoint/'summary.json')['fit'],checkpoint_sha256=sha(checkpoint/'frozen.pt'),automatic_full_comparison_estimate=False,comparison_still_running=True,manual_review='24 hash-ordered historical FIT images visually appeared five digits; no transcriptions used in training; limited sample not entire-family proof',validation_used_for_scheme=False,uncertainty_policy='User explicitly chose fixed5/numeric-only, superseding earlier weighted length/group sampling for this dataset; model uncertainty remains in provenance',real_accuracy_verified=False)
 write(OUT/'scheme.json',scheme)
 shards=[dict(id=i,count=10000,seed=202609295500+i) for i in range(10)]
 protocol=dict(stage='Official CAPRI Phase3 user-selected targeted dataset',family='city',images=100000,lengths=[5],alphabet='0123456789',buckets=['numeric'],scheme_sha256=sha(OUT/'scheme.json'),font_policy='35 audited fonts; uniform shuffled decks; mixed fraction0.5; same broad Phase2 typography',refinement='off',partition='train',text_partition='phase12-text-v1 train AND capri-phase3-city-v1 train; protect reader selection/calibration/test identities',workers=4,shards=shards,package=str(PKG),worker_sha256=sha(worker),package_manifest_sha256=sha(PKG/'INTEGRITY.json'),profile_sha256=sha(PROJECT/'releases/CAPRI_v2/profiles/conformer/city.json'),font_manifest_sha256=sha(PKG/'fonts.json'),checkpoint_sha256=scheme['checkpoint_sha256'],real_transcription_labels_used=False,comparison_limit='User-selected scheme, not completed architecture comparison or proven real accuracy. Original count-dependent layout retained; fixed length is not learned counting. All35 fonts include prior reader-heldout fonts.',production_recognizer_training=False)
 write(OUT/'protocol.json',protocol)
 def status(state,**extra):
  rows=[]
  for sh in shards:
   d=OUT/'shards'/f"{sh['id']:02d}";r={}
   try:r=read(d/'audit.json') if (d/'audit.json').exists() else read(d/'progress.json') if (d/'progress.json').exists() else {}
   except (OSError,ValueError):pass
   rows.append(dict(shard=sh['id'],generated=r.get('images',r.get('generated',0)),target=sh['count']))
  row=dict(state=state,pid=os.getpid(),time=time.time(),generated=sum(r['generated'] for r in rows),target=100000,dataset=str(OUT),shards=rows,**extra);write(OUT/'status.json',row);write(PROJECT/'monitor/city_phase3_5digit_status.json',row)
 record('City final targeted dataset STARTED by explicit user choice: '+str(OUT)+'. Exactly5 digits,100000images,all35fonts/mixed typography; optional refiners off. This fixed scheme overrides weighted uncertainty sampling for this corpus; it is not presented as a completed six-model inference result. Reader training/checkpoint evaluation continues unchanged. See scheme.json/protocol.json and dedicated monitor/city_phase3_5digit_status.json; completion awaits full audit.')
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 def run(sh):
  with (OUT/'logs'/f"{sh['id']:02d}.log").open('w') as log:
   subprocess.run([sys.executable,str(worker),'--base',str(PROJECT/'releases/CAPRI_v2'),'--family','city','--count',str(sh['count']),'--seed',str(sh['seed']),'--partition','train','--lengths','5','--mixed-fraction','.5','--out',str(OUT/'shards'/f"{sh['id']:02d}")],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
 status('generating')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  pending={pool.submit(run,sh) for sh in shards}
  while pending:
   done,pending=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:f.result()
   status('generating')
 seen=set();prior_hashes={json.loads(l)['pixel_sha256'] for l in (prior/'manifest.jsonl').open()};real_hashes={r['pixel_sha256'] for values in read(RUN/'data/real_locators.json').values() for r in values}
 fonts=collections.Counter();chars=collections.Counter();modes=collections.Counter();texts=collections.Counter();total=0;minimum_visibility=1.
 with (OUT/'labels.csv').open('w',newline='') as cf,(OUT/'manifest.jsonl').open('w') as mf:
  writer=csv.writer(cf);writer.writerow(['filename','label'])
  for sh in shards:
   d=OUT/'shards'/f"{sh['id']:02d}";labels=list(csv.DictReader((d/'labels.csv').open()));num=0
   for line in (d/'manifest.jsonl').open():
    r=json.loads(line);t=r['label'];assert labels[num]==dict(filename=r['filename'],label=t)
    path=d/r['filename']
    with Image.open(path) as original:im=original.convert('RGB')
    h=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest();assert h==r['pixel_sha256'] and h not in seen and h not in prior_hashes and h not in real_hashes;seen.add(h)
    assert len(t)==r['length']==5 and set(t)<=set('0123456789') and r['bucket']=='numeric' and r['partition']=='train'
    assert int(hashlib.sha256(('phase12-text-v1:'+t).encode()).hexdigest()[:8],16)%20<18
    assert int(hashlib.sha256(('capri-phase3-city-v1:'+t).encode()).hexdigest()[:8],16)%100<80
    assert ''.join(g['character'] for g in r['recipe']['glyphs'])==t and len(r['font_ids'])==5
    for g in r['recipe']['glyphs']:
     assert g['visible_fraction']>=.55;minimum_visibility=min(minimum_visibility,g['visible_fraction']);x0,y0,x1,y1=g['box'];assert 0<=x0<x1<=im.width and 0<=y0<y1<=im.height
    assert (len(set(r['font_ids']))>1)==r['mixed']
    r['filename']=str(path.relative_to(OUT)).replace('\\','/');r['scheme']='user_selected_5_digits';writer.writerow([r['filename'],t]);mf.write(json.dumps(r)+'\n');fonts.update(r['font_families']);chars.update(t);texts[t]+=1;modes['mixed' if r['mixed'] else 'single']+=1;num+=1;total+=1
    if total%5000==0:status('auditing',audited=total)
   assert num==sh['count']
 assert total==100000 and len(fonts)==35 and set(chars)==set('0123456789')
 stats=dict(length_counts={'5':total},group_counts={'digits_only':total},font_glyph_counts=dict(fonts),digit_counts=dict(chars),modes=dict(modes),unique_texts=len(texts),duplicate_text_images=total-len(texts),duplicate_text_policy='Allowed within train only; images unique; reader-heldout text hashes excluded')
 write(OUT/'sampling_statistics.json',stats)
 write(OUT/'audit.json',dict(passed=True,images=total,unique_pixels=len(seen),all_images_rehashed=True,zero_pixel_overlap_with_initial_or_real=True,all_labels_exactly5_ascii_digits=True,all_csv_labels_match_manifest=True,all_glyph_labels_match=True,minimum_glyph_visibility=minimum_visibility,glyph_bounds_passed=True,fonts=35,reader_text_boundaries_preserved=True,labels_sha256=sha(OUT/'labels.csv'),manifest_sha256=sha(OUT/'manifest.jsonl'),sampling=stats,real_accuracy_verified=False))
 (OUT/'README.md').write_text('# City final targeted synthetic dataset\n\n100,000 labelled images, exactly five ASCII digits each, all35 fonts and approximately50% mixed-font images. User-selected scheme after interim reader evidence and a24-image manual fit-pool check; not a verified whole-family ground truth or completed architecture-comparison result. Optional refiners off. Prior datasets and ongoing readers unchanged. Read scheme.json, protocol.json, audit.json, sampling_statistics.json, labels.csv and manifest.jsonl. Duplicate texts are permitted within train; image pixels are unique and reader-heldout text identities are excluded. This dataset includes all35fonts, so the first-reader font test is not font-heldout for a future recognizer trained here. Count-dependent layout remains; fixed-length generation is not proof of learned counting. No final production recognizer trained.\n',encoding='utf8')
 status('complete',audited=total,audit_passed=True)
 record('City final targeted dataset COMPLETE and audited: '+str(OUT)+'.100000unique images,exactly5ASCII digits,all35fonts,'+str(modes['mixed'])+'mixed-font images. CSV/manifest/glyph consistency,visibility/bounds,all pixel rehashes and reader text boundaries passed. User-selected fixed scheme; model uncertainty and manual-review exposure retained in provenance, no real accuracy claim. Ongoing reader comparison remains a separate active job; no production recognizer trained. See dataset audit.json and sampling_statistics.json.')
 print('COMPLETE',OUT,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if OUT.exists():write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));write(OUT/'status.json',dict(state='failed',error=repr(e),time=time.time()))
  raise
