"""Finish five100k corpora; reuse the already-running Darkbay dataset."""
import concurrent.futures,json,subprocess,sys,time,hashlib,os
from pathlib import Path
ROOT=Path(__file__).parent;DATA=Path(r'E:\pranav\Final sem\synth_data');PROJECT=Path(r'E:\pranav\Final sem\sourcefree_synth')
FAMILIES=['cryptbb','city','courier','darkbay','outlined_text_captcha']
def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)
def status():
 rows={}
 for f in FAMILIES:
  folder=DATA/(f+'_capri_v2_100k');p=folder/'status.json'
  try:r=json.loads(p.read_text()) if p.exists() else dict(state='queued',generated=0)
  except (OSError,ValueError):r=dict(state='status_unavailable',generated=0)
  if (folder/'failure.json').exists():r.update(state='failed',error=json.loads((folder/'failure.json').read_text()))
  rows[f]=r
 value=dict(pid=os.getpid(),time=time.time(),state='complete' if all(r['state']=='complete' for r in rows.values()) else 'failed' if any(r['state']=='failed' for r in rows.values()) else 'running',target=500000,generated=sum(r.get('generated',0) for r in rows.values()),families=rows)
 write(DATA/'CAPRI_BANK_STATUS.json',value);write(PROJECT/'monitor/capri_bank_100k_status.json',value);return value
def main():
 import msvcrt
 lock=(DATA/'capri_bank_job.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 base=(ROOT/'generate_darkbay_100k.py').read_text();jobs=[]
 for i,family in enumerate(FAMILIES):
  if family=='darkbay':continue
  folder=DATA/(family+'_capri_v2_100k')
  if folder.exists():raise RuntimeError('Existing output requires inspection before restart: '+str(folder))
  code=base.replace('darkbay',family).replace('Darkbay',family).replace('workers=8','workers=6').replace('max_workers=8','max_workers=6').replace('202609280+i',str(202610000+i*100)+'+i')
  code=code.replace("if OUT.exists():write(OUT/'failure.json',dict(error=repr(exc),time=time.time()))", "if OUT.exists():\n   write(OUT/'failure.json',dict(error=repr(exc),time=time.time()))\n   write(OUT/'status.json',dict(state='failed',pid=os.getpid(),error=repr(exc),time=time.time(),generated=0,target=100000))")
  script=ROOT/('generate_'+family+'_100k.py');script.write_text(code);compile(code,str(script),'exec');jobs.append((family,script))
 protocol=dict(families=FAMILIES,images_per_family=100000,total=500000,partition='train',family_parallelism=2,workers_per_new_family=6,darkbay='reuse existing active job; do not duplicate',source_scripts={f:dict(path=str(s),sha256=hashlib.sha256(s.read_bytes()).hexdigest()) for f,s in jobs})
 write(DATA/'CAPRI_BANK_PROTOCOL.json',protocol)
 (DATA/'README_CAPRI_BANK.md').write_text('# CAPRI v2 synthetic bank — generation in progress\n\nTarget100000images per family,500000total. Families cryptbb/city/courier/darkbay/outlined_text_captcha each have a separate *_capri_v2_100k directory. Exact synthetic labels are in each shard labels.csv during generation; combined per-family labels.csv appears after audit. Read CAPRI_BANK_STATUS.json for actual completion, not this heading. Do not use incomplete corpora.\n\nAll corpora: lengths2–8, numeric/alphabetic/alphanumeric, weighted top3font mixture, matched frozen per-family Meta-Sim-inspired adapter. Training partition only. No real text labels or target generator assets. Appearance and OCR performance are not guaranteed. Each completed corpus receives full100000image hash/duplicate/label/bucket audit. No model training launched.\n',encoding='utf8')
 for name in ['STYLE_HANDOFF.md','STYLE_DECISIONS.md']:
  with (PROJECT/'docs'/name).open('a',encoding='utf8') as stream:stream.write('\n\nUser requested same100k corpus for allfive CAPRI families:500000total at E:/pranav/Final sem/synth_data. Darkbay existing active run reused. Otherfour queued2families at a time,6shard workers per family; stable separate seeds. Status CAPRI_BANK_STATUS.json; exact protocols per corpus. No training. Wait for per-family audit.json passed before using corpus.\n')
 def launch(item):
  family,script=item
  with (ROOT/(family+'_100k.stdout.log')).open('w') as out,(ROOT/(family+'_100k.stderr.log')).open('w') as err:
   result=subprocess.run([sys.executable,str(script)],stdout=out,stderr=err)
  if result.returncode:raise RuntimeError(family+' failed; inspect stderr/failure.json, preserved output')
  print('COMPLETE FAMILY',family,flush=True)
 failures=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  pending={pool.submit(launch,j) for j in jobs}
  while pending:
   done,pending=concurrent.futures.wait(pending,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    try:f.result()
    except Exception as exc:failures.append(str(exc))
   status()
 if failures:raise RuntimeError('; '.join(failures))
 while True:
  s=status()
  if s['state']=='complete':break
  if s['state']=='failed':raise RuntimeError('Dataset failure; see bank status')
  time.sleep(5)
 audits={f:json.loads((DATA/(f+'_capri_v2_100k')/'audit.json').read_text()) for f in FAMILIES};assert all(a['passed'] and a['images']==100000 for a in audits.values());write(DATA/'CAPRI_BANK_AUDIT.json',dict(passed=True,total=500000,families=audits))
 with (DATA/'README_CAPRI_BANK.md').open('a',encoding='utf8') as f:f.write('\n\nCOMPLETE: all5family audits passed,500000images total. See CAPRI_BANK_AUDIT.json and per-family labels.csv.\n')
 for name in ['STYLE_HANDOFF.md','STYLE_DECISIONS.md']:
  with (PROJECT/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\nAllfive CAPRI_v2 100k datasets complete and audited:500000images at E:/pranav/Final sem/synth_data. See CAPRI_BANK_AUDIT.json; no OCR training performed.\n')
 print('BANK COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:write(DATA/'CAPRI_BANK_FAILURE.json',dict(error=repr(exc),time=time.time()));raise
