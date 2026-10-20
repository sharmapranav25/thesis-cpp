"""Recover failed finite datasets only after their worker pools have shut down."""
import json,time,subprocess,sys,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).parent;DATA=Path(r'E:\pranav\Final sem\synth_data');PROJECT=Path(r'E:\pranav\Final sem\sourcefree_synth');FAMILIES=['cryptbb','city','courier','darkbay','outlined_text_captcha'];attempted=set()
def write(p,v):
 import os
 t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,indent=2))
 for retry in range(100):
  try:t.replace(p);return
  except PermissionError:
   if retry==99:raise
   time.sleep(.2)
def recover(f):
 folder=DATA/(f+'_capri_v2_100k');assert folder.resolve().parent==DATA.resolve()
 history=folder/('recovery_history_'+str(time.time_ns()));history.mkdir(exist_ok=False)
 for name in ['failure.json','status.json','protocol.json']:
  p=folder/name
  if p.exists():p.rename(history/name)
 for shard in (folder/'shards').iterdir():
  if not shard.is_dir():continue
  state=json.loads((shard/'status.json').read_text()) if (shard/'status.json').exists() else {}
  if state.get('state')=='complete':continue
  assert shard.resolve().parent==(folder/'shards').resolve()
  if (folder/'logs'/(shard.name+'.log')).exists():shutil.copy2(folder/'logs'/(shard.name+'.log'),history/(shard.name+'.log'))
  shard.rename(history/shard.name)
 source=(ROOT/('generate_'+f+'_100k.py')).read_text();source=source.replace(" tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');tmp.replace(path)", " import os\n t=path.with_name(path.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(obj,indent=2))\n for retry in range(100):\n  try:t.replace(path);return\n  except PermissionError:\n   if retry==99:raise\n   time.sleep(.2)");source=source.replace("OUT.mkdir(parents=True,exist_ok=False);(OUT/'shards').mkdir();(OUT/'logs').mkdir()","OUT.mkdir(parents=True,exist_ok=True);(OUT/'shards').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)")
 source=source.replace(" def worker(s):", " def worker(s):\n  done=OUT/'shards'/f\"{s['id']:02d}\"/'status.json'\n  if done.exists() and json.loads(done.read_text()).get('state')=='complete':return s['id']")
 source=source.replace("str(PKG/'capri.py'),'generate'", "str(Path("+repr(str(ROOT/'capri_safe_generation.py'))+")),'generate'")
 script=ROOT/('resume_'+f+'_100k_v2.py');script.write_text(source);compile(source,str(script),'exec')
 write(folder/'recovery_protocol.json',dict(reason='Refinement invariant assertion previously aborted shard; invalid candidate now rejected and valid baseline returned',wrapper=str(ROOT/'capri_safe_generation.py'),wrapper_sha256=hashlib.sha256((ROOT/'capri_safe_generation.py').read_bytes()).hexdigest(),package_unchanged=True,partial_shards_preserved=str(history),completed_shards_reused=True))
 with (ROOT/(f+'_recovery_v2.log')).open('w') as log:r=subprocess.run([sys.executable,str(script)],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f+' recovery failed; inspect recovery log')
while True:
 for f in FAMILIES:
  folder=DATA/(f+'_capri_v2_100k')
  # failure.json is written only after ThreadPoolExecutor has joined all workers.
  if (folder/'failure.json').exists() and f not in attempted:
   attempted.add(f);recover(f)
 audits={}
 for f in FAMILIES:
  p=DATA/(f+'_capri_v2_100k')/'audit.json'
  if p.exists():audits[f]=json.loads(p.read_text())
 if len(audits)==5 and all(a['passed'] for a in audits.values()):
  write(DATA/'CAPRI_BANK_AUDIT.json',dict(passed=True,total=500000,families=audits,recovered=list(attempted)))
  # Let the original coordinator finish before replacing its historical failure status.
  if not (DATA/'CAPRI_BANK_FAILURE.json').exists():time.sleep(5);continue
  rows={f:json.loads((DATA/(f+'_capri_v2_100k')/'status.json').read_text()) for f in FAMILIES};value=dict(state='complete',target=500000,generated=500000,families=rows,recovered=list(attempted),time=time.time());write(DATA/'CAPRI_BANK_STATUS.json',value);write(PROJECT/'monitor/capri_bank_100k_status.json',value)
  for n in ['STYLE_HANDOFF.md','STYLE_DECISIONS.md']:
   with (PROJECT/'docs'/n).open('a',encoding='utf8') as out:out.write('\n\nCAPRI bank completed500000audited images. Failed refinement candidates reject to valid baseline via capri_safe_generation.py; partial failed shards preserved and completed shards reused. Original immutable package unchanged. Read dataset recovery_protocol.json; exact glyph/label constraints retained.\n')
  break
 time.sleep(10)
