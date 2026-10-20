from pathlib import Path
import json,time,subprocess,sys,os
W=Path(__file__).resolve().parent
D=W/'city_five_metasim_01'
SOURCE=Path('E:/pranav/Final sem/synth_data/city_capri_phase3_5digit_100k')
def status(state,**extra):
 p=D/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),time=time.time(),**extra),indent=2));tmp.replace(p)
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
try:
 with (D/'queue_started.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
 start=time.time()
 while True:
  q=read(W/'three_family_pairs_01/status.json');source=read(SOURCE/'status.json')
  if source.get('state')=='failed':raise RuntimeError('Source dataset audit failed')
  if q.get('state') in ('complete','failed') and source.get('state')=='complete':break
  if time.time()-start>86400:raise RuntimeError('Dependencies did not finish within 24 hours; inspect jobs')
  status('queued',waiting_for='three-family sequential queue and source audit');time.sleep(30)
 assert read(SOURCE/'audit.json').get('passed')
 status('running')
 with (D/'generation.log').open('w') as log:subprocess.run([sys.executable,str(D/'generate_city_metasim_100k.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 status('complete')
except BaseException as e:
 status('failed',error=repr(e));raise
