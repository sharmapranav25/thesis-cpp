"""Prepare a new finite recovery coordinator, preserving prior scripts and logs."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=(ROOT/'recover_capri_bank.py').read_text()
old=" t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)"
new=""" import os
 t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,indent=2))
 for retry in range(100):
  try:t.replace(p);return
  except PermissionError:
   if retry==99:raise
   time.sleep(.2)"""
assert old in source
source=source.replace(old,new)
source=source.replace("history=folder/'recovery_history';history.mkdir(exist_ok=False)","history=folder/('recovery_history_'+str(time.time_ns()));history.mkdir(exist_ok=False)")
oldline=" source=(ROOT/('generate_'+f+'_100k.py')).read_text();"
assert oldline in source
source=source.replace(oldline," source=(ROOT/('generate_'+f+'_100k.py')).read_text();source=source.replace(\" tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');tmp.replace(path)\", "+repr(new.replace('p.with_name','path.with_name').replace('p.name','path.name').replace('json.dumps(v,','json.dumps(obj,').replace('t.replace(p)','t.replace(path)'))+");")
source=source.replace("f+'_recovery.log'","f+'_recovery_v2.log'")
source=source.replace("'resume_'+f+'_100k.py'","'resume_'+f+'_100k_v2.py'")
compile(source,'recover_capri_bank_v2.py','exec')
(ROOT/'recover_capri_bank_v2.py').write_text(source)
print('Prepared retry-safe recovery coordinator; no dataset modified by preparation.')
