"""Prepare independent finite baseline/refinement jobs from the audited City implementation."""
from pathlib import Path
import json,hashlib
W=Path(__file__).resolve().parent
D=W/'three_family_pairs_01';D.mkdir(exist_ok=False)
families=[('cryptbb','CryptBB',202609297000),('darkbay','DarkBay',202609298000),('courier','Courier',202609299000)]
templates=['generate_city_variable_100k.py','generate_city_metasim_100k.py','city_metasim_worker.py','city_metasim_pair.py']
for family,title,seed in families:
 for filename in templates:
  s=(W/filename).read_text().replace('city',family).replace('City',title).replace('202609286000',str(seed))
  if filename.startswith('generate_'):
   s=s.replace('workers=6','workers=4').replace('max_workers=6','max_workers=4')
  target=D/filename.replace('city',family);compile(s,str(target),'exec');target.write_text(s)
 # Smoke replay reads only the separately generated synthetic fixture.
 source_root=Path('E:/pranav/Final sem/synth_data')/(family+'_capri_v3_variable_2to8_100k')
 smoke_root=D/'smoke'/family/'baseline'
 s=(D/(family+'_metasim_worker.py')).read_text().replace(repr(str(source_root)),repr(str(smoke_root)))
 target=D/(family+'_smoke_worker.py');compile(s,str(target),'exec');target.write_text(s)
(D/'build_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')},indent=2))
print(D)
