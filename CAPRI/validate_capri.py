import json,subprocess,sys
from pathlib import Path
root=Path(__file__).parent;package=root/'CAPRI';out=root/'capri_validation';out.mkdir(exist_ok=True)
commands=[]
def run(*args):
 subprocess.run([sys.executable,str(package/'capri.py'),*map(str,args)],check=True);commands.append(list(map(str,args)))
for kind in ['conformer','gridded']:
 for family in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
  run('generate','--family',family,'--kind',kind,'--count',21,'--seed',195,'--partition','train','--out',out/(kind+'_'+family),'--recipes')
profile=json.loads((package/'base/profiles/city.json').read_text());ref=out/'references.json';ref.write_text(json.dumps(dict(partition='style_fit',images=[e['reference_path_provenance'] for e in profile['entries'][:2]])))
run('profile','--references',ref,'--name','city_smoke_reused','--kind','conformer','--variant',3,'--out',out/'fresh_profile')
run('generate','--profile',out/'fresh_profile/profile.json','--kind','conformer','--count',21,'--seed',198,'--partition','validation','--out',out/'fresh_generation','--recipes')
run('audit','--corpora',out/'fresh_generation');run('verify')
report=dict(status='passed',commands=commands,generated=231,fresh_profile_images=2,real_smoke_is_reused_development=True,limits='Smoke and integrity validation, not quality validation; no dependency install from scratch tested.')
(out/'validation.json').write_text(json.dumps(report,indent=2))
print('ALL PASSED')
