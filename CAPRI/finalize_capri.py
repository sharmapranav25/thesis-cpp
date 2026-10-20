import json,hashlib,shutil
from pathlib import Path
local=Path(__file__).parent;src=local/'CAPRI';root=Path(r'E:\pranav\Final sem\sourcefree_synth');dst=root/'releases/CAPRI'
report=json.loads((local/'capri_validation/validation.json').read_text());assert report['status']=='passed'
shutil.copy2(local/'capri_validation/validation.json',src/'docs/PACKAGE_VALIDATION.json')
manifest={str(p.relative_to(src)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='INTEGRITY.json'}
# Preserve the nested base manifest in the outer integrity check too.
manifest['base/INTEGRITY.json']=hashlib.sha256((src/'base/INTEGRITY.json').read_bytes()).hexdigest()
(src/'INTEGRITY.json').write_text(json.dumps(manifest,indent=2))
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))
for name,digest in manifest.items():assert hashlib.sha256((dst/name).read_bytes()).hexdigest()==digest
shutil.make_archive(str(local/'CAPRI-seed731-experimental'),'zip',root_dir=dst.parent,base_dir='CAPRI')
note='\n\nCAPRI packaged as releases/CAPRI: portable runtime with seed731 checkpoints for gridded/hybrid, all5cached fused profiles, exact base release/fonts/licenses, fresh CV+frozen inference CLI, labeled generation2–8 in3buckets, comparison galleries, integrity manifests and limitations.231smoke images generated across both architectures/allfamilies plus fresh profiling2reused images and audit passed. No production promotion; other training jobs remain separate and must not replace snapshot silently. Read releases/CAPRI/NEXT_SESSION.md and README.md. Reproduction training scripts retain original paths and are explicitly reference-only. ZIP at C:/Users/pranav/Documents/ChatGPT/thesis/CAPRI-seed731-experimental.zip.\n'
for name in ['STYLE_HANDOFF.md','STYLE_DECISIONS.md']:
 with (root/'docs'/name).open('a',encoding='utf8') as f:f.write(note)
print('INSTALLED',dst)
