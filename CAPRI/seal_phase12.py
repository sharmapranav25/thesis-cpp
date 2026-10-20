from pathlib import Path
import hashlib,json
root=Path(__file__).parent/'phase12_release_01'
files={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='INTEGRITY.json' and '__pycache__' not in p.parts}
(root/'INTEGRITY.json').write_text(json.dumps(files,indent=2))
