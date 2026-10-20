"""Recover exact initial pilot sources after documented additive adapter changes.

Every recovered byte stream must match the hash saved at execution time.
"""
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read
root=Path(__file__).resolve().parents[1]
run=root/'runs/feature_bank_pilot_01'
for rel,expected in read(run/'status.json')['code_sha256'].items():
    source=root/rel; data=source.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:
        text=data.decode('utf-8')
        if source.name=='php_pilot.php':
            text=text.replace(", 'random_tape_replay'",'') # no-op: exact compact list below
            text=text.replace(",'random_tape_replay','capture_random_tape'",'')
            start=text.index("        if ($cfg['capture_random_tape'] ?? false) {")
            end=text.index('        spl_autoload_register',start)
            text=text[:start]+text[end:]
            start=text.index("        if ($cfg['capture_random_tape'] ?? false) {")
            end=text.index('    } else {',start)
            text=text[:start]+text[end:]
        elif source.name=='__main__.py':
            text=text.replace('from pathlib import Path\n','').replace('from .core import read,write,digest,new_run,audit_annotations\n','')
            start=text.index('    p = commands.add_parser("audit-freeze"')
            end=text.index('    args = parser.parse_args()',start)
            text=text[:start]+text[end:]
            start=text.index('    elif args.command == "audit-freeze":')
            end=text.index('\nif __name__',start)
            text=text[:start]+text[end:]
        data=text.encode('utf-8')
    if hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError('Cannot recover exact saved source: '+rel)
    dest=run/'code_snapshot'/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.read_bytes()!=data: raise ValueError('Existing snapshot differs')
    dest.write_bytes(data)
print('All initial pilot code snapshots match execution-time hashes')
