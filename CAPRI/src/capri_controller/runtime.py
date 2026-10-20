"""One boundary for machine paths, active checkpoints and run destinations."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
WORK=Path(__file__).resolve().parents[2]
CONFIG_PATH=Path(os.environ.get('CAPRI_CONFIG',WORK/'configs/local.json')).resolve()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n',encoding='utf8')

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

CONFIG=read(CONFIG_PATH)
def resolve(path):
    path=Path(path)
    return path.resolve() if path.is_absolute() else (WORK/path).resolve()

def registry():return read(WORK/'models/registry.json')
def model_path(name):
    entry=registry()['models'][name]
    if entry['status']!='active':raise ValueError('Checkpoint is not active: '+name)
    return resolve(entry['path'])

PROJECT=resolve(os.environ.get('CAPRI_PROJECT',CONFIG['project']))
RENDERER=resolve(os.environ.get('CAPRI_RENDERER',CONFIG['renderer']))
# Only external maintained dependencies need a path bridge. MVP directories are never imported.
for dependency in [PROJECT,RENDERER]:
    if str(dependency) not in sys.path:sys.path.insert(0,str(dependency))
RUN_ROOT=resolve(os.environ.get('CAPRI_RUN_DIR',CONFIG['default_run']))
HYBRID=model_path('hybrid')
ENCODER=model_path('custom_encoder')

def target_manifest():return resolve(os.environ.get('CAPRI_TARGET_MANIFEST',CONFIG['target_manifest']))
