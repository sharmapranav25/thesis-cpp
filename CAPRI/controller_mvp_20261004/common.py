"""Local MVP dependencies. No modification of the renderer or trained hybrid."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
PROJECT = Path(os.environ.get('CAPRI_PROJECT', 'E:/pranav/Final sem/sourcefree_synth'))
RENDERER = Path(os.environ.get('CAPRI_RENDERER', 'E:/pranav/Final sem/captcha-effects'))
sys.path[:0] = [str(PROJECT), str(RENDERER), str(WORK/'learned_controller_20261004')]
HYBRID = PROJECT/'runs/iter2_baseline_15_20261003_04/baseline_training/epoch_15/ema.pt'
ENCODER = WORK/'learned_controller_20261004/training/encoder.pt'
def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n', encoding='utf-8')
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
