from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
V3=WORK/'controller_mvp_v3'
V4=WORK/'controller_mvp_v4'
sys.path.insert(0,str(V3))
from boot import U,V1,V2
FAMILIES=['City','CryptBB','DarkBay','Outlined','SimpleCaptcha']
WEIGHTS_PATH=Path.home()/'.cache/torch/hub/checkpoints/dino_deitsmall8_pretrain.pth'
