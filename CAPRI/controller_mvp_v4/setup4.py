import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
V3=WORK/'controller_mvp_v3'
sys.path.insert(0,str(V3))
from boot import U,V1,V2
FAMILIES=['City','CryptBB','DarkBay','Outlined','SimpleCaptcha']
ARMS=['hybrid','single','diverse']
CHARSET='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

