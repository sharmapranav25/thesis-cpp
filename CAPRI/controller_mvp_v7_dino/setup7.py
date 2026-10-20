from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'controller_mvp_v6_random_start'))
from ablate6 import U,V1,V3,V4,WORK,FAMILIES,WEIGHTS_PATH,random_recipes
V5=WORK/'controller_mvp_v5'
