import os,sys,json
from pathlib import Path
native='C:/ProgramData/Anaconda3/Library/bin'
if hasattr(os,'add_dll_directory'):dll=os.add_dll_directory(native)
os.environ['PATH']=native+os.pathsep+os.environ['PATH']
sys.path.insert(0,'E:/pranav/model 2/third_party/parseq')
import torch
from strhub.models.parseq.system import PARSeq
torch.set_num_threads(4);torch.manual_seed(2026)
config=dict(charset_train='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',charset_test='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',max_label_length=12,batch_size=64,lr=.0003,warmup_pct=.1,weight_decay=.01,img_size=[32,128],patch_size=[4,8],embed_dim=384,enc_num_heads=6,enc_mlp_ratio=4,enc_depth=12,dec_num_heads=12,dec_mlp_ratio=4,dec_depth=1,perm_num=6,perm_forward=True,perm_mirrored=True,decode_ar=True,refine_iters=1,dropout=.1)
model=PARSeq(**config).cuda().eval()
with torch.no_grad():prediction=model(torch.zeros(2,3,32,128,device='cuda'))
assert torch.isfinite(prediction).all()
Path('parseq_transfer_preparation').mkdir(exist_ok=True)
Path('parseq_transfer_preparation/model_config.json').write_text(json.dumps(config,indent=2))
result=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,parameters=sum(p.numel() for p in model.parameters()),forward_shape=list(prediction.shape),training_started=False,optimizer_steps=0,pretrained_weights_loaded=False)
Path('parseq_transfer_preparation/readiness.json').write_text(json.dumps(result,indent=2));print(result)
