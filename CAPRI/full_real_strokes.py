import ast,json,time,hashlib,shutil
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw
root=Path('E:/pranav/Final sem/sourcefree_synth');src=root/'feature_specialists_v2/strokes_two_stage_v1';here=Path(__file__).parent;out=here/'full_real_stroke_review_01';out.mkdir(exist_ok=False)
# Load the exact model classes without executing the training module's filesystem side effects.
source=(src/'train.py').read_text();tree=ast.parse(source);classes=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['Detector','Shape']],type_ignores=[]);exec(compile(classes,str(src/'train.py'),'exec'))
torch.set_num_threads(2);det=Detector();shape=Shape();hashes={}
for name,model in [('detector',det),('shape',shape)]:
 p=src/'run'/f'{name}.pt';shutil.copy2(p,out/p.name);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));model.eval()
manifest=json.loads((here/'full_real_dot_review_02/manifest.json').read_text());(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'protocol.json').write_text(json.dumps(dict(checkpoint_hashes=hashes,source_sha256=hashlib.sha256(source.encode()).hexdigest(),threshold=.5,min_pixels=12,real_fitting=False,synthetic_gate='failed; user authorizes exploratory whole-dataset inference',preprocessing='RGB bilinear resize128x64',scope='same6909 image paths as full dot pass',limitations='one whole-mask shape output per image; cannot separate multiple strokes or represent both shapes; family distributions are not accuracy',reserved_data='same explicit whole-dataset exposure as dot pass'),indent=2))
def status(**kw):
 r=dict(time=time.time(),**kw);(out/'status.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
def infer(ims):
 x=torch.tensor(np.stack(ims).transpose(0,3,1,2)).float()/255
 with torch.inference_mode():
  mask=det(x).sigmoid()>=.5;pixels=mask.sum((1,2,3));p=shape(mask.float()).softmax(1);kind=torch.where(pixels>=12,p.argmax(1)+1,0)
 return mask[:,0].numpy(),pixels.numpy(),p.numpy(),kind.numpy()
summary={}
for family,paths in manifest.items():
 rows=[];errors=[]
 with (out/(family+'.jsonl')).open('w') as stream:
  for start in range(0,len(paths),32):
   ims=[];valid=[]
   for path in paths[start:start+32]:
    try:
     with Image.open(path) as im:ims.append(np.array(im.convert('RGB').resize((128,64),Image.Resampling.BILINEAR)))
     valid.append(path)
    except Exception as e:errors.append(dict(path=path,error=repr(e)))
   if ims:
    masks,pixels,probs,kinds=infer(ims)
    for path,n,p,k in zip(valid,pixels,probs,kinds):
     r=dict(path=path,prediction=['absent','straight','curved'][k],mask_pixels=int(n),curve_score_given_mask=float(p[1]));rows.append(r);stream.write(json.dumps(r)+'\n')
   if start%512==0:stream.flush();status(stage='inference',family=family,done=min(start+32,len(paths)),total=len(paths))
 counts={k:sum(r['prediction']==k for r in rows) for k in ['absent','straight','curved']};summary[family]=dict(scored=len(rows),counts=counts,errors=errors)
 # Score-independent evenly spaced manifest examples for visual mask review.
 chosen=[rows[i] for i in np.linspace(0,len(rows)-1,6).astype(int)];sheet=Image.new('RGB',(768,450),'#eeeeee');d=ImageDraw.Draw(sheet)
 for j,r in enumerate(chosen):
  im=Image.open(r['path']).convert('RGB').resize((128,64),Image.Resampling.BILINEAR);mask,_,_,_=infer([np.array(im)]);x=j%2*384;y=j//2*150
  sheet.paste(im.resize((256,128)),(x,y+20));sheet.paste(Image.fromarray(np.uint8(mask[0]*255)),(x+256,y+20));d.text((x+3,y+3),f"{j+1}: {r['prediction']} mask pixels={r['mask_pixels']}",fill='black')
 sheet.save(out/(family+'_masks.png'));(out/'partial_summary.json').write_text(json.dumps(summary,indent=2));status(stage='family_complete',family=family,**summary[family])
(out/'summary.json').write_text(json.dumps(summary,indent=2));status(stage='complete')
