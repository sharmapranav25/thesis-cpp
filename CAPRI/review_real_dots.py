import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
root=Path('E:/pranav/Final sem/sourcefree_synth');base=root/'feature_specialists_v2/dots';sys.path.insert(0,str(base))
from train_dots import Classifier
from corpus import pop
out=Path('real_dot_review_01');out.mkdir(exist_ok=False)
checkpoint=base/'runs/dots_v1/screen_5x3_pop.pt';shutil.copy2(checkpoint,out/'frozen_model.pt');digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
torch.set_num_threads(2);s=torch.load(out/'frozen_model.pt',map_location='cpu',weights_only=True);m=Classifier(s['grids'],s['use_pop']);m.load_state_dict(s['state_dict']);m.eval()
rows=[];images=[]
for family in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
 records=json.loads((root/f'runs/style_phase1_05/{family}_measurements.json').read_text())['fit']
 records=sorted(records,key=lambda r:hashlib.sha256(r['path'].encode()).hexdigest())[:6]
 sheet=Image.new('RGB',(720,3*150),'#ededed');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(records):
  im=Image.open(r['path']).convert('RGB');resized=np.array(im.resize((128,64),Image.Resampling.BILINEAR));images.append(resized)
  ident=f'{family}_{j+1}';rows.append({'id':ident,'family':family,'path':r['path'],'partition':'previously exposed fit/development','original_size':im.size})
  view=im.copy();view.thumbnail((340,115));x=(j%2)*360;y=(j//2)*150;sheet.paste(view,(x,y+25));draw.text((x+5,y+5),ident,fill='black')
 sheet.save(out/(family+'_blind.png'))
images=np.stack(images);pops=pop(images);x=torch.tensor(np.concatenate([images,pops[...,None]],3).transpose(0,3,1,2)).float()/255
with torch.no_grad():scores=torch.cat([m(x[i:i+4]).sigmoid() for i in range(0,len(x),4)]).numpy()
for r,p in zip(rows,scores):r.update(dot_score=float(p),provisional_present=bool(p>=.5))
(out/'predictions.json').write_text(json.dumps(rows,indent=2))
(out/'protocol.json').write_text(json.dumps({'checkpoint_sha256':digest,'checkpoint':str(checkpoint),'selection':'six fit images per family sorted by hash of path, chosen without model scores','preprocessing':'RGB resize128x64 bilinear then own morphology pop','threshold':.5,'threshold_status':'provisional, not calibrated on real images','review':'assistant visual review of blind sheets, no runtime VLM','limitations':'30 development images; screening checkpoint; final model training ongoing; not unbiased real accuracy'},indent=2))
print('Prepared five blind sheets and frozen-checkpoint predictions',out.resolve())
