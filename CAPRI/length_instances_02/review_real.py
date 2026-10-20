from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
from experiment import Net,W,H,ROOT
real=json.loads((ROOT/'real_inference.json').read_text());results=json.loads((ROOT/'results.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
net=Net(True).to(device);net.load_state_dict(torch.load(ROOT/'instance_transformer_731.pt',map_location=device,weights_only=True));net.eval();threshold=results['instance_transformer_731']['object_threshold'];report=[]
for fam,record in real.items():
 gallery=Image.new('RGB',(W*3,130*4),'white');d=ImageDraw.Draw(gallery)
 for i,path in enumerate(record['paths'][:12]):
  image=Image.open(path).convert('RGB').resize((W,H));x=torch.tensor(np.array(image).transpose(2,0,1)).float()[None].to(device)/255
  with torch.inference_mode():c,o,b=net(x)
  probs=o.sigmoid()[0].cpu().numpy();boxes=b[0].cpu().numpy();chosen=boxes[probs>=threshold];head=int(c.argmax(1));local=ImageDraw.Draw(image)
  for cx,cy,bw,bh in chosen:local.rectangle(((cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H),outline=(255,0,0),width=1)
  xx=i%3*W;yy=i//3*130;d.text((xx+3,yy+4),f'{fam} #{i}: head={head}, instances={len(chosen)}',fill='black');gallery.paste(image,(xx,yy+28));report.append(dict(family=fam,index=i,count_head=head,instances=len(chosen),boxes=chosen.tolist(),ground_truth=None))
 gallery.save(ROOT/(fam+'_predictions.png'))
(ROOT/'real_visual_predictions.json').write_text(json.dumps(report,indent=2));print('Five real prediction galleries; no real accuracy claim.')
