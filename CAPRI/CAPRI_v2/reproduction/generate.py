import argparse,json,sys,copy,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from models import Joint
from controls import decode
from corpus import ROOT,FEATURES,RELEASE
p=argparse.ArgumentParser();p.add_argument('--kind',required=True);kind=p.parse_args().kind;here=Path(__file__).parent;run=here/('run_'+kind);out=run/'real_generation';out.mkdir(exist_ok=False)
torch.set_num_threads(4);confirmation=json.loads((run/'confirmation.json').read_text());seed=max(confirmation,key=lambda r:r['selection_score'])['seed'];result=json.loads((run/'results.json').read_text());threshold=result['results'][str(seed)]['thresholds'];net=Joint(kind).cuda();net.load_state_dict(torch.load(run/f'model_{seed}.pt',weights_only=True));net.eval()
sys.path.insert(0,str(RELEASE));import pipeline
identity=pipeline.verify();source=(ROOT/'runs/style_multilabel_phase12_01/renderer_snapshot.py').read_text()
source=source.replace("    learned=style.get('classifier_features')","    linecount=int(round(style.get('pred_line_count',linecount)))\n    learned=style.get('classifier_features')")
source=source.replace("d.line(ends,fill=tuple(color),width=width)","width=max(1,round(style.get('pred_line_width',width)));d.line(ends,fill=tuple(color),width=width)")
source=source.replace("phase=float(rng.uniform(0,2*math.pi))","phase=float(rng.uniform(0,2*math.pi));amp=style.get('pred_curve_amplitude',amp)")
source=source.replace("d.line(points,fill=tuple(color),width=1)","d.line(points,fill=tuple(color),width=max(1,round(style.get('pred_curve_width',1))))")
source=source.replace("    if learned is not None and learned.get('dot_scatter')!='present':count=0","    count=int(round(style.get('pred_dot_count',count)))\n    if learned is not None and learned.get('dot_scatter')!='present':count=0")
source=source.replace("d.point((x,y),fill=tuple(color))","radius=max(1,round(style.get('pred_dot_radius',1)));d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=tuple(color))")
source=source.replace('ImageFilter.GaussianBlur(.6)',"ImageFilter.GaussianBlur(style.get('pred_blur_radius',.6))")
(out/'renderer_snapshot.py').write_text(source);scope={};exec(compile(source,str(out/'renderer_snapshot.py'),'exec'),scope);render=scope['render'];fonts=json.loads((RELEASE/'assets/fonts.json').read_text());means=decode(json.loads((run/'parameter_baseline.json').read_text())['normalized_median']);lookup={};summary={}
for family in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
 original=ROOT/'runs/style_multilabel_phase12_01';paths=[json.loads(l)['path'] for l in (original/(family+'_predictions.jsonl')).read_text().splitlines()];rows=[]
 with (out/(family+'.jsonl')).open('w') as stream:
  for start in range(0,len(paths),32):
   ims=[np.array(Image.open(path).convert('RGB').resize((192,96),Image.Resampling.BILINEAR)) for path in paths[start:start+32]];x=torch.tensor(np.stack(ims).transpose(0,3,1,2),device='cuda').float()/255
   with torch.inference_mode():logits,params=net(x);scores=logits.sigmoid().cpu().numpy();params=params.cpu().numpy()
   for path,s,v in zip(paths[start:start+32],scores,params):
    row=dict(path=path,scores=dict(zip(FEATURES,map(float,s))),present={f:bool(s[j]>=threshold[f]) for j,f in enumerate(FEATURES)},controls=decode(v));lookup[path]=row;rows.append(row);stream.write(json.dumps(row)+'\n')
 summary[family]={f:float(np.mean([r['present'][f] for r in rows])) for f in FEATURES}
 profile=json.loads((RELEASE/f'profiles/{family}.json').read_text());entries={str(Path(e['reference_path_provenance'])):e for e in profile['entries']};cases=json.loads((original/family/'recipes.json').read_text());folder=out/family;folder.mkdir();records=[];sheet=Image.new('RGB',(900,780),'white');draw=ImageDraw.Draw(sheet)
 for c,label in enumerate(['Real','Same classifier + median controls','Same classifier + learned controls']):draw.text((c*300+3,5),label,fill='black')
 for i,case in enumerate(cases):
  row=lookup[case['reference']];base=entries[str(Path(case['reference']))]['style'];renders=[]
  for mode,controls in [('median',means),('learned',row['controls'])]:
   style=copy.deepcopy(base);flags=row['present'];style['classifier_features']={f:'present' if flags[f] else 'absent' for f in ['straight_interference','curved_interference','dot_scatter']}
   if not flags['background_gradient']:style['gradient_strength']=0
   if not flags['background_blocks']:style['background_block_evidence']={'state':'unsupported'}
   if not flags['background_texture']:style['texture_fraction']=0
   w,h=style['size'];scale=h/96;style.update(pred_line_count=controls['line_count'],pred_line_width=controls['line_width']*scale,curve_count_gain=controls['curve_count']/2,pred_curve_width=controls['curve_width']*scale,pred_curve_amplitude=controls['curve_amplitude']*scale,pred_dot_count=controls['dot_count']*w*h/(192*96),pred_dot_radius=controls['dot_radius']*scale,multilabel_grid=flags['periodic_grid'],multilabel_grid_spacing=max(6,round(controls['grid_spacing']*scale)),multilabel_blur=flags['blur'],pred_blur_radius=controls['blur_radius']*scale)
   im,mask,recipe=render(style,case['text'],str(RELEASE/fonts[case['font_index']]['path']),case['recipe']['seed'],profile['variant']);im.save(folder/f'{i:03d}_{mode}.png');renders.append(im)
   records.append(dict(index=i,mode=mode,text=case['text'],reference=row['path'],controls=controls,flags=flags,style=style))
  if i in [2,4,9,11,16,18]:
   y=[2,4,9,11,16,18].index(i)*125+30
   for j,im in enumerate([Image.open(row['path']).convert('RGB')]+renders):im=im.copy();im.thumbnail((285,105));sheet.paste(im,(j*300+3,y))
 sheet.save(folder/'comparison.png');(folder/'recipes.json').write_text(json.dumps(records,indent=2));print(kind,'generated',family,flush=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'protocol.json').write_text(json.dumps(dict(seed=seed,model_sha256=hashlib.sha256((run/f'model_{seed}.pt').read_bytes()).hexdigest(),real_fitting=False,rendering='median versus learned parameters with same predicted flags/reference/text/font/seed; exploratory even for failed heads',limits='colors/fonts/positions remain CV or generic; captions are generated not decoded; not original library identification',images_inferred=6909,generated=210,release_unchanged=pipeline.verify()==identity),indent=2))
