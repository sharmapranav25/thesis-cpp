"""Frozen multi-label inference across five complete source datasets; additive Phase1/2 preview."""
import sys,json,hashlib,copy,time,shutil
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
root=Path('E:/pranav/Final sem/sourcefree_synth');release=root/'releases/phase12_v2';here=Path(__file__).parent;out=here/'multilabel_phase12_01';out.mkdir(exist_ok=True)
sys.path.insert(0,str(root/'feature_multilabel_v1'))
from train import Model
from corpus import FEATURES
sys.path.insert(0,str(release));import pipeline
from style_pipeline.render import render
torch.set_num_threads(4);cv2.setNumThreads(1);identity=pipeline.verify()
run=root/'feature_multilabel_v1/run';confirm=json.loads((run/'confirmation.json').read_text());seed=max(confirm,key=lambda r:r['selection_macro_ap'])['seed'];results=json.loads((run/'results.json').read_text());passed=results['per_feature_gate'];thresholds=results['results'][str(seed)]['thresholds'];checkpoint=run/f'model_{seed}.pt';model=Model().cuda();model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
shutil.copy2(checkpoint,out/'frozen_model.pt');manifest=json.loads((here/'full_real_dot_review_02/manifest.json').read_text());fonts=json.loads((release/'assets/fonts.json').read_text());summary={};lookup={}
def status(**kw):
 (out/'status.json').write_text(json.dumps(dict(time=time.time(),**kw),indent=2));print(json.dumps(kw),flush=True)
(out/'protocol.json').write_text(json.dumps(dict(model_seed=seed,model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),release_integrity=identity,thresholds=thresholds,passed=passed,real_training=False,scope='five complete Tier1 datasets, same6909 paths as prior runs',source='cached audited Phase1 CV styles and Phase2 font inference; new frozen15head image inference',mapping='seven passed heads control compatible effects; failed heads diagnostic only; existing straight CV/profile and font settings retained',phase3=False,production_promoted=False),indent=2))
for family,paths in manifest.items():
 existing=out/(family+'_predictions.jsonl')
 if existing.exists():
  saved=[json.loads(line) for line in existing.read_text().splitlines()]
  assert len(saved)==len(paths),'Partial inference file requires a separate recovery'
  for r in saved:lookup[str(Path(r['path']))]=r
  summary[family]={f:dict(present=sum(r['present'][f] for r in saved),total=len(saved),fraction=float(np.mean([r['present'][f] for r in saved])),synthetic_gate=passed[f]) for f in FEATURES}
  continue
 rows=[]
 with (out/(family+'_predictions.jsonl')).open('w') as stream:
  for start in range(0,len(paths),32):
   ims=[]
   for p in paths[start:start+32]:
    with Image.open(p) as im:ims.append(np.array(im.convert('RGB').resize((192,96),Image.Resampling.BILINEAR)))
   x=torch.tensor(np.stack(ims).transpose(0,3,1,2),device='cuda').float()/255
   with torch.inference_mode():probs=model(x).sigmoid().cpu().numpy()
   for path,p in zip(paths[start:start+32],probs):
    row=dict(path=path,scores=dict(zip(FEATURES,map(float,p))),present={f:bool(p[i]>=thresholds[f]) for i,f in enumerate(FEATURES)});rows.append(row);lookup[str(Path(path))]=row;stream.write(json.dumps(row)+'\n')
   if start%512==0:status(stage='inference',family=family,done=min(start+32,len(paths)),total=len(paths))
 summary[family]={f:dict(present=sum(r['present'][f] for r in rows),total=len(rows),fraction=float(np.mean([r['present'][f] for r in rows])),synthetic_gate=passed[f]) for f in FEATURES}
(out/'distribution_summary.json').write_text(json.dumps(summary,indent=2))
def adapted(style,row):
 s=copy.deepcopy(style);states=s.setdefault('classifier_features',{})
 for f in ['curved_interference','dot_scatter']:states[f]='present' if row['present'][f] else 'absent'
 if not row['present']['background_gradient']:s['gradient_strength']=0
 if not row['present']['background_blocks']:s['background_block_evidence']={'state':'unsupported'}
 if not row['present']['background_texture']:s['texture_fraction']=0
 return s
# Snapshot renderer and add only a generic classifier-gated grid function, preserving original/vendor.
source=(release/'style_pipeline/render.py').read_text()
source=source.replace('    background=image.copy();textmask=', '''    if style.get('multilabel_grid',False):
        spacing=max(6,int(style.get('multilabel_grid_spacing',12)));color=tuple(np.clip(np.array(bg[0],float)+(35 if style['dark_background'] else -35),0,255).astype(int));d=ImageDraw.Draw(image)
        for gx in range(0,w,spacing):d.line((gx,0,gx,h-1),fill=color,width=1)
        for gy in range(0,h,spacing):d.line((0,gy,w-1,gy),fill=color,width=1)
        operations.append(dict(kind='multilabel_grid',spacing=spacing,color=color,geometry='generic spacing, not estimated'))
    background=image.copy();textmask=''')
source=source.replace('    return image,textmask,dict(','''    if style.get('multilabel_blur',False):
        from PIL import ImageFilter
        image=image.filter(ImageFilter.GaussianBlur(.6));operations.append(dict(kind='multilabel_blur',radius=.6,geometry='generic strength, not estimated'))
    return image,textmask,dict(''')
(out/'renderer_snapshot.py').write_text(source);namespace={};exec(compile(source,str(out/'renderer_snapshot.py'),'exec'),namespace);newrender=namespace['render'];comparisons=[]
for family in manifest:
 profile=json.loads((release/f'profiles/{family}.json').read_text());folder=out/family;folder.mkdir(exist_ok=True);(folder/'images').mkdir(exist_ok=True);rng=np.random.default_rng(1801);records=[];previews=[]
 # Preserve per-reference co-occurrence: sample measured styles and their own predictions together.
 entries=[e for e in profile['entries'] if str(Path(e['reference_path_provenance'])) in lookup]
 assert entries,'No cached CV entries matched full-dataset inference'
 for i in range(21):
  bucket=['numeric','alphabetic','alphanumeric'][i//7];length=2+i%7;text,_=pipeline.make_text(rng,bucket,length,'train')
  entry=entries[int(rng.integers(len(entries)))];row=lookup[str(Path(entry['reference_path_provenance']))];s=adapted(entry['style'],row);s['multilabel_grid']=row['present']['periodic_grid'];s['multilabel_blur']=row['present']['blur'];s['multilabel_grid_spacing']=max(6,round(s['size'][1]/7))
  for retry in range(100):
   renderseed=int(rng.integers(0,2**31-1))
   try:im,mask,recipe=newrender(s,text,str(release/fonts[profile['font_index']]['path']),renderseed,profile['variant']);break
   except ValueError:
    if retry==99:raise
  filename=f'images/{i:03d}.png';im.save(folder/filename);records.append(dict(file=filename,text=text,bucket=bucket,length=length,reference=row['path'],scores=row['scores'],present=row['present'],style=s,recipe=recipe,font_index=profile['font_index'],glyphs_verified=len(recipe['glyphs'])==len(text)))
  if i in [2,4,9,11,16,18]:
   original=Image.open(row['path']).convert('RGB');baseline,_,_=render(entry['style'],text,str(release/fonts[profile['font_index']]['path']),renderseed,profile['variant']);previews.append((original,baseline,im))
 (folder/'recipes.json').write_text(json.dumps(records,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)));sheet=Image.new('RGB',(900,6*125+30),'white');d=ImageDraw.Draw(sheet)
 for col,title in enumerate(['Real reference','Previous Phase1/2','Multi-label Phase1/2']):d.text((col*300+5,5),title,fill='black')
 for i,triple in enumerate(previews):
  for col,im in enumerate(triple):
   im=im.copy();im.thumbnail((285,105));sheet.paste(im,(col*300+5,30+i*125))
 sheet.save(folder/'comparison.png');comparisons.append(str(folder/'comparison.png'));status(stage='generated',family=family,count=21)
assert pipeline.verify()==identity
status(stage='complete',inferred=sum(map(len,manifest.values())),generated=105,comparisons=comparisons,production_unchanged=True)

