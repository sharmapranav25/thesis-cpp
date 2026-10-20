"""Frozen DarkBay adapter, applied to background only with exact baseline glyph layers."""
from pathlib import Path
import copy, hashlib, importlib.util, math
import numpy as np
import torch
from PIL import Image
BASE=Path('E:/pranav/Final sem/sourcefree_synth/releases/CAPRI_v2')
TYPO=BASE.parent/'CAPRI_typography_v3'

class Adapter(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(8,24),torch.nn.Tanh(),torch.nn.Linear(24,6))
 def forward(self,x):return math.log(4)*torch.tanh(self.net(x))

source=(TYPO/'renderer.py').read_text()
capture=source.replace('operations=operations,glyphs=records,visible_fraction=visible,','operations=operations,glyphs=records,visible_fraction=visible,_layers=layers,')
ns={};exec(compile(capture,str(TYPO/'renderer.py'),'exec'),ns);render_layers=ns['render']
prefix=source.split('    rng=np.random.default_rng(np.random.SeedSequence([int(seed),204]))')[0]
bg_ns={};exec(compile(prefix+'    return image,operations\n','background_only_snapshot','exec'),bg_ns);render_background=bg_ns['render']
model=Adapter();checkpoint=BASE/'adapters/darkbay/adapter.pt';checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval();torch.set_num_threads(1)

def refine(style,text,fonts,seed,variant,base,mask,recipe,controls):
 replay,replaymask,captured=render_layers(style,text,fonts,seed,variant)
 assert replay.tobytes()==base.tobytes() and replaymask.tobytes()==mask.tobytes(),'Baseline replay mismatch'
 layers=captured.pop('_layers');s=copy.deepcopy(style);p=controls
 x=[s['background_edge_fraction'],min(s['small_component_density']*100,1),s.get('texture_fraction',0),s['text_height'],p['curve_count']/4,p['curve_amplitude']/24,min(s.get('background_block_evidence',{}).get('count',0)/35,1),float(s['dark_background'])]
 with torch.inference_mode():g=np.exp(model(torch.tensor([x],dtype=torch.float32))[0].numpy())
 for name,j in [('line_count_gain',0),('line_width_gain',1),('dot_count_gain',2),('curve_count_gain',3)]:s[name]=s.get(name,1)*float(g[j])
 if 'pred_curve_amplitude' in s:s['pred_curve_amplitude']*=float(g[4])
 else:s['curve_amplitude_gain']=s.get('curve_amplitude_gain',1)*float(g[4])
 if s.get('background_block_evidence',{}).get('state')=='supported':s['background_block_evidence']['count']=max(0,round(s['background_block_evidence']['count']*float(g[5])))
 background,ops=render_background(s,text,fonts,seed,variant);image=background.copy();r=copy.deepcopy(recipe)
 for color,layer in layers:image.paste(tuple(color),mask=layer)
 failure=None;final=np.asarray(image).astype(float)
 for i,(_,layer) in enumerate(layers):
  counter=background.copy()
  for j,(color,other) in enumerate(layers):
   if i!=j:counter.paste(tuple(color),mask=other)
  fraction=float((np.max(np.abs(final-np.asarray(counter).astype(float)),axis=2)[np.asarray(layer)>100]>=12).mean())
  r['glyphs'][i]['visible_fraction']=fraction
  if not np.isfinite(fraction) or fraction<.55:failure='Refinement reduced individual glyph visibility';break
 visible=float((np.max(np.abs(final-np.asarray(background).astype(float)),axis=2)[np.asarray(mask)>100]>=12).mean())
 if not np.isfinite(visible) or visible<.55:failure='Refinement reduced union glyph visibility'
 if failure:image=base;r=copy.deepcopy(recipe)
 else:
  r['operations']=ops;r['background_sha256']=hashlib.sha256(background.tobytes()).hexdigest();r['visible_fraction']=visible
  if style.get('multilabel_blur',False):
   from PIL import ImageFilter
   image=image.filter(ImageFilter.GaussianBlur(.6));r['operations'].append(dict(kind='multilabel_blur',radius=.6,geometry='generic strength, not estimated'))
 assert [{k:v for k,v in x.items() if k!='visible_fraction'} for x in r['glyphs']]==[{k:v for k,v in x.items() if k!='visible_fraction'} for x in recipe['glyphs']]
 r['refinement']=dict(method='Meta-Sim-inspired frozen DarkBay adapter',checkpoint_sha256=checkpoint_hash,gains=g.tolist(),fallback=failure is not None,fallback_reason=failure,exact_glyph_geometry_and_rgb_preserved=True,broad_font_refit=False,changed_pixels=image.tobytes()!=base.tobytes())
 return image,mask,r
