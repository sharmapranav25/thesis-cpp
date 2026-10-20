"""Frozen per-family recipe adapters; no DINO, training or VLM at generation time."""
import copy,math,hashlib
import numpy as np
import torch
from refinement_renderer import render as locked_render
class Adapter(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(8,24),torch.nn.Tanh(),torch.nn.Linear(24,6))
 def forward(self,x):return math.log(4)*torch.tanh(self.net(x))
def font_policy(profile,fonts,k):
 scores=np.asarray(profile['font_probabilities'],float)
 if len(scores)!=len(fonts) or not np.isfinite(scores).all() or (scores<0).any() or scores.sum()<=0:raise ValueError('Invalid font probabilities')
 indices=sorted(range(len(fonts)),key=lambda i:(-scores[i],i))[:min(k,len(fonts))]
 weights=scores[indices];weights=weights/weights.sum()
 return dict(indices=indices,weights=weights.tolist(),families=[fonts[i]['family'] for i in indices],basis='top-k frozen font scores; similarity proxy, not verified target font identity')
class Refiner:
 def __init__(self,checkpoint,base_render):
  self.model=Adapter();self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));self.model.eval();self.base_render=base_render;self.digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();torch.set_num_threads(2)
 def __call__(self,style,text,font_path,seed,variant=0):
  base,mask,recipe=self.base_render(style,text,font_path,seed,variant);p=style['_capri_joint_controls'];s=copy.deepcopy(style)
  x=[s['background_edge_fraction'],min(s['small_component_density']*100,1),s.get('texture_fraction',0),s['text_height'],p['curve_count']/4,p['curve_amplitude']/24,min(s.get('background_block_evidence',{}).get('count',0)/35,1),float(s['dark_background'])]
  with torch.inference_mode():g=np.exp(self.model(torch.tensor([x],dtype=torch.float32))[0].numpy())
  for name,j in [('line_count_gain',0),('line_width_gain',1),('dot_count_gain',2),('curve_count_gain',3)]:s[name]=s.get(name,1)*g[j]
  if 'pred_curve_amplitude' in s:s['pred_curve_amplitude']*=g[4]
  else:s['curve_amplitude_gain']=s.get('curve_amplitude_gain',1)*g[4]
  if s.get('background_block_evidence',{}).get('state')=='supported':s['background_block_evidence']['count']=max(0,round(s['background_block_evidence']['count']*g[5]))
  s['locked_glyph_colors']=[r['color'] for r in recipe['glyphs']];failure=None
  try:
   im,newmask,r=locked_render(s,text,font_path,seed,variant)
   assert [(v['character'],v['box'],v['angle'],v['color']) for v in r['glyphs']]==[(v['character'],v['box'],v['angle'],v['color']) for v in recipe['glyphs']]
   assert np.array_equal(np.array(mask),np.array(newmask))
  except ValueError as exc:im,newmask,r=base,mask,recipe;failure=str(exc)
  r['refinement']=dict(method='Meta-Sim-inspired',checkpoint_sha256=self.digest,gains=g.tolist(),fallback=failure is not None,fallback_reason=failure,exact_glyph_geometry_and_rgb_preserved=True)
  return im,newmask,r
