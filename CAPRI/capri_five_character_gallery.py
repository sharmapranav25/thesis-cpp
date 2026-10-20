import ast,copy,hashlib,json,math,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).parent;PKG=ROOT/'CAPRI';TRIAL=ROOT/'capri_metasim_01';OUT=ROOT/'capri_five_character_gallery';OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(PKG));import capri
capri.verify();torch.set_num_threads(2)
scope={};exec(compile((TRIAL/'renderer.py').read_text(),str(TRIAL/'renderer.py'),'exec'),scope);render=scope['render'];fonts=json.loads((PKG/'base/assets/fonts.json').read_text())
# Reuse only definitions, without executing training/module-level setup.
tree=ast.parse((TRIAL/'trial_snapshot.py').read_text());defs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['Adapter','context','adjusted']];exec(compile(ast.Module(body=defs,type_ignores=[]),'frozen_trial_definitions','exec'))
summary={}
for family in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
 folder=OUT/family;folder.mkdir();profile=json.loads((PKG/'profiles/conformer'/(family+'.json')).read_text());entries={str(Path(e['reference_path_provenance'])):e for e in profile['entries']};fit=json.loads((TRIAL/family/'splits.json').read_text())['fit'];model=Adapter();model.load_state_dict(torch.load(TRIAL/family/'adapter.pt',map_location='cpu',weights_only=True));model.eval();board=Image.new('RGB',(800,350),'white');draw=ImageDraw.Draw(board)
 draw.text((8,6),family+' | exactly 5 generated characters',fill='black');draw.text((8,27),'CAPRI',fill='black');draw.text((408,27),'Meta-Sim-inspired refinement',fill='black');records=[]
 for i,text in enumerate(['52847','RKMTA','A7K2P']):
  entry=entries[str(Path(fit[i]['path']))];style=copy.deepcopy(entry['style']);seed=63000+i;base,mask,recipe=render(style,text,str(PKG/'base'/fonts[profile['font_index']]['path']),seed,profile['variant']);style['locked_glyph_colors']=[g['color'] for g in recipe['glyphs']];case=dict(style=style,text=text,seed=seed,font=profile['font_index'],variant=profile['variant'],baseline=base,recipe=recipe)
  with torch.no_grad():z=model(torch.tensor([context(entry)],dtype=torch.float32))[0].numpy()
  after,ar,fallback=adjusted(case,z)
  assert len(recipe['glyphs'])==len(ar['glyphs'])==5
  for mode,im in [('before',base),('after',after)]:im.save(folder/f'{i}_{mode}.png')
  for j,im in enumerate([base,after]):
   scale=min(375/im.width,78/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST);board.paste(im,(8+400*j,64+95*i))
  draw.text((8,48+95*i),text,fill='black')
  records.append(dict(text=text,seed=seed,reference=fit[i]['path'],gains=np.exp(z).tolist(),fallback=fallback,before=recipe,after=ar))
 board.save(OUT/(family+'.png'));(folder/'recipes.json').write_text(json.dumps(records,indent=2));summary[family]=dict(samples=3,fallbacks=sum(r['fallback'] for r in records));print(family,summary[family])
(OUT/'README.md').write_text('# Five-character CAPRI / refinement gallery\n\nThree deterministic five-character examples per family: numeric, alphabetic, alphanumeric. First three fit recipes in the pre-existing split order; seeds63000–63002. Frozen hybrid seed731 CAPRI and per-family Meta-Sim-inspired adapters, no retraining or selection based on these renders. Same text/font/glyph geometry/RGB per pair. Not a new quality evaluation. Display enlarged with nearest-neighbor; source PNGs retained.\n\n'+json.dumps(summary,indent=2))
capri.verify()
