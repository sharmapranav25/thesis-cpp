"""Exploratory measured-recipe fusion; no retraining or family-specific rules."""
import sys,json,copy,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(r'E:\pranav\Final sem\sourcefree_synth');HERE=ROOT/'feature_joint_mapping_01'
sys.path.insert(0,str(HERE))
from corpus import RELEASE
sys.path.insert(0,str(RELEASE));import pipeline
identity=pipeline.verify()
OUT=HERE/'seed731_cv_fusion_01';OUT.mkdir(exist_ok=False)
src=(ROOT/'runs/style_multilabel_phase12_01/renderer_snapshot.py').read_text()
def replace(a,b):
 global src
 assert src.count(a)==1,(a,src.count(a))
 src=src.replace(a,b)
replace("    density=style['background_edge_fraction']", "    rng=np.random.default_rng(np.random.SeedSequence([int(seed),201]))\n    density=style['background_edge_fraction']")
replace("    if learned is not None and learned.get('curved_interference')=='present':", "    rng=np.random.default_rng(np.random.SeedSequence([int(seed),202]))\n    if learned is not None and learned.get('curved_interference')=='present':")
replace("phase=float(rng.uniform(0,2*math.pi))", "phase=float(rng.uniform(0,2*math.pi));amp=style.get('pred_curve_amplitude',amp)")
replace("            d.line(points,fill=tuple(color),width=1)","            curve_width=max(1,round(style.get('pred_curve_width',1)));d.line(points,fill=tuple(color),width=curve_width)")
replace("color=color,width=1,geometry='generic'", "color=color,width=curve_width,geometry='learned amplitude/width when supplied; generic position'")
replace("    count=int(np.clip(max(style['small_component_density']", "    rng=np.random.default_rng(np.random.SeedSequence([int(seed),203]))\n    count=int(np.clip(max(style['small_component_density']")
replace("    background=image.copy();textmask=", "    rng=np.random.default_rng(np.random.SeedSequence([int(seed),204]))\n    background=image.copy();textmask=")
(OUT/'renderer_snapshot.py').write_text(src)
scope={};exec(compile(src,str(OUT/'renderer_snapshot.py'),'exec'),scope);render=scope['render']
fonts=json.loads((RELEASE/'assets/fonts.json').read_text());audit=[]
for kind in ['gridded','conformer']:
 old=HERE/'seed731_preview'/('run_'+kind)/'real_generation'
 dest=OUT/kind;dest.mkdir()
 for family in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
  rows=[json.loads(l) for l in (old/(family+'.jsonl')).read_text().splitlines()];lookup={r['path']:r for r in rows}
  profile=json.loads((RELEASE/f'profiles/{family}.json').read_text());entries={str(Path(e['reference_path_provenance'])):e['style'] for e in profile['entries']}
  cases=json.loads((ROOT/'runs/style_multilabel_phase12_01'/family/'recipes.json').read_text())
  folder=dest/family;folder.mkdir();records=[];sheet=Image.new('RGB',(1200,780),'white');d=ImageDraw.Draw(sheet)
  for j,title in enumerate(['Real','Previous V2','Measured baseline','CV + V2 fusion']):d.text((j*300+3,5),title,fill='black')
  for i,c in enumerate(cases):
   base=copy.deepcopy(entries[str(Path(c['reference']))]);row=lookup[c['reference']];params=row['controls'];flags=row['present'];base.pop('classifier_features',None)
   # No negative learned prediction deletes measured colour, blocks, texture or density.
   # A periodic-grid class alone supplies neither geometry nor corroborating CV evidence.
   base['multilabel_grid']=False;base['multilabel_blur']=False
   fused=copy.deepcopy(base);fused['classifier_features']={'straight_interference':'present','dot_scatter':'present','curved_interference':'present' if flags['curved_interference'] else 'absent'}
   if flags['curved_interference']:
    fused['curve_count_gain']=params['curve_count']/2
    fused['pred_curve_width']=params['curve_width']*base['size'][1]/96
    if kind=='conformer':fused['pred_curve_amplitude']=params['curve_amplitude']*base['size'][1]/96
   ims=[];rs=[]
   for mode,style in [('measured',base),('fused',fused)]:
    im,mask,recipe=render(style,c['text'],str(RELEASE/fonts[c['font_index']]['path']),c['recipe']['seed'],profile['variant'])
    im.save(folder/f'{i:03d}_{mode}.png');ims.append(im);rs.append(recipe)
    records.append(dict(index=i,mode=mode,reference=c['reference'],style=style,prediction=row,recipe=recipe))
   # Independent streams preserve glyph geometry across changed background operations.
   assert [(g['box'],g['angle']) for g in rs[0]['glyphs']]==[(g['box'],g['angle']) for g in rs[1]['glyphs']]
   for key in ['background_palette','foreground_palette','gradient_strength','background_block_evidence','texture_fraction','small_component_density','background_edge_fraction']:
    assert base.get(key)==fused.get(key),key
   if i in [2,4,9,11,16,18]:
    y=[2,4,9,11,16,18].index(i)*125+30
    for j,im in enumerate([Image.open(c['reference']).convert('RGB'),Image.open(old/family/f'{i:03d}_learned.png')]+ims):
     im=im.copy();im.thumbnail((285,105));sheet.paste(im,(j*300+3,y))
   audit.append(dict(kind=kind,family=family,index=i,colour_changed=any(a['color']!=b['color'] for a,b in zip(rs[0]['glyphs'],rs[1]['glyphs']))))
  sheet.save(folder/'comparison.png');(folder/'recipes.json').write_text(json.dumps(records,indent=2));print(kind,family,'complete',flush=True)
assert pipeline.verify()==identity
(OUT/'protocol.json').write_text(json.dumps(dict(status='complete',training=False,real_fitting=False,inference='reuse frozen seed731 predictions on all6909 real images per architecture; no new inference needed',renders=420,scope='105 matched references per architecture, measured and fused variants',release_unchanged=True,colour_fallback_changes=sum(r['colour_changed'] for r in audit),limits='Development mapping trial after real visual review, not unseen evaluation. CV may itself be wrong. Learned grid/blur additions deferred without CV corroboration; dense measured counts retained. Curves remain generic positions, learned settings. Independent operation RNG prevents count-driven glyph RNG drift, but readability fallback can change colour choices.'),indent=2))
note='\n\nV1/V2 fusion trial complete at feature_joint_mapping_01/seed731_cv_fusion_01. Reason: learned negative gates erased measured blocks/texture; narrow synthetic counts suppressed dense backgrounds. Preserve all measured palettes/gradients/blocks/line and dot density, retain Phase2 fonts; frozen seed731 curve presence/count/width contribute, hybrid amplitude used only after synthetic baseline pass. Do not add grids/blur from classification alone without measured support. Both architectures, fivefamilies,105 matched cases each, measured+fusion=420renders. Independent RNG streams isolate line/curve/dot/glyph randomness; glyph geometry and measured style preservation asserted, colour readability fallback remains possible. No training change, no production promotion, previously exposed real data only. Comparison includes historical V2 which retains old RNG; measured-vs-fused is the controlled pair. Release hash check passed.\n'
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (ROOT/'docs'/name).open('a',encoding='utf8') as f:f.write(note)
(OUT/'README.md').write_text('# V1 measurements + V2 frozen inference\n'+note+'\nSee protocol.json and per-family recipes.json for all inputs, decisions and actual rendering operations. This is mapping fusion, not knowledge distillation or retraining.\n')
print('COMPLETE',flush=True)
