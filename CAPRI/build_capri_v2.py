from pathlib import Path
import shutil,json,hashlib
ROOT=Path(__file__).parent;OUT=ROOT/'CAPRI_v2';OUT.mkdir(exist_ok=False)
shutil.copytree(ROOT/'CAPRI',OUT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','INTEGRITY.json'))
# Nested original release is immutable and needs its original manifest.
shutil.copy2(ROOT/'CAPRI/base/INTEGRITY.json',OUT/'base/INTEGRITY.json')
shutil.copy2(ROOT/'capri_refinement_runtime.py',OUT/'refinement.py');shutil.copy2(ROOT/'capri_metasim_01/renderer.py',OUT/'refinement_renderer.py')
(OUT/'adapters').mkdir();registry={}
for family in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
 dest=OUT/'adapters'/family;dest.mkdir()
 for name in ['adapter.pt','results.json','freeze.json','splits.json']:shutil.copy2(ROOT/'capri_metasim_01'/family/name,dest/name)
 registry[family]=dict(kind='conformer',profile_sha256=hashlib.sha256((OUT/'profiles/conformer'/(family+'.json')).read_bytes()).hexdigest(),checkpoint='adapters/'+family+'/adapter.pt')
(OUT/'adapters/registry.json').write_text(json.dumps(registry,indent=2))
shutil.copy2(ROOT/'CAPRI_METASIM_PLAN.md',OUT/'docs/METASIM_PLAN.md');shutil.copy2(ROOT/'capri_metasim_01/RESULTS.md',OUT/'docs/METASIM_RESULTS.md')
src=(OUT/'base/pipeline.py').read_text();src=src.replace("ROOT=Path(__file__).resolve().parent","ROOT=Path(__file__).resolve().parent/'base'")
src=src.replace('def generate(args):','def generate(args):\n    from refinement import font_policy')
src=src.replace("if args.count<1:","policy=font_policy(profile,fonts,args.font_top_k) if args.font_mode=='similar' else dict(indices=[profile['font_index']],weights=[1.],families=[fonts[profile['font_index']]['family']]);font_rng=np.random.default_rng(np.random.SeedSequence([args.seed,918]))\n    if args.count<1:")
src=src.replace("style=copy.deepcopy(entry['style']);font_index=profile['font_index'];", "style=copy.deepcopy(entry['style']);style['_capri_joint_controls']=entry['joint_prediction']['controls'];font_index=int(font_rng.choice(policy['indices'],p=policy['weights']));")
src=src.replace("                    if rng.random()<.3:font_index=int(rng.integers(len(fonts)))", "                    # Font variety always follows the similar-font pool, including diverse mode.")
src=src.replace("if args.recipes:record['recipe']=recipe", "record['font_family']=fonts[font_index]['family'];record['font_policy']=policy;record['refinement']=recipe.get('refinement',{'method':'off'})\n            if args.recipes:record['recipe']=recipe")
(OUT/'generation.py').write_text(src)
cli=(OUT/'capri.py').read_text();cli=cli.replace("g.add_argument('--recipes',action='store_true')", "g.add_argument('--recipes',action='store_true');g.add_argument('--font-mode',choices=['similar','single'],default='similar');g.add_argument('--font-top-k',type=int,choices=range(1,13),default=3);g.add_argument('--refinement',choices=['auto','on','off'],default='auto')")
cli=cli.replace('  pipeline.render=render;pipeline.generate(args)', '''  import generation
  from refinement import Refiner
  registry=json.loads((ROOT/'adapters/registry.json').read_text());adapter=registry.get(profile['family']);matched=adapter is not None and adapter['kind']==args.kind and adapter['profile_sha256']==hashlib.sha256(Path(args.profile).read_bytes()).hexdigest()
  if args.refinement=='on' and not matched:raise ValueError('No adapter fitted to this exact profile and architecture; use --refinement off or fit a new adapter separately')
  enabled=args.refinement!='off' and matched
  generation.render=Refiner(ROOT/adapter['checkpoint'],render) if enabled else render
  print('Font mode:',args.font_mode,'Refinement:', 'on' if enabled else 'off (disabled or no matching fitted adapter)')
  generation.generate(args)''')
cli=cli.replace("data['capri']=profile['capri'];", "data['capri']=profile['capri'];data['font_mode']=args.font_mode;data['font_top_k']=args.font_top_k;data['refinement_enabled']=enabled;data['refinement_request']=args.refinement;data['version']='CAPRI-v2-font-mixture-metasim';")
(OUT/'capri.py').write_text(cli)
old=(OUT/'README.md').read_text();(OUT/'docs/SEED731_ORIGINAL_README.md').write_text(old)
(OUT/'README.md').write_text('''# CAPRI v2 — similar-font variety + Meta-Sim-inspired refinement

Default generation now samples one font per image from the top3 frozen font-model scores, weighted by those scores, and applies the independently fitted family refinement adapter when available. This is a new generation policy, not classifier retraining. Phase1 remains CV+frozen joint inference; Phase2 retains font adjustment/effects while varying font identity. Text lengths2–8 and numeric/alphabetic/alphanumeric buckets remain balanced by21-cell cycle.

## Use

```powershell
python capri.py verify
python capri.py generate --family cryptbb --kind conformer --count 210 --seed 731 --partition train --out output/cryptbb --recipes --masks
```

Defaults: --font-mode similar --font-top-k 3 --refinement auto. Existing five hybrid profiles have matching adapters: cryptbb, city, courier, darkbay, outlined_text_captcha. Single-font baseline: --font-mode single --refinement off. Font-only ablation: --refinement off. Require refinement explicitly: --refinement on (fails if no matching adapter). All flags and actual font IDs/families/weights/refinement gains/fallbacks are recorded. One font is chosen per image, never arbitrary per-character mixing. Fonts outside the selected pool are never sampled, including diverse mode.

## New families

```powershell
python capri.py profile --references references.json --name new_family --kind conformer --variant 3 --device cpu --out output/new_profile
python capri.py generate --profile output/new_profile/profile.json --kind conformer --count 210 --seed 42 --partition train --out output/new_images --recipes
```

references.json: {"partition":"style_fit","images":["path/to/image.png", ...]}. This runs CV AND frozen inference. Font variety works immediately. Refinement does NOT transfer adapters from another family: auto prints off for an unmatched profile; on errors. A changed/custom profile also fails exact adapter binding. Gridded generation supports font variety but has no fitted adapter. Fitting new adapters is a separate research workflow, not performed silently by generate.

## What refinement changes

Linecount/width,dotcount,curvecount/amplitude,measured blockcount. It renders the chosen font baseline first, then locks its exact glyph RGB/geometry while applying learned recipe gains. Invalid visibility candidates revert to that baseline with an explicit fallback record. Palettes, gradients, fonts, text and layout are not learned by this adapter. No DINO/VLM is loaded for generation; DINO was used only during adapter fitting. Runtime is CPU-capable.

## Evidence and limits

Similar-font selection is a top3 probability-ranking heuristic, not verified font similarity or guaranteed OCR benefit. Probabilities are not target-domain calibrated. Meta-Sim-inspired12step adapters were fitted using the SINGLE-font baseline; applying them with varied fonts is a new combination. Smoke validation establishes operation and invariants, not real OCR gains. DINO and independentCV improved4/5families in the earlier pilot, CryptBB had mixed results. Do not claim full Meta-Sim reproduction, exact target matching, or a production quality gate. All existing real families have historical development exposure. Read docs/METASIM_PLAN.md and docs/METASIM_RESULTS.md.

## Handoff

Read NEXT_SESSION.md. Frozen seed731 models/base/font assets and licenses are included. Original package preserved. requirements.txt lists tested dependencies, not bundled Python. Legacy reproduction sources retain original machine paths and are reference-only. Base integrity manifest remains intact; outer manifest covers the new snapshot. Outputs must be written to fresh directories. Do not change this package in place; create a new version for further experiments.
''')
(OUT/'NEXT_SESSION.md').write_text('''# CAPRI v2 handoff

User explicitly requested font variety plus Meta-Sim-inspired refinement packaged together. Start README.md and run verify. Default similar-font top3 weighted sampling, family-specific refinement auto-enabled for exact five hybrid cached profiles. Prior CAPRI package is historical baseline. No neural classifiers were retrained for this change. Existing three-seed jobs finished before refinement trial; verify actual monitors before starting new work.

Important: adapters trained on single-font recipes. Combination with variety has smoke/invariant checks only; downstream decoding improvement remains untested. No cross-family adapter reuse. New-family profiling does CV and learned inference, but no adapter is fitted automatically. Exact family/profile/kind match required for refinement.

No runtimeVLM, no target generator/fonts/labels. Preserve original/vendor/history, realfit/evaluation boundaries and truthful gates. Papers/failed outcomes in docs and canonical thesis logs. User stopped continuous loop earlier; do not restart automation. Length/alphabet remain chosen synthetic buckets, not inferred real characteristics. Production use and PARSeq training are not authorized by package creation alone.
''')
manifest={str(p.relative_to(OUT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(OUT/'INTEGRITY.json').write_text(json.dumps(manifest,indent=2));print(OUT)
