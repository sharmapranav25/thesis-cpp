from pathlib import Path
import shutil,json,hashlib
ROOT=Path(__file__).parent;src=ROOT/'CAPRI_v2';project=Path(r'E:\pranav\Final sem\sourcefree_synth');dst=project/'releases/CAPRI_v2'
results=json.loads((ROOT/'capri_v2_validation/results.json').read_text());assert results['status']=='passed'
shutil.copy2(project/'docs/THESIS_PAPERS.md',src/'docs/THESIS_PAPERS.md')
shutil.copy2(ROOT/'capri_v2_validation/results.json',src/'docs/FONT_REFINEMENT_VALIDATION.json');shutil.copy2(ROOT/'validate_capri_v2.py',src/'reproduction/validate_font_refinement.py')
(src/'previews').mkdir()
for f in results['families']:shutil.copy2(ROOT/'capri_v2_validation'/f/'preview.png',src/'previews'/(f+'.png'))
manifest={str(p.relative_to(src)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=src/'INTEGRITY.json'}
(src/'INTEGRITY.json').write_text(json.dumps(manifest,indent=2));shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))
for name,digest in manifest.items():assert hashlib.sha256((dst/name).read_bytes()).hexdigest()==digest
shutil.make_archive(str(ROOT/'CAPRI-v2-fonts-metasim'),'zip',root_dir=dst.parent,base_dir=dst.name)
note='\n\nCAPRI_v2 packaged at releases/CAPRI_v2 by explicit user request: Phase2 weighted top3 frozen-font candidate sampling per image plus optional Meta-Sim-inspired family adapter. Default similar-font mode and refinement auto(on for5exact hybrid profiles). Single/off flags retain baselines. Gridded/new/changed profiles have no fitted adapter; auto visibly off, on rejects; no cross-family sharing.378generated smoke samples verify all5families use>=2fonts from pool, exact glyph geometry/RGB locking, deterministic replay, single-font/off and mismatched-adapter rejection. Results/retries/fallbacks in docs/FONT_REFINEMENT_VALIDATION.json in package. No OCR/real quality validation of combined variety+adapter policy; adapters were fitted on single-font generation. Prior CAPRI preserved; no classifier retraining. ZIP C:/Users/pranav/Documents/ChatGPT/thesis/CAPRI-v2-fonts-metasim.zip. Read new README/NEXT_SESSION.\n'
for n in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (project/'docs'/n).open('a',encoding='utf8') as f:f.write(note)
print('PACKAGED',dst)
