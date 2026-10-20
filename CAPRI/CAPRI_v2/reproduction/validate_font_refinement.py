from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).parent;PKG=ROOT/'CAPRI_v2';OUT=ROOT/'capri_v2_validation';OUT.mkdir(exist_ok=False);report={}
def run(folder,*extra,count=63):
 cmd=[sys.executable,str(PKG/'capri.py'),'generate','--count',str(count),'--seed','812','--partition','train','--out',str(OUT/folder),'--recipes',*extra];subprocess.run(cmd,check=True)
 return [json.loads(l) for l in (OUT/folder/'manifest.jsonl').read_text().splitlines()]
for family in ['cryptbb','city','courier','darkbay','outlined_text_captcha']:
 rows=run(family,'--family',family,'--refinement','on');ids={r['font_index'] for r in rows};assert len(ids)>=2
 for r in rows:
  assert r['font_index'] in r['font_policy']['indices'];assert len(r['label'])==len(r['recipe']['glyphs']);assert r['refinement']['exact_glyph_geometry_and_rgb_preserved'];assert r['refinement']['method']=='Meta-Sim-inspired'
 report[family]=dict(images=len(rows),font_ids=sorted(ids),fallbacks=sum(r['refinement']['fallback'] for r in rows),retries=json.loads((OUT/family/'status.json').read_text())['retries'])
again=run('repeat','--family','city','--refinement','on',count=21);original=[json.loads(l) for l in (OUT/'city/manifest.jsonl').read_text().splitlines()][:21];assert [r['pixel_sha256'] for r in again]==[r['pixel_sha256'] for r in original]
single=run('single','--family','city','--font-mode','single','--refinement','off',count=21);assert len({r['font_index'] for r in single})==1;assert all(r['refinement']['method']=='off' for r in single)
grid=run('gridded','--family','city','--kind','gridded',count=21);assert all(r['refinement']['method']=='off' for r in grid)
bad=subprocess.run([sys.executable,str(PKG/'capri.py'),'generate','--family','city','--kind','gridded','--refinement','on','--seed','812','--partition','train','--out',str(OUT/'must_not_exist')],capture_output=True,text=True);assert bad.returncode!=0 and 'No adapter fitted' in bad.stderr;assert not (OUT/'must_not_exist').exists()
subprocess.run([sys.executable,str(PKG/'capri.py'),'verify'],check=True)
result=dict(status='passed',families=report,generated=378,repeat_identical=True,single_font_off=True,gridded_auto_off=True,mismatched_adapter_rejected=True,limits='Smoke, determinism and glyph invariants only. New font-mixture/refinement combination has no real quality or OCR evaluation.')
(OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
