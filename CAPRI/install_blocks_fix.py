from pathlib import Path
import shutil,sys,subprocess
R=Path(__file__).parent;root=Path('E:/pranav/Final sem/sourcefree_synth');source=R/'phase12_release_02';dest=root/'releases/phase12_v2';assert not dest.exists();shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('measure.py','render.py','background_blocks.py'):shutil.copy2(source/'style_pipeline'/name,root/'style_pipeline'/name)
run=root/'runs/style_background_blocks_fix_01';assert not run.exists();shutil.copytree(R/'background_blocks_fix_01',run)
for filename in ('docs/PHASE12_START_HERE.md',):
    p=root/filename;p.write_text(p.read_text(encoding='utf8').replace('phase12_v1','phase12_v2'),encoding='utf8')
p=root/'docs/PHASE12_START_HERE.md'
with p.open('a',encoding='utf8') as f:f.write('\n\nRelease02 fixes false background rectangles. Do not use release01 for new corpora. See releases/phase12_v2/BACKGROUND_FIX.md. Existing profiles were refreshed; custom legacy profiles without rectangle evidence fail closed. Original release01/history remains available for reproduction.\n')
(root/'docs/STYLE_BACKGROUND_BLOCKS_FIX_01.md').write_text((source/'BACKGROUND_FIX.md').read_text(),encoding='utf8')
for filename,text in {'STYLE_DECISIONS.md':'D077 — Traced Courier numeric5 preview to white-background realreference but35syntheticblocks. Old colorvariance heuristic invalidly inferred rectangles. Replaced with conservative connected rectangular-color evidence; no familybranches.596profileaudit: Courier64/96triggered->0; DarkBay74/96supported; CryptBB17/96supported (recalllimitation). Fixtures/168generation checks pass. Release02/current modules fixed, v1/history preserved. BlockRNG isolated going forward; before/after randomstream transition disclosed. No classifiertraining or newpaper.', 'STYLE_HANDOFF.md':'LATEST FIX: use releases/phase12_v2 and docs/PHASE12_START_HERE.md. V1backgroundblock heuristic caused Courier hallucinatedrectangles. CorrectedCV evidence and refreshedprofiles596; 168samples/21cells per8families pass. Courier0/96blocktriggers. Conservative detector canmissrealblocks; no overallqualitypass. No training, automationPAUSED.', 'THESIS_PAPERS.md':'Background-block correction01 is original CV engineering: connected quantized-color rectangular components, fill/text-overlap/boundary checks. No newpaper adopted and no trained rectangleclassifier claimed. See STYLE_BACKGROUND_BLOCKS_FIX_01.md for validation and false-negative limitations.'}.items():
    with (root/'docs'/filename).open('a',encoding='utf8') as f:f.write('\n\n'+text+'\n')
subprocess.run([sys.executable,'-B',str(dest/'pipeline.py'),'verify'],check=True)
sys.path.insert(0,str(root));from style_pipeline.monitor import update
update('background_block_fix_complete','Courier false rectangles fixed; release02 verified across8families. Conservative detector limitations documented. No training running.',running=False,release=str(dest))
print('INSTALLED',dest)
