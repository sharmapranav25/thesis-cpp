from pathlib import Path
import json,shutil,sys
from PIL import Image,ImageDraw
HERE=Path(__file__).parent;out=HERE/'count_font_bank_01';ROOT=Path('E:/pranav/Final sem/sourcefree_synth')
results=json.loads((out/'results.json').read_text());counts=json.loads((out/'count_inference.json').read_text());total=0
board=Image.new('RGB',(1000,8*200),'#f1f3f5');draw=ImageDraw.Draw(board)
for j,(name,r) in enumerate(results.items()):
    draw.text((8,j*200+3),name+' | reference above, final Phase 2 below',fill='black')
    comparison=Image.open(out/f'{name}_comparison.png')
    for i,record in enumerate(r['recipes']):
        recipe=record['recipe'];n=record['requested_length'];assert len(recipe['glyphs'])==len(recipe['text'])==n
        assert all(g['visible_fraction']>=.55 for g in recipe['glyphs']);total+=1
        reference=comparison.crop((10+i*250,30,245+i*250,130));reference.thumbnail((235,75));board.paste(reference,(10+i*250,j*200+20))
        final=Image.open(out/f'{name}_{i:02d}.png');final.thumbnail((235,75));board.paste(final,(10+i*250,j*200+105));draw.text((10+i*250,j*200+184),f'Estimated length: {n}',fill='black')
assert total==32
board.save(out/'all_families_final.png')
(out/'validation.json').write_text(json.dumps(dict(renders=32,families=8,requested_equals_rendered_glyph_count=True,individual_glyph_visibility_pass=True,background_operations_unchanged_between_phases=True,real_accuracy_measured=False,gate_pass=False),indent=2))
shutil.copy2(HERE/'render_count_font_bank.py',out/'render_count_font_bank.py');shutil.copy2(HERE/'train_character_count.py',out/'train_character_count.py')
table='\n'.join('| '+name+' | '+', '.join(map(str,c['lengths']))+' |' for name,c in counts.items())
report='''# Phase 1 count + Phase 2 fonts: all-family preview 01

User requested testing through Phase2 across the existing bank. Completed32 final renders,4 per8sources, plus32 Phase1 renders. No retraining or optional refinement/DINO. Preserved historical experiments and vendor files. Automation remains paused.

Sequence: Phase1 fresh CV with pixel-hash-matched cached frozen learned feature predictions, plus fresh frozen countCNN inference; Phase2 fresh frozen font inference. Count aggregated over212 existing fit/reference images; font inference over up to16 per family. These are exposed development references, not independent validation. No real labels used. Existing selected rendering variants retained. Generic uppercase/digit generation alphabet is a fixed choice, NOT an inferred character type; it explains letters in numeric-looking families.

Length policy declared in script before rendering: mean calibrated synthetic-model posterior per family; four deterministic stratified quantiles(.125,.375,.625,.875). This explores model uncertainty and is NOT a proven family length distribution, calibrated real margin, or guaranteed +/-1 interval. Supported lengths2–8 only. No manual family-specific count correction. Font and count checkpoint hashes verified unchanged. Original three learned feature models hash-checked for original5; cached pixel-matched inference reused across all8. Fresh CV and font inference performed for every displayed source.

| Family | Four generated lengths |
|---|---|
'''+table+'''

Validation:32/32 requested text lengths equal renderer glyph records; all glyph contributions>=0.55 and built-in bounds checks pass. For each paired Phase1/Phase2 image, background-operation records match exactly. These check rendering integrity, not style or count correctness.

Assistant visual review of both full comparison boards: City/Pymlex counts plausible; Courier and Roncuevas visibly overcount; outlined usually overcounted too. CryptBB/Sloth length variation may reflect model errors rather than real variation. DarkBay retains broad palette/block resemblance but count variations remain unverified. Font matching still selects pixel-like fonts for outlined/Pymlex; outlined effect remains incorrect and Sloth hollow styling weak. Some foreground contrast and background/noise mismatches persist. No quality gate pass, real accuracy claim or promotion as a reliable count system. This is a requested integration preview.

Artifacts: all_families_final.png, bank_1.png, bank_2.png, per-family comparison images, individual Phase1/final PNGs and masks, results.json with text/recipes, count_inference.json with212 probabilities/pixel hashes, protocol.json with checkpoint hashes, validation.json. Source entrypoint preserved in run with count model dependency. No new paper adopted; existing count/font/feature methods only.
'''
(out/'REPORT.md').write_text(report,encoding='utf8')
if '--save' in sys.argv:
    dst=ROOT/'runs/style_count_font_bank_01';assert not dst.exists();shutil.copytree(out,dst)
    (ROOT/'docs/STYLE_COUNT_FONT_BANK_01.md').write_text(report,encoding='utf8')
    updates={
      'STYLE_PIPELINE_PLAN.md':'LATEST USER ORDER: Phase1 now includes synthetic-trained frozen CHARACTER COUNT inference with uncertainty, alongside CV and frozen style features. Phase2 font adjustment. OptionalPhase3 refinement remains outside default pipeline; DINO comparison only proposed. No character-type inference. Count uncertainty is not a verified distribution or +/-1 bound. AutomationPAUSED.',
      'STYLE_HANDOFF.md':'LATEST COMPLETE: runs/style_count_font_bank_01 tests revised Phase1(count included)->Phase2 fonts across8sources,32renders. No optional refinement. Read STYLE_COUNT_FONT_BANK_01.md. Countoverestimates Courier/Roncuevas/outlined; no reliability gatepass. All render integritychecks pass; noactivejob. AutomationPAUSED.',
      'STYLE_DECISIONS.md':'D071 — User moved count into Phase1 and requested allfamilies Phase2preview. Used frozen countCNN meanposterior with4predeclared stratifiedquantiles, preserving uncertainty without asserting real calibration. Freshcountinference212references; CV plus cached pixel-matched features and freshfonts.32finalrenders pass glyphcount/visibility/background-operation checks. Real count andfontfailures documented; no silent family overrides or successgate. No retraining/optionalrefinement/automationrestart.',
      'THESIS_PAPERS.md':'All-family count+font integration01 reuses previously attributed feature/font methods and original countCNN. No newpaper or DINO integration. See STYLE_COUNT_FONT_BANK_01.md for mixed results; renderer integrity is not real length accuracy.'}
    for filename,text in updates.items():
        with (ROOT/'docs'/filename).open('a',encoding='utf8') as f:f.write('\n\n'+text+'\n')
    sys.path.insert(0,str(ROOT));from style_pipeline.monitor import update
    update('count_font_bank_complete','All8families Phase1(count)+Phase2fonts complete:32renders. Count transfer failures documented; automation paused.',running=False,run=str(dst))
print(report)
