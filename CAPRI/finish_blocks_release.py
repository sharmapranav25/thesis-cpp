from pathlib import Path
import json,hashlib,sys,subprocess
R=Path(__file__).parent;root=R/'phase12_release_02';out=R/'background_blocks_fix_01'
def seal():
    files={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='INTEGRITY.json' and '__pycache__' not in p.parts};(root/'INTEGRITY.json').write_text(json.dumps(files,indent=2))
report='''# Background rectangle false-positive fix (release02)

Confirmed exact faulty Courier sample: reference captcha_screenshot_1772072964.png has text and interference on white, but old renderer produced35background rectangles. Its broad color-variation/palette trigger did not measure rectangle presence. No classifier for background blocks existed. This shared rule, not mixed GAN weights, caused the misleading background.

Replaced it with conservative connected-color-region evidence: median3filter, RGB32quantization, solid connected rectangle fill>=.88, minimum dimensions, text-mask overlap<=.20, at least2sharp sides on each region, at least2regions. Color variation/saturation no longer enables rectangles. Save candidate dimensions/colors and measured count; sample generic rectangles with those measured dimensions/colors and random positions. No reference pixels copied. Missing/uncertain evidence draws no blocks. This is heuristic CV, not trained classification or semantic certainty.

Audited596existing fit profiles, unchanged learned feature/font outputs. Courier64/96old positives ->0/96supported. DarkBay87/96 ->74/96. CryptBB91/96 ->17/96; this drop shows conservative recall, not proof the other79lack blocks. City0->1, outlined0->2, Roncuevas26->6, Pymlex0->0, Sloth0->0. These are unlabeled detection counts, not accuracy. No manually keyed family exception. The shown DarkBay sample loses some real bands, and CryptBB blocks may be missed under clutter. Do not claim full background reconstruction solved.

Fixtures: plain background, colored crossing lines and smooth gradient rejected; three genuine rectangles accepted. Exact problematic Courier sample now has0block operations. Compared same seed/text/profile across all8families. Separate block RNG prevents toggling blocks from consuming line/dot/text random draws going forward; transition from old interleaved RNG can change other sampled details, so before/after is not a pure block-pixel-only ablation. Fonts, fixed feature models and nonblock mapping formulas unchanged. Prior rendering defect and v1artifact preserved; do not reuse stale v1previews for new corpus generation.

Current recommended entrypoint: release02. New profiles must include background_block_evidence; legacy profiles fail closed for rectangles. No target generators/fonts, no VLM, no countmodel, no Phase3, no PARSeqtraining. This is a correction to feature-to-function mapping, not a new paper-derived technique.
'''
(root/'BACKGROUND_FIX.md').write_text(report,encoding='utf8');(out/'REPORT.md').write_text(report,encoding='utf8')
readme=(root/'README.md').read_text();readme=readme.replace('generation release 01','generation release 02').replace('reproduces the accepted Phase1+Phase2 behavior','uses the corrected Phase1+Phase2 behavior');readme='> Release02 replaces the false color-variation-to-rectangles rule. Read [BACKGROUND_FIX.md](BACKGROUND_FIX.md) for evidence and conservative-recall limits.\n\n'+readme;(root/'README.md').write_text(readme,encoding='utf8')
handoff=(root/'TRAINING_HANDOFF.md').read_text().replace('release01','release02');(root/'TRAINING_HANDOFF.md').write_text(handoff,encoding='utf8')
oldvalidation=(root/'VALIDATION.md').read_text();(root/'VALIDATION_V1_HISTORY.md').write_text(oldvalidation,encoding='utf8')
(root/'VALIDATION.md').write_text('''# Release02 validation

Rectangle fixtures pass (plain/lines/gradient negative, rectangles positive).596reference styles refreshed; exact faultyCourier sample has0blocks. All8same-seed comparison boards visually reviewed; background recall remains imperfect. New21cell-generation smoke checks are recorded in validation_summary.json. Prior release01 byteparity test is historical only and intentionally does not apply after the mapping correction; see VALIDATION_V1_HISTORY.md. No recognition training/evaluation claimed.
''',encoding='utf8')
provenance=json.loads((root/'PROVENANCE.json').read_text())
for relative in ('style_pipeline/render.py','style_pipeline/measure.py','style_pipeline/background_blocks.py'):
    provenance[relative]=dict(source='original local rectangle-evidence correction; previous snapshot releases/phase12_v1',sha256=hashlib.sha256((root/relative).read_bytes()).hexdigest())
(root/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2));seal()
checks={}
for family in json.loads((root/'profiles/index.json').read_text()):
    dest=out/'smoke'/family;subprocess.run([sys.executable,'-B',str(root/'pipeline.py'),'generate','--family',family,'--count','21','--seed','7781','--partition','train','--out',str(dest),'--recipes'],check=True)
    rows=[json.loads(s) for s in (dest/'manifest.jsonl').read_text().splitlines()];assert len(rows)==21 and len(set((r['bucket'],r['length']) for r in rows))==21
    assert all(len(r['recipe']['glyphs'])==r['length'] and r['minimum_glyph_visibility']>=.55 for r in rows)
    if family=='courier':assert all(not any(o['kind']=='block' for o in r['recipe']['operations']) for r in rows)
    checks[family]=dict(samples=21,all_cells=True,glyph_checks_pass=True)
(root/'validation_summary.json').write_text(json.dumps(dict(release=2,fixtures_pass=True,profile_audit=json.loads((out/'reference_audit.json').read_text()),generation_checks=checks),indent=2));seal()
subprocess.run([sys.executable,'-B',str(root/'pipeline.py'),'verify'],check=True)
