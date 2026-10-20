from pathlib import Path
root=Path('E:/pranav/Final sem/sourcefree_synth')
local=Path(__file__).resolve().parent
p=root/'docs/STYLE_PIPELINE_PLAN.md'
p.write_text((local/'pipeline_sequence_header.md').read_text(encoding='utf-8')+p.read_text(encoding='utf-8'),encoding='utf-8')
(root/'scripts/run_style_fonts_direct.py').write_bytes((local/'run_style_fonts_direct.py').read_bytes())
for name,text in {
'STYLE_DECISIONS.md':'D064: User changes sequence: Phase1 unchanged, Phase2 fonts (formerly3), optional Phase3 distribution refinement (formerly2) deferred pending assessment. New direct font runner bypasses all old recipe adjustments. Preserve old runs and numbering as historical. No claim old font renders followed new order.',
'STYLE_HANDOFF.md':'LATEST USER OVERRIDE: Phase1 unchanged -> Phase2 fonts directly -> assess whether optional Phase3 distribution refinement needed. Do NOT apply old Phase2 recipes/refiners by default. Use scripts/run_style_fonts_direct.py for new training/search runs, prefix style_phase2_fonts_. Show direct Phase1+font results across families before discussing optional refinement. Historical font scripts inherit old recipes and must not be mistaken for direct pipeline. Continuous loop now restricted to first two phases until assessment.',
'THESIS_PAPERS.md':'Pipeline order revised by user: DeepFont-inspired adjustment is now Phase2. Prior recipe/CNN/transformer/GAN work is historical and optional Phase3, deferred pending assessment. Existing paper attribution and failed results remain unchanged.'
}.items():
    with (root/'docs'/name).open('a',encoding='utf-8') as f:f.write('\n\n'+text+'\n')
