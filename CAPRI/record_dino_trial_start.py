from pathlib import Path
import sys
root=Path('E:/pranav/Final sem/sourcefree_synth')
for file,text in {
 'STYLE_PIPELINE_PLAN.md':'LATEST USER OVERRIDE: remove character-count inference from active pipeline. Phase1 CV+frozen style features; Phase2 font adjustment; optionalPhase3 now testing frozen DINO feature/attention-guided noncolor parameter search. Fixed5 characters is a generic preview setting, not a family count estimate. Automation remains paused.',
 'STYLE_DECISIONS.md':'D072 — User drops unreliable length module and authorizes optional DINO parameter adjustment. Predeclared15 candidates (baseline and one-control low/high variants); fit first4 references and report next4, Pymlex2/2. All previously exposed development, not unseen. Frozen DINO CLS + attention-weighted patch means + spatial attention occupancy guide selection; CLS-only ablation isolates selection effect. No model training; baseline eligible. Exact glyph/drawing RGB, palettes/gradient, font/effect/text/count locked, failed visibility rejected. Historical scripts/artifacts preserved. Local trial dino_recipe_01 running; no promotion.',
}.items():
    with (root/'docs'/file).open('a',encoding='utf8') as f:f.write('\n\n'+text+'\n')
sys.path.insert(0,str(root));from style_pipeline.monitor import update
update('dino_recipe_search','Optional DINO parameter search running; count disabled, colors/fonts locked, automation paused.',running=True,run='C:/Users/pranav/Documents/ChatGPT/thesis/dino_recipe_01')
