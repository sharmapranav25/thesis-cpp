from pathlib import Path
import sys
root=Path('E:/pranav/Final sem/sourcefree_synth')
plan='''# V2 Tier1 scope and first milestone

User authorizes starting stronger synthetic-trained multi-view feature models. Tier1 is FIVE real family cohorts (CryptBB,City,Courier,DarkBay,Outlined) and NINE independent fontfamilies. These five are real inference cohorts, not classifier training labels or family-specific generator libraries. Tier2 expansion and gigantic Tier3 corpus are deferred. No PARSeqtraining or failedPhase3 restart.

Fontbank chosen deterministically as first9 audited independent fonts: IBM PlexMono,BalsamiqSans,PixelifySans,IBMPlexSerif,Bitter,Oswald,Anton,Lora,Caveat (verify run fonts.json for exact names/hashes).6training fonts,1selectionfont,2heldoutfonts. Real labels never enter fitting/selection.

First pilot: original synthetic generator;1200train+240selection+240same-recipe test+240heldoutfonts/Bezier-curves. Five supervised featureheads: straightlines,curves,dots,rectangles,grain. Presence plus visible compositor-support localization; grain here means dense single-pixel scatter, NOT comprehensive texture understanding. Full/visible coverage and generated counts/widths recorded for later density/geometry heads. Those heads are not yet implemented.

SharedCNN encoder receives63 actual resized crops from grids(cols,rows):1x1,2x1,3x1,1x2,2x2,3x2,5x3,10x3. Learned coordinate-aware attention pooling combines regiontokens; fullimage spatial branch predicts five coarse8x16masks. This is an original attention-pooling prototype, not a transformer or complete cross-block relational model. Whole-image-only model trained under same budget for initialablation.600steps/model, best synthetic selection composite BCE+dice checkpoint, fixed0.5presence threshold. Same-parentviews never cross splits. Heldout test changes fonts and curve recipe simultaneously, so cannot isolate their effects.

Gates: generator/annotation sanity first; synthetic presenceF1,false-positive-rate,visiblemaskIoU and ablation next; only then frozen real inference on5families, qualitative predictions only. This pilot is NOT robust V2 acceptance and no rendererintegration occurs automatically. Required next: more independent synthetic rendering recipes, label-visibility audit, clean/glyph hardnegative suite, gridablation against single10x3, density/geometryheads and calibrated uncertainty. Do not treat this initial1200image corpus as the promised robust bank. Keep releasephase12_v2 unchanged until validated replacement.
'''
(root/'docs/V2_TIER1_PLAN.md').write_text(plan,encoding='utf8')
for file,text in {'STYLE_DECISIONS.md':'D078 — User starts V2 Tier1:5families9fonts, tiers2/3deferred. Firstmilestone presence/localization syntheticpilot + wholeimage ablation;63actualcrops/8grids with sharedCNN+positionawarelearnedpooling. No realtraining, no VLM, no classifier promotion. Exact scope/gates in V2_TIER1_PLAN.md.', 'STYLE_HANDOFF.md':'LATEST ACTIVE: V2 Tier1 authorized5families9fontfamilies. Local start_v2_tier1.py ->v2_tier1_01. Inspect job before duplicate. Presence/location pilot only, no density/geometryheads yet. Default generationrelease unchanged. Tiers2/3deferred; PARSeqnottraining; continuousautomation remainsPAUSED.'}.items():
    with (root/'docs'/file).open('a',encoding='utf8') as f:f.write('\n\n'+text+'\n')
sys.path.insert(0,str(root));from style_pipeline.monitor import update
update('v2_tier1_pilot_training','V2 Tier1:5family inference cohorts,9fonts; synthetic presence/location pilot and multi-view ablation running.',running=True,run='C:/Users/pranav/Documents/ChatGPT/thesis/v2_tier1_01')
