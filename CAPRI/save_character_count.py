from pathlib import Path
import shutil,json
root=Path('E:/pranav/Final sem/sourcefree_synth');local=Path(__file__).resolve().parent;target=root/'runs/style_character_count_01'
if target.exists():raise RuntimeError('Preserve historical output')
shutil.copytree(local/'character_count_02',target)
script=(local/'train_character_count.py').read_text(encoding='utf-8').replace("Path(__file__).resolve().parent/'character_count_02'","ROOT/'runs/style_character_count_01'")
(root/'scripts/train_character_count.py').write_text(script,encoding='utf-8')
header='''# Active pipeline: Phase3 character count (user override)

Phase1 remains visual extraction/mapping with CV and frozen learned features. Phase2 is font adjustment directly. Phase3 now predicts CHARACTER COUNT ONLY: no text recognition or character-type/alphabet prediction. Train on synthetic images only; real images inference only. Prior color-locked refinement remains optional and OUTSIDE the active pipeline. Continuous automation remains paused at user request; this count-training request does not resume it.

Count pilot supports2-8 characters. Existing character strings remain configurable. Do not automatically change generation lengths from unverified real predictions. See STYLE_CHARACTER_COUNT_01.md.

'''
p=root/'docs/STYLE_PIPELINE_PLAN.md';p.write_text(header+p.read_text(encoding='utf-8'),encoding='utf-8')
report='''# Synthetic-only character-count pilot01

Task: predict count2-8 only. No OCR, alphabet prediction, VLM, real labels or real training. Original small CNN trained locally onRTX A4000. Independent licensed12fontbank: first10training, last2withheldfonttest.1792training images;448selection,448calibration,448ordinarytest,448withheldfonttest. Random text,length,position,spacing,rotation,outline,noise,lines,blur and polarity. Seedsdisjoint; shortstrings may overlap acrosspartitions, explicitlynotclaimed globallydisjoint. Whole-image gray+edge encoding; no croppedfontviews thatwouldlosecharacters.

1200steps,batch64,AdamW; bestcheckpoint500 selected by minimumsyntheticselectioncrossentropy. Calibrationtemperature1.45; exploratory confidencecutoff.5 chosen onsyntheticcalibration to reach90%empiricalprecision with>=30support. In-sample calibration is not realconfidenceguarantee.

| Test | Exactcount | MAE | Confidentsubset accuracy/coverage |
|---|---:|---:|---:|
| Ordinary synthetic448 |89.29%|.161|92.74%,413/448|
| Withheldfonts448 |93.53%|.085|95.54%,426/448|

Hardercounts6-8 produce mosterrors. Chanceclassaccuracy14.29%. Withheldfont result can be easier thanordinarytest; not evidence of general realaccuracy. Syntheticgenerator constrains widths/layout bycount, a possible shortcut to investigate with independent layouts/generators. Rotationclipping caughtbeforetraining; glyphs scaled tofit, failedlocalattemptretained. All countlabels equal generatedstringlength; seedpartitionsdisjoint; checkpointselection validated; probabilitynormalization passes.

Frozeninference on212 existingfit/reference images across8sources. CV glyphproposalcount also recorded as diagnostic; no CV or labels fitmodel. Modalpredictions: CryptBB5,City5,Courier6,DarkBay6,Outlined6,Roncuevas7,Pymlex4,Sloth6(tie6/7 resolvedlower). These are predictions NOT verifiedfamilylengths. Assistant inspected32examples: City/Pymlex plausible, Courier/Roncuevas oftenovercount, outlined andSloth errors too. Syntheticconfidence can behighonwrongrealcounts. No realaccuracymeasured, no runtimegenerationintegration or automaticcountpromotion.

Model and per-imageprobabilities saved; no priorcheckpoints modified. Nextvalidation should target densecurves/outlines/noise and count-preserving layout shifts before relying on familylengthdistribution. Count range2-8 cannot detect unsupportedlengths; no claim it can. Priorrefinement remainsoptionaloutsidepipeline, automationpaused.

No newpaperadopted: original smallCNN classifier and syntheticaugmentation with existingfont/descriptor utilities. Do not cite DeepFont as a character-count implementation or claim a publishedmethod reproduction.
'''
(root/'docs/STYLE_CHARACTER_COUNT_01.md').write_text(report,encoding='utf-8')
for name,entry in {
'STYLE_DECISIONS.md':'D069: User defines activePhase3 as countonly, optionalrefinement outsidepipeline; automationpaused. TrainedoriginalsyntheticCNN2-8, no realtraining/OCR/type/VLM. Beststep500: synthetic89.29%,withheldfonts93.53%.212realinferences mixed, Courier/Roncuevas overcount; no generationpromotion. Split/label/checkpoint checks pass. See STYLE_CHARACTER_COUNT_01.md; countsupport limited2-8, possiblesyntheticlayoutshortcuts. Failedclipping fixture caughtbeforetraining and fixed.',
'STYLE_HANDOFF.md':'LATEST USER OVERRIDE: activepipelinePhase1visual,Phase2fonts,Phase3CHARACTERCOUNTONLY. Optionalrefinement outsidepipeline. LoopPAUSED, do notresume. Countpilot trained/tested/inferred runs/style_character_count_01, scripttrain_character_count.py; synthetic89.29%,withheldfonts93.53%; real212 mixed. No generationintegration/realaccuracy. Followup improves noise/layoutrobustness; do nottrust syntheticconfidence onreal. No activejob.',
'THESIS_PAPERS.md':'Character-count pilot01 is original smallCNN syntheticclassifier, no newpaperadopted. Existingaugmentationutilities reused; not DeepFont countmethod or OCRreproduction.89.29%synthetic/93.53%withheldfont, realtransfer mixed. See STYLE_CHARACTER_COUNT_01.md.'
}.items():
    with (root/'docs'/name).open('a',encoding='utf-8') as f:f.write('\n\n'+entry+'\n')
import sys
sys.path.insert(0,str(root))
from style_pipeline.monitor import update
update('Character count pilot complete','Synthetic-only count model trained/tested; real predictions mixed, not integrated. Continuous loop remains paused.',run=str(target))
