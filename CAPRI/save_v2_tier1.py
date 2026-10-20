from pathlib import Path
import json,shutil,sys
import numpy as np
HERE=Path(__file__).parent;ROOT=Path('E:/pranav/Final sem/sourcefree_synth');source=HERE/'v2_tier1_02';results=json.loads((source/'results.json').read_text())
rows=[]
for model,r in results.items():
    rows.append(f"| {model} | {np.mean(r['test']['f1']):.3f} | {np.mean(r['heldout']['f1']):.3f} | {np.mean(r['test']['visible_mask_iou']):.3f} |")
report='''# V2 Tier1 initial experiments: not a replacement model

User scope confirmed: five core real cohorts(CryptBB,City,Courier,DarkBay,Outlined); nine independent font families. Tier2 expansion/Tier3 giganticcorpus deferred. No PARSeqtraining, failedPhase3restart or modelpromotion. Real families are inference cohorts; classifiers train on synthetic pixels only, never family labels or target generator assets.

## Implemented

Original synthetic generator with straightlines,sinecurves,dots,rectangles and dense single-pixel grain; combinations/clean examples; text as nuisance with six training fonts,one selection font,two heldout fonts. Full masks and visible compositor-support masks stored separately. Glyph occlusion and later effect occlusion are represented. Counts,widths,full/visible coverage stored. These are compositor-support annotations, not a perceptual contrast audit; same-color effects can be hard to see. Grain is a limited procedural texture, not arbitrary texture understanding. Text nuisance can overlap/crop; this is not a valid OCRcorpus.

Run01:1200train+240eachselection/test/heldout,600steps each whole-image/multi-view; collapsed largely to negative predictions, zero maskIoU at0.5. Failed and preserved. Run02corrected batchnormalization, positive weighting, increased classification contribution, higher-resolution32x64location outputs and added single10x3ablation.2400train+420eachselection/test/heldout=3660images,1500steps/model. Fresh disjoint seed ranges from run01. Full-resolution input/fullmask/visiblemask arrays saved in compressed supervision files; training mask targets downsampled32x64. No new real data read to choose checkpoints.

Multiview:63actual resized crops from1x1,2x1,3x1,1x2,2x2,3x2,5x3,10x3(cols×rows). SharedCNN produces crop tokens. Learned position-aware tanhattention pooling combines tokens; full-image spatial branch predicts location. This is not a transformer and not a full pairwise relational model. Small crops resized48x24; aliasing/tiny structure loss remains a limit. Whole-image/single10x3/multiview same stepbudget, not matchedcompute. Checkpoint selected by synthetic selection compositeBCE+dice, fixed0.5presence threshold. Whole-image and singlegrid still have full-image spatial mask branch; ablation concerns aggregation branch only.

## Measured results

| Model | Same-recipe macroF1 | Heldout macroF1 | Same-recipe meanmaskIoU |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Multiview same-recipe F1:line0.558,curve0.672,dot0.580,rectangle0.691,grain0.991. Corresponding false-positive rates:0.472,0.414,0.516,0.225,0.000. Visible-maskIoU:0.063,0.205,0.020,0.443,0.724. Grain is easy under this generator; these scores do NOT establish robustness or real transfer. Multi-view does not clearly beat baseline. No presence/localization gatepass. Heldout changes both fonts and curve recipe(Bezier vs sine); factors confounded, not isolated fontgeneralization.

Annotation audit passes full-vs-visible containment,finite pixels,seed disjointness and cross-partition pixel uniqueness; every crop stays with parent. All five feature classes have positive and negative examples per split. Fonts/hash audit passes, exact nine listed in fonts.json. Fontbank:IBMPlexMono,BalsamiqSans,PixelifySans,IBMPlexSerif,Bitter,Oswald,Anton,Lora,Caveat. Samples/selection fonts are intentionally distinct. Real inference20previously exposed fitimages(4per5families) after checkpointfreeze only; no realaccuracy and no renderingintegration. Gallery probabilities are exploratory, not calibratedconfidence.

## Work still required within Tier1

This is scaffolding/diagnosis, not the stronger finished classifier user requested. Increase independent synthetic renderrecipes, realistic nuisances and clean/glyph hardnegatives; add lowcontrast visibility checks. Resolve sparse-line/dot localization and high false positives before scaling blindly. Test higher-resolution multiscale spatial decoder and relational aggregation, compare selectedviews/fullviews with matchedcompute. Add separate density/geometry/textureheads only after defensible targets/annotation tests. Finegrain density/count definitions remain perfeature, not one genericdensity. Train calibrated uncertainty on separate synthetic calibration. Original pending audited realfeature review remains pending; do not silently invent groundtruth or claim success. Tier2/Tier3remain outofscope.

The earlier functional Phase1+Phase2 release remains default. No oldclassifier overwritten. No paper reproduction/newmethod claim: generator,position-aware weighted pooling and multitaskloss here are original engineering choices; copied infrastructure remains attributable to prior project work. Exact scripts/checkpoints/masks/seeds/results preserved.
'''
(source/'REPORT.md').write_text(report,encoding='utf8')
for name in ('v2_tier1_01','v2_tier1_02'):
    dest=ROOT/'runs'/('style_'+name);assert not dest.exists();shutil.copytree(HERE/name,dest)
(ROOT/'docs/V2_TIER1_INITIAL_RESULTS.md').write_text(report,encoding='utf8')
with (ROOT/'docs/V2_TIER1_PLAN.md').open('a',encoding='utf8') as f:f.write('\n\nExecutionupdate: pilots01/02 COMPLETE, both NOT acceptance. Run02trains3models on3660syntheticimages,32x64location maps,balancedpositive losses; allmask/seed/split audits pass but falsepositives remain high. Read V2_TIER1_INITIAL_RESULTS.md. Density/geometryheads stilldeferred. No activejob, no promotion.\n')
for file,text in {'STYLE_DECISIONS.md':'D079 — Tier1pilot01 failed(nearlyallnegative/zero maskIoU). Retained failure; run02 addsbatchnorm,weighted positives,classificationweight3,32x64masks,single-grid ablation and freshseeds. Full/visible masks stored and audited. Run02multiview meanF1about0.70 but line/dotFPR~0.47/0.52; no clear multiviewgain, no gatepass. Next focus semantic/visibility/decoder/recipe quality before moredata. No classifierreplacement.', 'STYLE_HANDOFF.md':'LATEST COMPLETE: V2Tier1initial pilots01/02 saved runs/style_v2_tier1_*. Read docs/V2_TIER1_INITIAL_RESULTS.md.5families9fonts exactscope, tiers2/3deferred. Run02presence/location modelsNOTready(highFPR,weak sparsemasks). All trainingcompleted, noactivejob. Pipeline release02unchanged; no PARSeqtraining, no Phase3, automationPAUSED. Density/geometryheads notyetimplemented.', 'THESIS_PAPERS.md':'V2Tier1initial pilots01/02 use original procedural mask supervision, sharedCNN cropencoder,position-aware tanhattentionpooling and BCE+dice multitasktraining. No newpaper directly adopted and no transformer/MILpaper reproduction claimed. Full/visible support masks and failureoutcomes documented in V2_TIER1_INITIAL_RESULTS.md. These are pilotmodels, not a robust upgrade or source of thesis success claims.'}.items():
    with (ROOT/'docs'/file).open('a',encoding='utf8') as f:f.write('\n\n'+text+'\n')
sys.path.insert(0,str(ROOT));from style_pipeline.monitor import update
update('v2_tier1_initial_tests_complete','V2Tier1 initial corpus/model comparisons complete; high false positives, no promotion.5families9fonts; no job running.',running=False,run=str(ROOT/'runs/style_v2_tier1_02'))
print('SAVED')
