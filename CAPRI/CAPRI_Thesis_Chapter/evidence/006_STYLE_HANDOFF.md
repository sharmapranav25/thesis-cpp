# Current handoff

Phase 1 active. Read STYLE_PIPELINE_PLAN.md and append STYLE_DECISIONS.md. New canonical work is E:/pranav/Final sem/sourcefree_synth/style_pipeline; previous feature-bank staging is C:/Users/pranav/Documents/ChatGPT/thesis/feature_bank_work. Do not redo existing 28-minute classifier training without evidence. Old trained LR models exist in staging runs/feature_bank_exploratory_01, not necessarily canonical. They are exploratory, not reliable complete style detectors.

Latest user authorizes repeated retraining, overnight persistence, primary-paper research, modest assistant visual review, and Phase 2 comparison of CNN/transformer/GAN only after Phase 1 passes. No runtime VLM. No target-family generator libraries/fonts. Open source only. Existing native manifests are runs/pixel_statistics_01/{cryptbb,city,courier,darkbay}/split.json and runs/external_pixels_outlined_01/outlined_text_captcha/split.json. Reuse feature_bank.prepare.resolve_image for relocation. Excluded external eval pools stay excluded. Earlier complete/partial reference profiling has exposed all CryptBB and much City data; don't call it pristine held-out.

No Phase 1 gate pass yet. No Phase 2 run yet. Check active processes/status files before spawning jobs. All prior subprocesses for five-dataset/City profiling were stopped at user request and saved as interrupted; that stop did not cancel the later expanded pipeline authorization.

Active initial Phase 1 pilot: `scripts/run_style_phase1.py`, output `runs/style_phase1_01`, exec session 46752 at launch. Measurements: `style_pipeline/measure.py`, generic renderer: `style_pipeline/render.py`. Three new tests passed. Review gate.json, failures.json, results.json and all five galleries when done; append evidence to decisions log. Do NOT automatically declare pass or start Phase 2. Save revisions under new numbered runs. Existing 3-class learned detectors are intentionally not used as truth because of saturated real predictions.

Monitor: http://127.0.0.1:8765/monitor/index.html served by `scripts/serve_style_monitor.py`, session 34818 at launch. Update through `style_pipeline.monitor.update`. Confirm port/process before restarting. Automation id `continue-captcha-style-pipeline` ACTIVE, every 30 minutes, same chat. Local workstation and app must remain available for continued scheduling.

Update: run style_phase1_01 FINISHED; session 46752 no longer active. Gate FAILED: nearest-own-family 3/5; improved-vs-baseline 5/5; failures retained. Assistant reviewed all five galleries: CryptBB/City/DarkBay partial, Courier/outlined poor. See STYLE_PHASE1_01_REVIEW.md. Phase 2 remains blocked. Next substantive work is D013 generic segmentation/palette/gradient/periodicity/hollow-text cues and controlled fixtures, then a new numbered run. Do not just re-run unchanged code. Descriptor loss and visible-text contribution alone cannot pass the gate. No new model training required until evidence supports it.

Revision style_phase1_02 actually launched after four tests passed. Session 19076 at launch; inspect monitor/status.json and its gate/results before starting another run. D014 records palette/gradient changes and prior inaccurate start claim. Phase 2 remains gated.

LATEST OVERRIDE — 2026-09-28: User requested an additional downloaded CAPTCHA scheme BEFORE Phase 2. Phase 2 ON HOLD; no neural training has started. Run 05 completed 5/5 primary nearest-family diagnostics and all renderer checks; assistant described partial visual resemblance on 4/5, but its gate.json remains pending and the independent descriptor only separated 3/5. Do not treat prior spoken gate passage as robust validation. New source: data/external/roncuevas_captchas_01, 168 seeded samples from pinned MIT-declared repository roncuevas/Captchas. Test script scripts/test_unseen_style.py, immutable output runs/style_phase1_unseen_01; 128 style-reference images/40 separate evaluation images, unchanged run-05 code and search space, no labels, target fonts, generator code or retraining. Inspect process/results before relaunch. New decision entries D023-D024 govern this task. Prepared refiners.py and its test are architecture preparation only.

LATEST — frozen learned inference COMPLETE: runs/style_frozen_inference_01, scripts/infer_unseen_classifiers.py. All 168 images/5,040 blocks processed by historical synthetic-trained LR models; hashes unchanged. 32 classifier-guided renders, zero generation failures. Read docs/STYLE_FROZEN_INFERENCE_01.md. Saturated straight/curve predictions are unvalidated; don't claim real accuracy. Real data MUST NOT train or calibrate feature classifiers (explicit user clarification). Phase 2 remains on hold. Renderer now accepts optional classifier_features gating straight/curved/dot functions; no classifier flags retains previous CV behavior. Rotation containment implemented; count_evidence diagnostic exists but is NOT reliable or wired into generation. Five-character assumption remains unresolved. Do not rerun scripts with immutable existing run directories or overwrite history.

LATEST COMPLETE — all five previous families now use frozen classifier inference -> joint feature gating + shared CV measurements -> generation. Exact run-05 cohort: 168/family, total 840 images /25,200 blocks; 32 outputs/family (160), zero rendering failures. Runs style_frozen_{cryptbb,city,courier,outlined_text_captcha}_01 and style_frozen_darkbay_02. DarkBay _01 was interrupted by monitor file lock; preserved. Dashboard now retries transient locks; simulated recovery checked. Report docs/STYLE_FROZEN_FIVE_01.md and runs/style_frozen_five_report_01. All model hashes unchanged; no real-data training. Strong line/curve saturation remains a domain-shift concern, not a success measure. Prepared Phase 2 remains on hold. User asked whether CV-only generation was assistant-added: yes, disclosed that distinction; do not describe old CV-only generation as classifier-guided.

LATEST ACTIVE — user explicitly authorized Phase 2 after reviewing limitations (D033), superseding earlier hold; do not falsify Phase 1 gate history. Pilot actually launched: scripts/run_style_phase2.py -> runs/style_phase2_pilot_01, console runs/style_phase2_pilot_01_console.log, exec session 67277, Python process 22864 at verification. Nine preflight tests passed. Read STYLE_PHASE2_PROTOCOL_01.md and D033-D035. Current run executes recipe search/distillation, discriminator recipe ranking, then direct CNN/transformer/GAN refinement, all from scratch. Five original families; 96 historical fit profiles for fitting, last 32 development; final hash-sorted historical validation window 80:120 held for reporting only after selection frozen. Existing real pools had past exposure, not pristine. Check active process and output before launching anything; no duplicate. User asked for status during preparation and was told no results yet. Maintain monitor and inspect all five comparison galleries plus comparison.json before claiming improvement. No trained feature-classifier weights or thresholds may change. Renderer supports bounded optional recipe controls; zero parameters preserve default behavior. Phase 2 models may learn from unlabeled real fit images, as explicitly authorized.

LATEST COMPLETE — Phase 2 pilot 01 has FINISHED (session 67277), not still training. Read docs/STYLE_PHASE2_PILOT_01.md and runs/style_phase2_pilot_01/{comparison,visual_review,status}.json. Seven methods plus baseline, 1,280 final renders, no render failures, unchanged feature classifiers. Recipe approaches improve measured mean distance 32–37%, 3/5 families; direct refiners improve 61–63% numerically but rejected visually due to text halos and broad tone shifts. All five galleries inspected (first four seeds per method). No full Phase 2 success or research completion claim. Final historical validation window 80:120 is now exposed reporting data and MUST NOT train/select further models or be relabeled untouched. Subsequent work must use only fit/development pools, preserve this initial comparison, and predeclare an independent evaluation before further claims. Monitor should say pilot complete/direct artifacts, not active training. User's status question received evidence-based intermediate updates.

LATEST ACTIVE — Phase 3 font matcher run 01 is trained/tested (docs/STYLE_PHASE3_FONT_01.md): 50.52% synthetic top1, 81.64% top3, 92.19% hollow; 160 real-development renders, outlined font effect still fails. Run 02 actually launched: scripts/run_style_phase3.py --run style_phase3_font_02 --steps 2000 --test-seed 30000 --warmstart runs/style_phase3_font_01/font_matcher.pt, exec session 83606. Check process/log before any duplicate. Synthetic-only model training; real fit96/development32 inference/search, no Phase 2 reserve access. docs/THESIS_PAPERS.md is the curated thesis paper register user explicitly requested; update actual usage and outcomes, not merely citations. No original classifier retraining. User asks continuous research/iteration and a new downloaded family each loop; maintain honest frozen tests and keep CV PLUS learned inference in every generated family pipeline. No stop request.

LATEST — Phase 3 runs 01 and 02 are COMPLETE; sessions 53923/83606 ended. Run 02 synthetic-only font model selected step2000; matched fresh synthetic test top1 56.25%, top3 84.77%, hollow90.23%, better than run01 on that same set. Both ran original CV + frozen feature inference as foundation and produced 160 development outputs each. Real fonts remain unreliable; outlined fails. Newly downloaded pymlex_preview_01 has only 12 published clean previews, not a representative sample; frozen CV+feature+recipe+font pipeline test runs/style_phase3_new_family_01 completed48outputs, incorrectPixelifyfont, visualsmoketestfails. No labels/model/generator downloaded. Read docs/STYLE_PHASE3_FONT_02.md and docs/THESIS_PAPERS.md (explicitly user-requested maintained paper register). All original feature classifiers unchanged. Further work: investigate synthetic noise/aliasing sensitivity and robust font aggregation on fitting/development evidence, not Phase2reservedwindow. Preserve completed results. User wants continuous loops and new-family tests, not automatic success.

LATEST — Run style_phase3_font_03 COMPLETE; session 79828 ended, no font training remains active. Synthetic robustness matched test improves top1 45.96% -> 54.56%, top3 74.09% -> 84.24%. 160 development renders, zero failures, outlined still wrong; no Phase 3 pass. Read STYLE_PHASE3_FONT_03.md and THESIS_PAPERS.md. Run03 has not yet had a new independent-family evaluation. Continue authorized research loops, both CV and learned inference, preserving histories and Phase2 reserve. Investigate outline/segmentation mismatch before more blind training.

LATEST — Continuous three-phase heartbeat updated ACTIVE every30min, no duplicate. Outline diagnostic COMPLETE (session8711): synthetic hollow .836 vs outlined real fit .250; MSER median0, City hollow .472. See D045 and runs/style_phase3_outline_diagnostic_01. Next: synthetic outline/contrast/scale ablations and region extraction, then revised synthetic training if supported. Inspect fit_diagnostic.png before visual conclusions. No training currently running. New-family test still owed for run03; source new independent family per complete loop. Maintain THESIS_PAPERS.md.

LATEST ACTIVE — Run style_phase3_font_04 actually launched, session56437, synthetic-only warmstart03 to4400, robust+contrast flags, testseed70000. Inspect process/log before duplicate. Contrast ablation completed: lowcontrast breaks font and hollow discrimination; see D046-D047 and runs/style_phase3_contrast_diagnostic_01/results.json. Two unittest font checks pass; pytest unavailable, no installation needed. Next after completion: inspect selected checkpoint and galleries, matched contrast-stratified test03vs04; report no real accuracy. New independent family still owed; preserve reserve.

LATEST COMPLETE — Run04 finished, no active training. Matched contrast comparison completed (session54994), results runs/style_phase3_contrast_comparison_01/results.json. Lowcontrast improves but standardcontrast regresses; realoutlined remainsfilled. Do not promote universalreplacement; see D048, STYLE_PHASE3_FONT_04.md. Continue outlinegeometry/scale or segmentation investigation and newfamily frozenCV+inference test; not another blindtraining. No reserveddata touched. Allpaper/outcome notes updated, current threephase authorization appended to PLAN.

LATEST ACTIVE — Geometry diagnostic completed: runs/style_phase3_geometry_diagnostic_01, D049-D050. Large outer outlines fail both frozen models; control grid inspected. Optional geometry synthetic argument now inner/outer/thin with height20–54, default preserved; two font tests passed. Run style_phase3_font_05 ACTUALLY RUNNING session3490: warmstart03 to4400, robust+geometry, test110000, no contrast flag. Check log/process first. Next inspect results and matched geometry/standard controls without selecting on reporting data; real fit/dev only. New independently sourced family evaluation still outstanding; prioritize after this revision, do not call completed loop. No reserveddata access or runtimeVLM. Papers updated.

LATEST COMPLETE — Run05 finished; new-source test02 finished (session91803),32slothimages/960blocks/48renders/0errors,unchangedmodels. VisualFAIL: extraneouscoloredlines, mixed5chars vsreference6digits, falsehollow; Phase2 worseclutter. See D051-D052. Next substantive loop shift back to Phase1 frozenfeature falsepositive gating and Phase2 recipe shift/fallback, preserving synthetic-only classifiers and realCV+inference. Sloth32 nowexposeddevelopment. Matched geometry03vs05 remainspending. Noactivejob exceptmonitor. Papers/report updated.

LATEST COMPLETE — Optional constraints.preserve_achromatic implemented, render achromatic_noise flag defaultsFalse; tests constraint2 and recipe2 pass. runs/style_color_constraint_01 paired16devimages removes color, excessive lines remain, blackfallback someletters. See STYLE_COLOR_CONSTRAINT_01.md,D053-D054. Noactivejob. Nextdiagnose frozen feature input/preprocessing/syntheticnegative behavior and density mapping; never calibrate onreal. Newfamily validation stillowed forconstraint; sloth exposed.

LATEST COMPLETE — Featureaudit01 completed(session4431). Replay48 exact, blankslow. Synthetic dot-only40points yields100%curve>.99;100points87.5%straight>.99 and100%curve>.99. Inference preprocessing/calibration not broken; descriptor textureconfounding. Next substantive step synthetic-only hardnegative classifier retraining with separate train/selection/calibration/test basegroups, ordinarysynthetic regression, freeze then realCV+inference. Noactivejobs beyondmonitor. See STYLE_FEATURE_AUDIT_01.md,D055-D056; originalmodels unchanged.

LATEST ACTIVE — Synthetic-only classifier revision launched session44644, scripttrain_feature_negatives.py, runs/style_feature_negatives_01. D057 protocol: straight/curve LR originaltrain+512dotnegatives,3C candidates, ordinaryselectionAP regressionlimit.02, separate128selection/calibration/test newseeds. No realimages. Models savedwith thresholdsNone: MUST NOT promote until selection/testreview and synthetic-only abstentionthreshold determination. Dotmodel original unchanged. Inspect logs/processbeforeduplicate.

LATEST COMPLETE — session44644 ended. style_feature_negatives_01 trained/tested. StraightselectiongatePASS,C1; negativeFP71.09->.78%, ordinaryAP.8832->.8709. CurveALLselectioncandidatesFAIL, savedfallback NOTaccepted. BoththresholdsNone, no runtimeintegration. Next synthetic-only thresholdcalibration forstraight and frozenCV+realdevinference; curve retrainingbalance investigation usingselection only, samegate,freshnegative tests. See D058,STYLE_FEATURE_NEGATIVES_01.md.

LATEST COMPLETE — session35068 ended. style_straight_revision_eval_01 calibratedsyntheticthresholds .546983/.785389; reranCV+frozenstraight inference16exposedslothimages. FAILEDTRANSFER:480/480positive,295linesbotharms,4pairsunchanged. No runtimepromotion. Next synthetictexture/antialiasing/scale plusCVlinegeometry investigation; originalcurve/dot unchanged. Docs/papers/monitor updated; noactivejob.

LATEST COMPLETE — line_support experiments01/02 FAILEDsyntheticgate (recall.6875/.59375,FP0). No realablation, no runtimeenablement. New optional straight_line_cap onlyappliesifexplicitlypassed; historicaldefaultpreserved. Need inspect missedpositivefixtures/proposalcoverage before changes; supportcode experimental. Noactivejobs. Useraskedstatus and got honestpartialresults.
\nLATEST: User background audit COMPLETE, runs/style_background_regression_01, docs/STYLE_BACKGROUND_REGRESSION_01.md. Font improvements retained; recipe background amplification identified for three families. Rollback not uniformly better; shared RNG must be separated for exact glyph preservation. No active training. Canonical writes currently need sandbox approval.


LATEST USER OVERRIDE: Phase1 unchanged -> Phase2 fonts directly -> assess whether optional Phase3 distribution refinement needed. Do NOT apply old Phase2 recipes/refiners by default. Use scripts/run_style_fonts_direct.py for new training/search runs, prefix style_phase2_fonts_. Show direct Phase1+font results across families before discussing optional refinement. Historical font scripts inherit old recipes and must not be mistaken for direct pipeline. Continuous loop now restricted to first two phases until assessment.


LATEST: All8source directPhase1+fonts gallery complete runs/style_direct_bank_01,32renders, no oldrecipe/refiner. Showed bank_1.png and bank_2.png inchat. Fontselection freshmeanposterior (not oldrecipe-dependent fontsearch); typography needs new-order validation. Phase3 stilldeferred. Noactivejob. See STYLE_DIRECT_BANK_01.md.


LATEST USER OVERRIDE: Phase3 trial authorized with NO color changes. Completed32renders/8sources runs/style_phase3_color_locked_01. Fonts/effects retained from directbank; noncolorrecipecontrols retained, colorslocked explicitly. No training or globalpromotion; Slothworseclutter, Citythickerlines. AnyfuturePhase3 must preservecolors; never use unrestrictedpixelrefiners thataltercolor. See report. Experimentalrenderer snapshot only, mainrenderer unchanged.


LATEST COMPLETE: colorlocked Phase3 clutter ablation01,20diagnosticrenders,4baseline exactreplays. Linecount/width amplify clutter, dots actually reduced. Usercontrols remain unchanged. No activejob; saved report. Continue respecting colorlocks and userothercontrols constraint; do not silentlypromote an ablation. No newpaper.


LATEST: userparameteraudit complete runs/style_phase3_parameter_audit_01,320renders. Report listsfamilyconditionalcandidates, no globalwinner. Citylinewidth, Couriercurvecount,Pymlexheight tentative; others weak/noeffect. No runtimechange. Colorlocks preserved; check reportlimitations (4exposeddevsamples,currentvalues,reusedmetric,RNGcoupling). Need broadercandidatevalidation notblindcombinedrecipe. Noactivejob.


LATEST USER OVERRIDE: activepipelinePhase1visual,Phase2fonts,Phase3CHARACTERCOUNTONLY. Optionalrefinement outsidepipeline. LoopPAUSED, do notresume. Countpilot trained/tested/inferred runs/style_character_count_01, scripttrain_character_count.py; synthetic89.29%,withheldfonts93.53%; real212 mixed. No generationintegration/realaccuracy. Followup improves noise/layoutrobustness; do nottrust syntheticconfidence onreal. No activejob.


LATEST: count heatmap comparison COMPLETE, runs/style_count_heatmaps_01; DINO/CV/density all fail to replace existing experimental countCNN. Read STYLE_COUNT_HEATMAPS_01.md. Real212 inference only, no real labels or accuracy. No active training, automation PAUSED. Active Phase3 remains count-only, optional refinement outsidepipeline.


LATEST COMPLETE: runs/style_count_font_bank_01 tests revised Phase1(count included)->Phase2 fonts across8sources,32renders. No optional refinement. Read STYLE_COUNT_FONT_BANK_01.md. Countoverestimates Courier/Roncuevas/outlined; no reliability gatepass. All render integritychecks pass; noactivejob. AutomationPAUSED.


LATEST COMPLETE: user DROPPED count module. DefaultPhase1 CV+frozenfeatures ->Phase2fonts; optionalDINO parametertrial complete runs/style_dino_recipe_01.30report renders, colors/fonts/textlocked, fixed5. Weak score improvements/mixedCV, no promotion. Noactivejob, automationPAUSED. See STYLE_DINO_RECIPE_01.md; historicalcount preserved.


LATEST ACTIVE: separate own-family GAN trial family_phase3b_01 at writable workspace. User rejected shared-family approach. Own real and synthetic fit inputs only, validation checkpoints and early stopping. Local script run_family3b.py. Automation staysPAUSED. Prior sharedrun saved runs/style_direct_phase3b_01. Do not launch duplicates.


LATEST USER SCOPE: this session prepares generation pipeline and training guide only, NO PARSeq training. Completed portable releases/phase12_v1; start docs/PHASE12_START_HERE.md.336allfamily samples+freshinference+byteparity+relocation+leaknegative+determinism checks pass. Models/assets/fonts bundled.21generic cells, trainvaltest manifests, no countinference/Phase3. Prior GANtask stopped and failed by user, noactivejob. AutomationPAUSED. Next session trains PARSeq onefamilyatime after corpus/split audits.


LATEST FIX: use releases/phase12_v2 and docs/PHASE12_START_HERE.md. V1backgroundblock heuristic caused Courier hallucinatedrectangles. CorrectedCV evidence and refreshedprofiles596; 168samples/21cells per8families pass. Courier0/96blocktriggers. Conservative detector canmissrealblocks; no overallqualitypass. No training, automationPAUSED.


LATEST ACTIVE: V2 Tier1 authorized5families9fontfamilies. Local start_v2_tier1.py ->v2_tier1_01. Inspect job before duplicate. Presence/location pilot only, no density/geometryheads yet. Default generationrelease unchanged. Tiers2/3deferred; PARSeqnottraining; continuousautomation remainsPAUSED.


LATEST COMPLETE: V2Tier1initial pilots01/02 saved runs/style_v2_tier1_*. Read docs/V2_TIER1_INITIAL_RESULTS.md.5families9fonts exactscope, tiers2/3deferred. Run02presence/location modelsNOTready(highFPR,weak sparsemasks). All trainingcompleted, noactivejob. Pipeline release02unchanged; no PARSeqtraining, no Phase3, automationPAUSED. Density/geometryheads notyetimplemented.


D080: User requested fixing known weaknesses rather than stopping at diagnosis. Revision03 tests a native-resolution detail decoder with stride2 and stride8 context plus spatial evidence for presence. Reason:02 reduced sparse features to8x16 then upsampled. First isolate architecture on same synthetic corpus; previously examined test/heldout now development, not unseen acceptance. 1800steps/model; whole/single/multi comparisons. Five real cohorts/nine fonts retained; no real training, no PARSeq, no Phase3. Further generator hard-negative/contrast work follows this diagnostic.


D080 outcome: revision03 and fresh validation complete; V2_TIER1_REVISION03.md. Fresh multiview dot F1 .587->.860, rectangle .585->.828; false positives .575->.028 and .252->.016. Common-resolution dot IoU .018->.211. Lines/curves still high FPR. Separate calibration rejected (poor recall/shift robustness). No promotion, no active training. Preserved02/03, five cohorts/nine fonts. Next generator visibility/hard negatives and multi-font calibration; no PARSeq or Phase3. Automation remains paused.


D081: User authorized grid specialists. Implemented eight per-grid heads with shared encoder, explicit synthetic crop supervision, coordinate/feature-aware within-grid pooling and across-grid pooling. Matched shared-local-head control isolates per-grid specialization from auxiliary/fusion changes. 1800steps each, selection synthetic only; fresh51m/52m/53m evaluation after freeze. Existing five families/nine fonts unchanged. Training active locally; no duplicate job, no pipeline promotion.


D081 outcome: grid-specialist experiment04 completed. Eight independently parameterized local heads plus learned hierarchical aggregation implemented; shared-local-head matched control and original03 evaluated. Fresh macroF1 specialists .822/.814/.813 versus shared-local .815/.819/.825 and original03 .831/.817/.807 (same/font/recipe). Mixed, no consistent specialist win, no promotion. Results V2_TIER1_GRID_SPECIALISTS_04.md. Jobs completed; automation remains paused.


D082: Debug curve regression before claiming cause. Found near-straight geometry in340/341eligible10x3single-curve crops, similar03/04curveAUC but higher04negative scores, and minimal direct score influence(mean0.00221when neutralized). Train matched04architecture interventions: no localaux vs remove onlyfine-grid curveaux. New61m/62m/63m evaluation afterfreeze. Training active, no duplicates or pipeline promotion.


D082 outcome: curve debug05 complete. Local ambiguity exists; neutralizing explicit curvehead scores barely affects final predictions. Removing all localaux improves fresh curveF1 .754->.793/FPR .456->.262, but curve-recipe generalization worsens; masking onlyfine curveaux helps modestly. Calibration trades falsepositives for misses, no fullfix. See V2_CURVE_DEBUG_05.md. No promotion/active job; automation paused.


D083: User authorized cross-block tilt/position reasoning. Implemented8local geometry heads, native axial angles/coherence,84neighbor relations with learned visible-connectivity proxy and gated messages. Train matched with/without neighbor aggregation, identical initialization and data; tests pass. Synthetic only; real inference afterfreeze,5cohorts9fonts. Training active; no pipeline promotion.


D083 outcome: cross-block geometry06 implemented/trained; tilt gate FAILED.96.98%valid-crop predictions nearhorizontal; MAE21.50 vs constant21.31. Neighbor curveF1 .751->.759 on same recipe but .675->.607 on curve shift, rectangle regression. No claim of learned tilt reasoning/no promotion. V2_STROKE_RELATIONS_06.md records fullfailure; no active training; automation paused.


D084: User requests independent per-feature training/pop/view selection. Added feature_specialists_v2, starting CURVES ONLY; others queued. CorpusG0passed20000images/10000pairedgroups, no duplicatepixels, disjointseeds,12orientationbins,9auditedfonts. Raw versus localcontrastpop for8grids then top2grid combination; selected configuration retrained3seeds; calibration separate;4lockedstrata unused untilfinal. Strict syntheticgates predeclared, no automaticpromotion. No shared5classclassifier or rushedrobustnessclaim. Original/vendor/defaultrelease unchanged. Finiteworker launch next; continuousautomation remains paused. Read V2_FEATURE_SPECIALIST_PLAN.md.


D084 launch verified: finite independent curve-specialist worker PID 26316 active; latest observed batch_progress screen_1x2_raw. Read feature_specialists_v2/runs/curves_v1/status.json/events.jsonl and monitor for live state. Whole-image raw/pop screening near chance; no learning-success claim. View search ongoing; do not duplicate worker. No real fitting, no pipeline promotion, other features queued, continuous automation still paused. Corpus20000images includes12000training images; remaining8000selection/calibration/locked evaluation.


D085: User authorizes concurrent independent specialist training. SampledRTXA4000headroom16GBtotal/1.6GBused/37%GPU. Add dot worker alongside existing curve PID26316; own20kpairedcorpus,seeds91m..97m,pop(top/bottom-hat),models,viewsearch,calibration/lockedtests. Dot means non-text roundscatter radius1..3, not arbitrarygrain. Tests4/4pass; fullcorpusgate runs beforetraining. Use monitor/specialists.html for per-job states; legacy singleton/status and old completion prose do not describe all workers. Initial concurrency capped2; no sharedweights/no curve modifications/no productionpromotion. Other specialists queued, automationpaused.


LATEST: independent curve-specialist finite search complete. Read V2_CURVE_SPECIALIST_RESULTS.md and feature_specialists_v2/runs/curves_v1/gate.json. No worker remaining; other feature specialists queued. Automation paused.


Curve v1 investigation: near-constant training predictions (~0.496; AP0.513 on 512 training examples), despite successful 16-image memorization control. Calibration tie chose0.99, explaining zero recall but not chance learning. See V2_CURVE_FAILURE_DIAGNOSIS.md for confirmed findings versus unproven training-design hypotheses. No production promotion; curve worker completed, dot worker remains independent. Monitor specialist status endpoints verified HTTP200 after narrow server allowlist fix. Further curriculum/localization ablations proposed, not yet run.


Started authorized finite two-stage stroke learnability trial: see V2_TWO_STAGE_STROKES.md. Reason: v1 curve classifier collapsed under whole-image supervision. Explicit masks teach localization; oracle versus predicted-mask shape tests isolate failure. Preserve dot worker, old runs, release, and paused automation. No robust success claimed.


User-requested10x3 raw RGB multicolour stroke retraining completed. Shared per-patch detector, stitched mask, no feature-pop, fresh synthetic seeds and font holdouts. See V2_STROKE_GRID_10X3.md for protocol/results. Failed diagnostic gate remains truthful; no production promotion or automatic loop. Prior full-dataset stroke inference exposed background/text mask contamination.


LATEST: independent dot-specialist finite search complete. Read V2_DOT_SPECIALIST_RESULTS.md and feature_specialists_v2/runs/dots_v1/gate.json. Dot worker completed; check curve-worker status separately. Automation paused.


User shifts focus to joint multi-label feature learning. New finite feature_multilabel_v1 run:15operational features, raw RGB, multiscale1x1/2x2/5x3/10x3 shared feature maps, independent feature attention/heads. Synthetic-only13400image corpus, counterfactual visibility masks for label validity, heldout fonts/combinations/recipes,3seeds, early stopping and per-feature gates. See V2_MULTILABEL_PLAN.md. No success/promotion claimed; old specialists and release preserved. Dot specialist completed but failed strict synthetic gate on cluster layout/rasterizer; do not call prior highAP a robustness pass. Automation remains paused.


Multi-label v1 finite run complete. See V2_MULTILABEL_RESULTS.md; passed features=['curved_interference', 'dot_scatter', 'periodic_grid', 'background_texture', 'background_gradient', 'background_blocks', 'blur']. No production promotion.


Multi-label real Phase1/2 experiment complete:6909 real images inferred,105 renders across fivefamilies. See V2_MULTILABEL_REAL_PHASE12.md and runs/style_multilabel_phase12_01. CachedCV/font profiles plus fresh15head inference; only7passedheads map compatible effects, straight legacy settings remain. Darkbay falsegrid68.7%, outlined curve0.1% show transfer failures; no promotion. City dot0% promising qualitative correction. Original release integrity verified. All outputs and failed serialization recovery documented; no real fitting.


User requests transformer instead of CNN. Launched finite feature_transformer_multilabel_v1: pure8x8patch transformer,15feature tokens, same train/selection/calibration corpus and optimizer/stopping rules,3seeds. Fresh304m..307m test images; frozenCNN replay on same fresh tests for comparison. Historical CNN preserved, no real fitting, no renderer-parameter extension yet. Read V2_TRANSFORMER_MULTILABEL_PLAN.md. No success/promotion claim; automation paused.


Transformer multi-label v1 finite run complete. See V2_TRANSFORMER_MULTILABEL_RESULTS.md; passed features=['periodic_grid', 'background_texture', 'background_blocks', 'blur']. No production promotion.


Transformer Phase1/2 test complete by user request:6909real predictions,105matched renders, seed733. Same seven mapping heads as CNN, explicitly exploratory including failedheads; no joint parameter training. City/Courier falsegrids remain, no clear visual improvement. See V2_TRANSFORMER_REAL_PHASE12.md and runs/style_transformer_phase12_01. Production unchanged.


Research started/completed initial assessment: see HYBRID_GRID_RESEARCH.md. GriddedTNP offers spatial latent-grid encoding; visualConformer is closer to requested bidirectional CNN-transformer interaction. Research references only, no new architecture trained or copied yet. Plan controlled one-way/bidirectional/grid-pooling ablations; do not infer success from papers.


User requests BOTH architectures and clarifies joint feature-to-library mapping. Implemented gridded pseudo-token and visualConformer-inspired bidirectional models with15feature heads plus9renderer-control heads, exact synthetic control tracing tested against20render replays. Train synthetic only, same data, fresh404m..407m tests,3seeds each. Finite workers include fullreal6909 inference and matched median-vs-learned render comparisons afterward. Do not call this fullGriddedTNP/Conformer reproduction or all-parameter solution. See V2_JOINT_MAPPING_PLAN.md. Oldrelease/history retained, no promotion/automation restart.


Seed731 preliminary testing authorized by user. Separate checkpoint copies in feature_joint_mapping_01/seed731_preview are SHA256-verified; synthetic calibration thresholds and four locked strata evaluated, then all6909 real images inferred and matched median-vs-learned controls rendered. Both original training jobs continue unchanged; no test-based tuning or seed selection. These locked results are now exposed for seed731, so preserve the precommitted training protocol for remaining seeds and do not reuse this test set for later architecture tuning as unseen. Snapshot evaluator: C:/Users/pranav/Documents/ChatGPT/thesis/seed_one_preview.py. Status: seed731_preview/status.json; logs in thesis workspace. No production promotion; real accuracy unverified. Evaluation started, completion/results pending.



Seed731 preview COMPLETED: both models inferred all6909 real images and rendered210 images each across fivefamilies. Each passed7/15 synthetic feature gates (curves,dots,grid,texture,gradient,blocks,blur). Gridded6/9 and hybrid7/9 controls beat training medians in all4 test strata; hybrid additionally passes curve amplitude. Both fail width and dotradius baseline comparisons. Assistant visual review of hybrid Courier/CryptBB/Darkbay: Courier sampled grids reduced relative to gridded, but CryptBB lacks dense clutter and Darkbay loses blocks/occasionally adds grid. No overall real-transfer pass. Matched seeds do NOT fully isolate parameter effects: changing primitive counts changes shared RNG consumption and downstream glyph colors, visible in paired images. Thus comparisons are end-to-end recipe outcomes, not clean causal controls-only ablations. A future renderer with independent RNG streams is needed; do not alter running precommitted jobs. Release integrity verified; training remains active seed732. Detailed results under feature_joint_mapping_01/seed731_preview/run_{gridded,conformer}/results.json; sheets under real_generation/<family>/comparison.png. No promotion.



V1/V2 fusion trial complete at feature_joint_mapping_01/seed731_cv_fusion_01. Reason: learned negative gates erased measured blocks/texture; narrow synthetic counts suppressed dense backgrounds. Preserve all measured palettes/gradients/blocks/line and dot density, retain Phase2 fonts; frozen seed731 curve presence/count/width contribute, hybrid amplitude used only after synthetic baseline pass. Do not add grids/blur from classification alone without measured support. Both architectures, fivefamilies,105 matched cases each, measured+fusion=420renders. Independent RNG streams isolate line/curve/dot/glyph randomness; glyph geometry and measured style preservation asserted, colour readability fallback remains possible. No training change, no production promotion, previously exposed real data only. Comparison includes historical V2 which retains old RNG; measured-vs-fused is the controlled pair. Release hash check passed.


CAPRI packaged as releases/CAPRI: portable runtime with seed731 checkpoints for gridded/hybrid, all5cached fused profiles, exact base release/fonts/licenses, fresh CV+frozen inference CLI, labeled generation2–8 in3buckets, comparison galleries, integrity manifests and limitations.231smoke images generated across both architectures/allfamilies plus fresh profiling2reused images and audit passed. No production promotion; other training jobs remain separate and must not replace snapshot silently. Read releases/CAPRI/NEXT_SESSION.md and README.md. Reproduction training scripts retain original paths and are explicitly reference-only. ZIP at C:/Users/pranav/Documents/ChatGPT/thesis/CAPRI-seed731-experimental.zip.


Joint mapping finite run complete: gridded. See per-run results; passed features=['dot_scatter', 'periodic_grid', 'background_texture', 'background_gradient', 'background_blocks', 'blur']. No production promotion.


Joint mapping finite run complete: conformer. See per-run results; passed features=['curved_interference', 'dot_scatter', 'periodic_grid', 'background_texture', 'background_gradient', 'background_blocks', 'blur']. No production promotion.


CAPRI Meta-Sim-inspired trial finished; runs/capri_metasim_01 and docs/CAPRI_METASIM_RESULTS.md. Five independent residual MLP adapters, frozenDINO GaussianMMD and finite-difference renderer gradients, immutable fonts/text/colours/layout;64paired report generations/family. Twelve steps, no classifier training or runtimeVLM. Historical development subsets, not unseen validation. No base CAPRI promotion/modification. See results and visual review; DINO-only improvement insufficient.


CAPRI_v2 packaged at releases/CAPRI_v2 by explicit user request: Phase2 weighted top3 frozen-font candidate sampling per image plus optional Meta-Sim-inspired family adapter. Default similar-font mode and refinement auto(on for5exact hybrid profiles). Single/off flags retain baselines. Gridded/new/changed profiles have no fitted adapter; auto visibly off, on rejects; no cross-family sharing.378generated smoke samples verify all5families use>=2fonts from pool, exact glyph geometry/RGB locking, deterministic replay, single-font/off and mismatched-adapter rejection. Results/retries/fallbacks in docs/FONT_REFINEMENT_VALIDATION.json in package. No OCR/real quality validation of combined variety+adapter policy; adapters were fitted on single-font generation. Prior CAPRI preserved; no classifier retraining. ZIP C:/Users/pranav/Documents/ChatGPT/thesis/CAPRI-v2-fonts-metasim.zip. Read new README/NEXT_SESSION.


User requested same100k corpus for allfive CAPRI families:500000total at E:/pranav/Final sem/synth_data. Darkbay existing active run reused. Otherfour queued2families at a time,6shard workers per family; stable separate seeds. Status CAPRI_BANK_STATUS.json; exact protocols per corpus. No training. Wait for per-family audit.json passed before using corpus.


Darkbay100k CAPRI_v2 dataset complete at E:\pranav\Final sem\synth_data\darkbay_capri_v2_100k. All100000decoded pixel hashes/unique pixels/labels/length-bucket counts/train-text partition audited. Similar top3font mixture + matched frozen adapter. See audit/protocol; no OCR training or real-performance claim.


city100k CAPRI_v2 dataset complete at E:\pranav\Final sem\synth_data\city_capri_v2_100k. All100000decoded pixel hashes/unique pixels/labels/length-bucket counts/train-text partition audited. Similar top3font mixture + matched frozen adapter. See audit/protocol; no OCR training or real-performance claim.


outlined_text_captcha100k CAPRI_v2 dataset complete at E:\pranav\Final sem\synth_data\outlined_text_captcha_capri_v2_100k. All100000decoded pixel hashes/unique pixels/labels/length-bucket counts/train-text partition audited. Similar top3font mixture + matched frozen adapter. See audit/protocol; no OCR training or real-performance claim.
