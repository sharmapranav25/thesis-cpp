# Style pipeline decisions log

Append-only substantive decisions, reasons, evidence and outcomes. Earlier experiment decisions are in their existing reports; this log begins with the expanded two-phase request.

## D001 — 2026-09-28 — Scope and preservation
Decision: implement the expanded pipeline under a new canonical `style_pipeline` package and immutable run folders. Reason: the user now authorizes generation and distribution refinement; previous experiments must remain reproducible. Outcome: protocol saved as STYLE_PIPELINE_PLAN.md; no original/vendor files edited.

## D002 — 2026-09-28 — Coherent measurement before retraining
Decision: add whole-image constrained block measurements rather than treating the current three-class predictions as complete style profiles. Reason: CryptBB predictions saturated and City recreation missed geometry/typography. No retraining on these mistaken predictions as truth. Outcome: implementation in progress.

## D003 — 2026-09-28 — Validation and modest gate
Decision: freeze fit/validation roles and a baseline-relative, cross-family plus visual gate before running new candidates. Reason: distinct output alone is insufficient, and exact target replication is not required. Prior target exposure is explicitly recorded. Outcome: numerical criteria in protocol 01; no gate result yet.

## D004 — 2026-09-28 — Runtime and licenses
Decision: start with NumPy/Pillow/OpenCV/scikit-learn and existing audited OFL fonts; use local PyTorch for later neural comparisons. No runtime VLM; assistant may review galleries. Reason: transparent open-source CPU measurements first; local RTX A4000 verified for Phase 2. Outcome: torch 2.5.1+cu121, CUDA available. License inventory must be saved before training.

## D005 — 2026-09-28 — Repeated training and research
Decision: retain every meaningful training revision, log trigger and changed settings, select using development results. Reason: user explicitly expects multiple retrainings; avoid silently optimizing held-out scores. Consult primary research when stuck, recording read content and limitations. Outcome: ongoing authorization recorded; not a claim that any new model has been trained.

## D006 — 2026-09-28 — Overnight continuation
Decision: establish a same-chat continuation every 30 minutes, checking active jobs first. Reason: user requests unattended overnight progress rather than status-only updates. Routine notifications suppressed when unchanged. OpenAI automation docs consulted: https://learn.chatgpt.com/docs/automations?surface=app . Outcome: see tool-confirmed scheduling result; this does not imply the computer can work while powered off.

## D007 — 2026-09-28 — Local monitoring
Decision: read-only HTTP dashboard bound to 127.0.0.1:8765, refreshing atomic status JSON every five seconds, with append-only events. Reason: user explicitly requests anytime monitoring without interrupting experiments. Restrict serving to monitor, STYLE documentation and new style-run artifacts. Outcome: server running; same-chat automation `continue-captcha-style-pipeline` confirmed ACTIVE. Initial scheduling call lacked destination; corrected to destination=thread without creating a duplicate.

## D008 — 2026-09-28 — Core ownership and line continuity
Decision: use non-overlapping cores for aggregate pixel statistics while retaining overlapping boxes and adjacency for context. Detect global background line segments and attach their IDs to every intersected block. Reason: avoid counting overlap twice and let one structure connect several blocks. Outcome: three tests pass: odd-size exact coverage, shared line links, deterministic measurements. This is an initial line-based consistency model, not a complete semantic scene graph.

## D009 — 2026-09-28 — Typography uncertainty
Decision: use contrast-connected components as unverified glyph proposals; save position/scale and PCA axis with explicit `tilt_is_rotation=false`. Renderer tries both zero tilt and conservative proxy-derived tilt using fit-only selection. Reason: glyph shape can mimic rotation; no OCR labels or VLM estimates enter the runtime. Foreground/background palettes derive from these heuristic masks and may be wrong. Outcome: pilot underway; failures must inform a generic segmentation revision, not family-specific rules.

## D010 — 2026-09-28 — Bounded generic mapping search
Decision: four procedural variants and three audited fonts, evaluated on 16 fit profiles per family; same fixed text/seed budget. Fit up to 128 profiles and reserve 40 reused validation profiles per family for diagnostics. Render 32 post-selection examples, recording exact replay, masks, operations and text contribution. Compare with one family-agnostic procedural baseline using training-only descriptor scaling. Reason: test end-to-end style conditioning cheaply before expensive retraining. No family-name branches in render.py. Outcome: `runs/style_phase1_01` launched; validation and visual gate pending. This hand-coded CV model complements rather than retrains the previous three-class models.

## D011 — 2026-09-28 — Dependency inventory correction
Decision: inspect installed distribution metadata by prefix, including headless OpenCV, rather than assume the `opencv-python` distribution exists. Reason: initial inventory lookup failed because installed package is `opencv-python-headless`. Outcome: saved STYLE_DEPENDENCY_LICENSES.json. No package was downloaded or replaced; audited font licenses remain hash-checked.

## D012 — 2026-09-28 — Reject first Phase 1 gate
Decision: keep Phase 2 blocked. Reason: only 3/5 generated families nearest their own references, versus >=4 required; rendering failures retained; qualitative broad match partial only for CryptBB/City/DarkBay. Courier and outlined examples have major topology/background mismatches. Outcome: detailed STYLE_PHASE1_01_REVIEW.md saved. Improvement over generic baseline on 5/5 families is useful but insufficient and does not override failures. Do not treat inspected validation images as fresh on the next iteration.

## D013 — 2026-09-28 — Next revision direction
Decision: improve generic segmentation/palette separation and gradient, periodicity and hollow-text evidence with synthetic fixtures before another mapping search. Reason: foreground contamination and coarse block mapping explain observed failure better than simply retraining the previous line/curve/dot classifiers. Continue to store uncertain character rotation explicitly. Outcome: queued for continued development; no claim this revision exists yet.

## D014 — 2026-09-28 — Actual revision start and reporting correction
Decision: implement a small, traceable background revision and run it immediately, rather than claim an unstarted queued revision is active. Reason: status inspection confirmed only the dashboard server was running after the earlier verbal promise. Outcome: prior statement corrected to the user. Added contrast-filtered background palette, block eligibility requiring three background colors, and robust column-median gradient parameters with an optional procedural ramp. No copied patches or family-name conditions. Original run remains immutable; revision uses style_phase1_02. Same metric and gate retained; exposed validation remains development evidence. Segmentation, hollow text and periodicity are still outstanding, not claimed fixed.

## D015 — 2026-09-28 — Region proposal and topology revision
Decision: test grayscale MSER with nested-region suppression, candidate stroke-width proxy and contour-hole ratio, plus an eligible hollow-font rendering option. Reason: single global contrast threshold loses outlined and connected glyphs; documented generic region cues offer a transparent alternative without transcription. Research recorded in STYLE_RESEARCH.md. Gradient uses its measured cue regardless of rotation variant to avoid coupling unrelated controls. Outcome: code implemented for run 03; not a reproduction of SWT or a proven character detector.

## D016 — 2026-09-28 — Revision 2 result
Decision: continue Phase 1. Reason: still only 3/5 nearest-own centroids, though all improve over baseline; Courier typography and outlined contours remain poor in inspected galleries. Outcome: run 02 preserved; no Phase 2 launch or gate relaxation. Upcoming comparisons are iterative development on already exposed references.

## D017 — 2026-09-28 — Revision 3 outcome and next measured correction
Decision: retain MSER as an uncertain proposal source but do not call it a solution. Reason: run 03 still gives 3/5 nearest-own centroids; hollow-ring fixture passes but outlined real glyphs are often missed. CryptBB maps nearest City (too sparse); Courier maps nearest DarkBay. Reviewed CryptBB/Courier/DarkBay/outlined galleries; failures retained. Next: measure small-scale background residual occupancy and use it for texture density; sample actual measured line angles instead of biased histogram-bin centers. Reason: bin center creates an artificial 7.5-degree tilt for horizontal lines, and counting tiny connected edge components misses connected texture.

## D018 — 2026-09-28 — Explicit local contrast safeguard
Decision: choose only measured foreground palette colors with >=65% significant glyph-mask pixels differing >=12 from the rendered background; if none qualify, record a contrasting gray fallback. Reason: several MSER-derived palettes included background-like colors, causing invisible generated glyphs. This is an explicit generation safeguard, not a claim of matching target foreground colors. Character labels and seeds stay recorded; no reference label is consulted. Outcome: implemented for next run, retain old failures.

## D019 — 2026-09-28 — Revision 4 numeric success, gate still pending
Decision: keep the full gate pending despite 5/5 nearest-own centroids and 5/5 improvements. Reason: all 160 final images rendered, but two search candidates failed visibility; outlined typography is still a poor match. Reviewed all five galleries: CryptBB texture improved substantially; City, Courier and DarkBay have partial broad resemblance, with remaining interference, contrast and geometry differences. No exact-resemblance claim. Outcome: preserve run 04 and strengthen readability before the next run.

## D020 — 2026-09-28 — Alpha-aware, individual-character integrity
Decision: evaluate candidate color contrast after glyph alpha, require 75% support above 24 levels for palette eligibility, and use the better of black/white only when measured colors fail. Then verify every glyph via a counterfactual rendering that omits that glyph (55% significant support must change by >=12), as well as the existing union check. Reason: pre-alpha contrast can pass while actual anti-aliased text remains invisible; union checks can hide a bad individual character. This changes rendering safeguards, not the frozen family-comparison metric or gate. Outcome: run 05 will evaluate the same fit/search/validation budget; no previous artifacts overwritten.

## D021 — 2026-09-28 — Revision 5 visual review and independent diagnostic
Decision: rate broad resemblance partial for CryptBB, City, Courier and DarkBay; poor for outlined. Reason: CryptBB now has multicolor textured blocks; City has the light background, small separated text and horizontal interference (but strokes are much too thin); Courier has varied light/dark/color backgrounds with colored rotated text (but lacks strong curved interference); DarkBay has muted backgrounds and mixed light/dark glyphs (but wrong block geometry/font). Outlined has a gradient but fails hollow typography. This is a modest assistant visual judgment, not a runtime VLM score or human audit. All 160 final renders and all search renders passed individual-character and union integrity, with exact recipe replay. No catastrophic missing text is evident, but several examples have weak visual contrast. Numeric gate remains 5/5 and 5/5. Before closing the gate, measure an independent RGB/gradient/spatial descriptor against historical validation positions 40:80, frozen here; do not tune on that result or call it pristine, given earlier pool exposure.

## D022 — 2026-09-28 — Bounded Phase 2 comparison preparation
Decision: prepare three small, from-scratch models: residual CNN, two-layer patch transformer, and CNN with conditional local GAN critic. Same 192x64 input warp, balanced family batches, synthetic corpus and initial 400 optimizer-step budget; no pretrained models or VLM. All three preserve synthetic glyph-mask pixels exactly, including a safety dilation, so this first comparison learns background/interference shifts rather than changing fonts or character geometry. Reason: preserve generated transcription provenance while testing whether learned distribution adjustment adds measurable value. Use fixed low-level moment alignment and a small identity penalty, plus adversarial loss only for GAN. Record losses, wall time, checkpoints, unpaired fit data and independent development metrics. Training remains forbidden until an explicit gate-pass record. Prepared refiner mask/gradient/range test passes; no model trained yet. The transformer architecture is much smaller than ViT and the loss is not a Deep CORAL reproduction.

## D023 — 2026-09-28 — User requests an unseen scheme before Phase 2
Decision: hold Phase 2 and test an additional downloaded scheme with frozen run-05 extraction/render code. Source selected before inspecting examples: roncuevas/Captchas, first new directly downloadable licensed text dataset found in this search; this is arbitrary scheme selection, not a statistical random draw over all schemes. Repository declares MIT; retain its license and provenance, with no separate claim about source-site ownership. Select 168 image paths by seeded hash ranking, pin commit, download images only (no labels, models, generator or fonts). Partition 128 style-reference/40 evaluation by pixel hash, check overlap with all five historical manifests. Run the same 4-variant/3-font/16-reference mapping search without code changes or evaluation-driven tuning. Generate 32 examples and compare against the fixed generic baseline using old metric scaling plus independent low-level metrics. Reason: test generalization to a previously unseen style rather than continue optimizing the five development schemes. Phase 2 remains on hold. Earlier spoken Phase 1 pass was a modest visual interpretation; gate.json was still pending when interrupted, and the independent descriptor only separated 3/5; preserve that distinction.

## D024 — 2026-09-28 — Complete sampling frame and download recovery
Decision: reject GitHub's truncated image-tree API result and fetch a shallow, blob-filtered bare Git metadata repository to enumerate the complete image list. Reason: sampling a truncated prefix would bias the promised random image sample. No upstream scripts are checked out or executed; labels/models/generator blobs are not requested. Download 168 seeded image paths through pinned raw URLs, retain MIT license and manifest. Python's HTTPS support was unavailable in this interpreter, so use PowerShell HTTPS rather than changing installed dependencies. Outcome: 168 images downloaded at commit 61024dad7c77fb94880f6d5490a0d6293e2d2607; frozen test started. No Phase 2 training process is running.

## D025 — 2026-09-28 — Unseen test outcome, no generalization pass
Decision: preserve the frozen test as incomplete and keep Phase 2 on hold. Outcome: 32/32 final renders, primary distance 62.79% better than generic baseline, independent distance 43.97% better; generated centroid nearest the new reference family among six. However nine search candidates rejected clipped glyphs, and assistant visual review finds weak resemblance: five forced characters versus three prominent reference characters, missing dense grid/curved structure and several dim outputs. No classifier accuracy claim. New collection is now exposed development data; any subsequent revision on it is not another unseen test. Historical-pixel overlap check covered five fit/validation manifests, not every possible external image in the world.

## D026 — 2026-09-28 — Generic rotation containment and count investigation
Decision: fit each rotated glyph uniformly inside its assigned horizontal cell and image height, recording the scale, then retain all existing visibility checks. Reason: expansion after rotation caused clipping on narrow 100x50 inputs; cell containment also prevents accidental character overlap without family-specific rules. Separately investigate a tall-region count estimator with explicit uncertainty, rather than replace five with the new dataset's known three. Raw MSER proposal counts on its 128 fit images range 0–12, so using their count directly would be misleading. Preserve frozen run and source snapshot; validate geometry across old and new fit profiles before claiming a fix. No Phase 2 launch.

## D027 — 2026-09-28 — Explicitly integrate frozen learned-feature inference
Decision: run the saved three synthetic-trained Logistic Regression classifiers with their existing synthetic calibration on all 168 new images, 30 overlapping blocks each. Do not fit, recalibrate or use real labels. Verify model hashes before and after. Save probabilities, states, 3x10 maps and per-image joint states; preserve 128/40 partition in summaries. Use fit images only to guide a 32-image generation batch: learned presence gates straight lines, curved interference and dots; unknown/absent is not rendered for these three types. Curves use two explicitly generic sinusoidal strokes, not inferred curvature/density. Keep CV measurements for other attributes. Reason: user clarified the required synthetic-training -> frozen real inference -> generation flow; previous CV-only test did not execute the learned models. Caveats: historical any-positive-block aggregation is uncalibrated and can saturate; five-character layout remains unresolved, so this integrates inference without claiming that all style gaps are solved. Includes the already implemented generic rotation containment fix; comparison is not a pure classifier-only ablation.

## D028 — 2026-09-28 — Frozen inference completed; saturation disclosed
Outcome: 168 images / 5,040 blocks, unchanged model hashes, no training or recalibration, 32 classifier-guided renders with zero rendering failures. Straight and curved image predictions saturate at 168/168; dots at 3/168. Positive blocks: 4,362 straight, 4,810 curved, 4 dots. Decision: keep Phase 2 on hold and do not claim real-domain accuracy or successful family reproduction. Reason: saturation plus known five-character/geometry mismatch makes a success claim unjustified. Six tests passed; report and galleries in STYLE_FROZEN_INFERENCE_01.md. Preserve original CV-only test and current inference run. Subsequent classifier improvements must train/calibrate on synthetic data only; real images remain inference and evaluation inputs.

## D029 — 2026-09-28 — Apply the same frozen inference-to-generation flow to all five previous families
Decision: process the exact 128 style-reference + 40 evaluation images per family used by Phase 1 run 05 (840 images / 25,200 blocks total), then generate 32 examples per family with the same classifier gating as the new dataset. Reason: user requested consistent infer-and-produce comparisons for the five previous datasets. Reuse identical synthetic-trained model files, synthetic thresholds, existing fit-only font/variant selections, text and seeds. Verify source pixels/file hashes and classifier hashes. No retraining, new calibration, family-specific rendering branches or real labels. Preserve all old runs and save five new immutable runs. This is the matched Phase 1 cohort, not a claim to have reprocessed every image in each full dataset. Retain saturation, any-positive-block aggregation and five-character layout caveats; Phase 2 stays on hold.

## D030 — 2026-09-28 — Clarify the CV-only implementation choice
User asked whether CV-only generation was added by the assistant. Clarification: yes, the shared CV measurements were added for the expanded color/layout/tilt/density work; generating without the existing classifiers was an assistant implementation choice, not a user requirement. The current five-family runs combine frozen learned-feature predictions with those shared measurements. Preserve this distinction in documentation; do not describe old run-05 generation as learned-classifier inference.

## D031 — 2026-09-28 — Monitor lock recovery without altering models
Observed: DarkBay attempt 01 stopped during a dashboard os.replace with Windows PermissionError, after inference records had been saved. Decision: preserve that partial run; make monitor status replacement retry transient locks with a process-specific temporary file and retain events even when the dashboard replacement is deferred. Retry DarkBay as attempt 02; then run outlined. No changes to descriptors, classifiers, thresholds or rendering. Reason: a dashboard reader must not terminate scientific work. The failed run remains available for audit.

## D032 — 2026-09-28 — Five-family inference and generation complete
Outcome: all five matched cohorts completed, 840 images / 25,200 blocks, three unchanged model hashes in every run, 160 generated images with zero rendering failures. Image-level present counts (straight/curve/dots): CryptBB 168/168/110; City 168/166/27; Courier 168/168/9; DarkBay 168/158/56; outlined 168/168/0. Decision: retain descriptive results without a generalization or accuracy pass. Reason: straight predictions saturate in every family and curves nearly saturate; assistant inspection of the first-seed comparison still shows weak typography/geometry matches despite reduced unsupported dot clutter. Six existing mapping/measurement tests passed; simulated transient monitor lock recovered. DarkBay attempt 01 remains partial; completed attempt 02 is used in the report. See STYLE_FROZEN_FIVE_01.md and style_frozen_five_report_01/comparison.png. No real-data training or Phase 2 run.

## D033 — 2026-09-28 — User explicitly authorizes Phase 2 experiments
Decision: proceed with Phase 2 as an exploratory comparison, following the user's explicit approval after viewing frozen-inference results and discussing limitations. Do not rewrite failed/pending Phase 1 gates as passes. This authorization supersedes the earlier operational hold, not the evidence. Feature classifiers remain synthetic-trained, hash-frozen and inference-only. Unlabeled real fit images may train Phase 2 distribution/recipe models. Existing source-family code/fonts and runtime VLM remain prohibited.

## D034 — 2026-09-28 — Freeze Phase 2 pilot protocol before evaluation
Decision: five original families, first 96 of the existing 128 fit profiles for Phase 2 fitting and last 32 for development. Reserve historical validation pixel-hash positions 80:120 (40/family) for end-of-pilot reporting only, with no selection/training on them. These are additional evaluation images, not globally pristine given historical pool exposure. Recipe search: zero baseline plus 31 seeded bounded candidates, 8 fit render profiles each; top candidates selected using 8 development profiles. Learn 11 bounded controls: RGB background offsets, text height, tilt, spacing, line count/width, dot count, curve count/amplitude. No character-count inference or source-font recovery claimed. Train small CNN/transformer recipe predictors from scratch against fit-derived search targets; this is search distillation, not gradient descent through Pillow. Pilot 250 steps each. Also test a fit-trained discriminator to rank the same recipe candidates (adversarial feedback, not a differentiable-renderer GAN). Then compare 400-step CNN, transformer and GAN direct refiners using the same balanced synthetic/real fitting pools. Exact glyph-pixel protection limits direct refinement to background/interference. Preserve zero baseline, all candidate failures and checkpoints. Final evaluation happens only after model/recipe selection is frozen; report metrics, variability, glyph integrity, galleries, runtime and failures without silently weakening criteria.

## D035 — 2026-09-28 — Phase 2 pilot launched and verified
Outcome: nine preflight tests passed (recipe bounds/gradients/zero-control identity, exact refiner mask preservation, classifier gates and existing measurement fixtures). scripts/run_style_phase2.py is actually running, output runs/style_phase2_pilot_01; process observed, protocol/splits/scale saved, baseline synthetic pool prepared and candidate search active. Initial methods: search-only recipe, CNN/transformer search-distillation predictors, discriminator-ranked recipes, and direct CNN/transformer/GAN refiners. Discriminator-ranked recipes are explicitly not called an end-to-end differentiable GAN. Frozen classifiers are not fitted or recalibrated. Monitor reports concrete current stage; no improvement claim before reserved evaluation and visual review.

## D036 — 2026-09-28 — First Phase 2 comparison completed, direct refiners rejected visually
Outcome: pilot completed in 151.88 seconds of measured process wall time after preflight (small pilot, not an overnight-scale training claim). Seven learned/search methods plus baseline, 32 generated images per family/method: 1,280 final renders, no rendering rejections; feature-classifier hashes unchanged. Recipe search/CNN/transformer/discriminator reduce mean reserved descriptor distance by 33.2/34.7/31.6/36.5%, improving three families. Direct CNN/transformer/GAN reduce it by 62.6/61.5/61.9%, improving five numerically. Decision: reject direct-refiner visual success because all five reviewed galleries show artificial halos around protected text and broad tone shifts. Recipe methods remain exploratory candidates, not uniformly validated visual improvements. Reason: training/reporting descriptors reward distribution moments without ensuring perceptual fidelity. All output/checkpoints and visual_review.json preserved; docs/STYLE_PHASE2_PILOT_01.md records details. Reserved window is now exposed and must not become a future training/selection source or be called untouched again. Further optimization, if pursued, must use only fit/development evidence and separately predeclared evaluation. Do not claim the overall research objective achieved.

## D037 — 2026-09-28 — Phase 3 font adjustment authorized and research direction
Decision: implement visual font-style matching without OCR, runtime VLM, target generator fonts or real-label classifier training. Research: read DeepFont primary paper sections 2.1–2.3 and 3.1 (https://arxiv.org/html/1507.03196): synthetic-to-real mismatch, text-specific augmentation, multi-view font inference. Its unsupervised real-image adaptation is NOT adopted: the new font classifier will train on synthetic examples only, consistent with user constraints. Use a small original CNN, not Adobe weights or a claimed reproduction. Source a broader independent OFL font bank; retain licenses/hashes and exclude any file whose bytes match fonts in the imported repository. Model predicts approximate font candidates and hollow/filled evidence; no exact source-font identity claim. Original three feature classifiers stay hash-frozen. First experiment uses existing 96 fit / 32 development profiles only; do not access Phase 2 reserve. Preserve earlier outputs and report development-only results. Synthetic holdout metrics and real uncertainty must be reported separately.

## D038 — 2026-09-28 — Phase 3 synthetic training/testing protocol and thesis register
Decision: train a 12-font original CNN on 256 synthetic images/font, select a checkpoint on 64 separate synthetic images/font, then test once on 64 further images/font. Seeds/content are disjoint across partitions; background/augmentation seeds shared across fonts to avoid class-specific nuisance cues. 700 steps, batch 64, font classification plus hollow/filled head. Freeze before any real inference. Infer on five families' existing 96 fit/32 development profiles with four views/image, then search top-three-font typography candidates (filled/bold/hollow, width, height) and retain baseline as an option. All real results are exploratory development, not held-out accuracy. No Phase 2 reserved-window access. Maintain docs/THESIS_PAPERS.md as the curated thesis register, linking actual code and distinguishing inspiration from reproductions and consulted-only sources. Nine additional OFL fonts were downloaded; two requested ofl directories did not exist and were replaced with available independent OFL families; variable-font brackets required literal-path hashing. No dependencies replaced. Imported repository font scan found zero font files, so no claim that matching a known source-font hash list proves absence of unknown overlap; fonts were independently sourced, not extracted from targets.

## D039 — 2026-09-28 — Font pilot tested and next training iteration
Outcome: step-700 selected model yields synthetic test top-1 50.52%, top-3 81.64%, hollow/filled accuracy 92.19%; real accuracy unknown. All five families inferred and 160 development candidates rendered without failures. CryptBB serif change is plausible in inspected examples; outlined still filled and visually fails. Decision: continue from the synthetic-selected checkpoint to step 2,000 with lower learning rate in immutable run 02, use the same synthetic training/selection pools and a new synthetic test seed range 30,000:30,064. Reason: synthetic selection accuracy improved from 24% at step 200 to 47.1% at 700 and loss was still falling; undertraining is plausible. Do not choose a checkpoint using test scores. Preserve run 01, disclose repeated development exposure, and retain real evaluation embargo. Original feature classifiers remain frozen. Curated thesis paper register now exists and identifies the exact borrowed ideas and non-reproductions.

## D040 — 2026-09-28 — New external family for the next frozen-loop check
Decision: use all 12 clean previews published by pymlex/4-captcha, pinned HF revision d36ce8b0cf06f2ffcdc459482a0c94b8f8380f78. Reason: it is an independently generated new CAPTCHA family with a GPL-3.0 dataset declaration; the full archive is 3.04 GB, so this initial run is deliberately a tiny smoke test, not a representative benchmark or a random sample. Downloaded images and source card only, no labels, source generator or fonts. Four published training previews guide inference aggregation, eight val/test previews remain outside parameter selection. Freeze completed font model 02, Phase 2 CNN recipe predictor and all original feature classifiers before image testing. Execute CV AND frozen feature inference, then frozen recipe and font predictions, with no new-family training or search. Keep five-character limitation explicit. Record all outcomes; do not turn a small preview test into evidence of broad generalization.

## D041 — 2026-09-28 — Record improvements and font-domain failures separately
Outcome: run 02 completed synthetic training/test and 160 development renders. Matched fresh synthetic test: top1 45.70->56.25%, top3 77.47->84.77%, hollow 89.06->90.23%. New independent family's preview smoke test executed CV + original feature inference + recipe + font inference with unchanged hashes and 48 renders. Decision: do not declare Phase 3 successful. Reason: outlined typography remains filled/bold and the new family is assigned Pixelify Sans, visually inconsistent with references. Reports preserve the failure and distinguish synthetic classifier gains from real-domain performance. Thesis paper register updated with both experiment outcomes and exact used ideas. Next investigation should isolate synthetic augmentation/low-resolution clutter sensitivity using synthetic fixtures and development-only diagnostics, not the Phase 2 reserve.

## D042 — 2026-09-28 — Synthetic robustness loop started without real-data classifier fitting
Decision: preserve both completed models and start font run 03 from run02 to step3200. Alternate standard and 3x synthetic background noise/line clutter in training, synthetic selection and fresh synthetic test; cycle training images through the same four view types used by real inference. Use new synthetic test seeds40000:40064; choose checkpoints only on synthetic selection. Reason: whole-image-only training versus multi-view inference is an implementation mismatch, and the independent preview test demonstrates poor transfer under clutter. No new real labels/gradients or target-specific rules. DeepFont's augmentation and multi-view ideas directly motivate this change; no new paper method claimed. Run03 synthetic metrics use a harder mixed-clutter distribution and are not directly comparable with run01/02 headline numbers without a matched evaluation. Reusing the new preview family after this revision would be development, not another unseen test; a later loop needs a genuinely additional family.

## D043 — Evaluate robustness fairly and retain the failure (2026-09-28)

Run 03 completed, selected checkpoint 2800. Because its test includes stronger clutter, compare both frozen models on exactly the same 768-image test rather than comparing incompatible headlines. Top-1 improved 45.96% to 54.56%; top-3 74.09% to 84.24%. All 160 development renders succeeded, but assistant gallery review still finds outlined filled/bold instead of hollow. Keep Phase 3 unpassed; do not promote lower self-model development loss to real accuracy. Paper register updated with actual outcome and attribution limits. Next work should investigate outline evidence/segmentation and synthetic-to-real mismatch using fit/development data; no reuse of reserved Phase 2 evaluation for selection.

## D044 — Continuous three-phase loop and outline diagnostic

User renewed ongoing authorization. Updated existing 30-minute heartbeat to all three phases and paper maintenance, rather than creating duplicate scheduling. Next run diagnoses hollow-style domain mismatch with frozen run03 inference and CV MSER topology on fit images only plus synthetic positive/negative controls. No training or parameter selection from reserved evaluation. Reason: more training alone improved synthetic scores but did not fix real outlines. Diagnostic is not real-label accuracy and cannot pass a gate.

## D045 — Outline diagnostic outcome

Frozen run03 evaluated 160 existing fit images (32/family) and 384 fresh synthetic controls using four-view inference. Synthetic filled mean hollow probability .0388; hollow .8363. Real outlined mean .2505, while City .4722. MSER median hole ratio for outlined is zero. These are diagnostic probabilities, not real accuracy. Both visual representations appear mismatched to target outlines; prioritize synthetic outline/thickness/contrast/scale ablations and segmentation diagnostics rather than further identical training. Never choose a family-name-specific effect override. Evidence: runs/style_phase3_outline_diagnostic_01/diagnostics.json; no checkpoint changes or reserved evaluation access.

## D046 — Paired synthetic contrast ablation

Assistant fit-only gallery review found outlined strokes are low contrast against a gradient, unlike the mostly high-contrast synthetic training examples. Test frozen run03 on paired synthetic images at contrast gains 1, .35, .15 around gray128, preserving labels/content/noise arrangement. This is a local diagnostic, not a paper reproduction or real-label metric. No training/checkpoint selection from these diagnostic seeds. If probability collapses under contrast reduction, expand synthetic contrast support using new train/selection/test seeds and retain standard controls.

## D047 — Contrast diagnostic and controlled training revision

Paired 384-image synthetic diagnostic: font accuracy at gains1/.35/.15 is55.99/53.91/18.23%; hollow accuracy87.50/85.16/41.67%. At .15 both hollow and filled mean hollow probabilities converge near .72. Contrast alone does not explain the outlined false negative; it does expose a real robustness failure. Add optional contrast-training gains1/.65/.35/.15, evenly shared across font classes, with view cycling independent of gain. Warmstart run03 synthetic-selected step2800, train to4400, select on synthetic selection only, fresh test seed70000. Retain standard contrast quarter; report matched contrast-stratified evaluation before success claims. Preserve old data/checkpoints. This is original local augmentation under the previously documented DeepFont inspiration, no additional paper adopted.

## D048 — Contrast revision tested; no universal promotion

Matched fresh reporting test shows gain.15 hollow39.71->68.75%, font20.96->26.30%; standard gain1 hollow95.31->91.93%, font51.82->50.00%. Run04 trades robustness against normal contrast and still fails real outlined rendering. Retain both checkpoints, no automatic replacement or gate pass. Four outlined development samples inspected visually, no runtimeVLM. Full report STYLE_PHASE3_FONT_04.md. Next test outline geometry/scale independently; do not tune on this reporting test.

## D049 — Controlled outline geometry probes

Before another training change, compare frozen03/04 on synthetic filled, inner-boundary, outer-boundary and half-pixel inner-boundary glyphs at24/40/52px. Same12 independently licensed fonts, eight shared random texts each; no real images, training or checkpoint selection. Reason: original hollow training only uses a one-pixel inner erosion boundary, while real outlines vary in stroke geometry and size. This original diagnostic tests representation support, not real accuracy. Keep classifier versions unchanged and preserve results.

## D050 — Geometry support expansion

Diagnostic1152 paired synthetic fixtures: at52px outer outlines, hollow recall03=.177/04=.125 versus thin-inner .896/.781; clean filled false positives zero at this size. Assistant checked control grid. Add optional synthetic geometry variation: inner/outer/subpixel-inner boundaries, height20–54, keep default output unchanged and all historical snapshots. Warmstart03 (not universal promotion of04) to4400 with robust+geometry, no contrast flag, fresh test110000. Select on synthetic only; real96fit/32development inference remains exploratory. No claim these controls prove real source geometry. This is local original ablation, no new research paper.

## D051 — New independent-source frozen test

Run05 complete, synthetic test47.92%font/90.10%hollow; outlined real stillbold. Freeze05 before testing project-sloth/captcha-images revisioneeaf2b6ec9086645f270f7e2aaa9ea90730683de, declaredWTFPL license saved. Download image test archive only pluscard/license, no generator/model/font files. Seed20260928 selects32 images from archive; local16 reference for inference aggregation,16 evaluation excludedselection. Archive names may contain labels; never parse/use as text and extract only imagebytes toopaque numeric filenames. Archive retained for provenance; do not claim labels absent from downloaded archive. Existing generic5-character rule retained despite datasetcard textlength. This is new-source testing; generator independence from prior families is not proven. No realclassifiertraining, CV+historicalfrozenfeature+recipeCNN+font05 inference all required.

## D052 — New-source frozen test complete; reject visual success

32images/960blocks,48renders across3phases,0rendererrors, all model hashes unchanged. Assistant inspected four evaluation examples and corresponding generated previews. Target is pale grayscale, dotted noise, slanted filled digits; generated outputs introduce colored line clutter, five mixed characters, and Phase3 incorrectlyhollow text. Phase2 visibly worsens clutter relativePhase1. This is a failed style-smoke test, not benchmarkaccuracy. Do not use observed reference text as labels or hardcode a family rule. Prioritize next loop Phase1 false line/curve gating and Phase2 bounded recipe fallback under shift; stop spending every loop on font-only retraining. New source is independently obtained but generator identity not established. These32images are now exposed; any subsequent adjustment using them is development, not unseen.

## D053 — Achromatic consistency ablation

Inspection found two mechanisms invent color: RGB-separated recipe offsets and independent per-channel random dot noise. Implement OPTIONAL measured-achromatic projection (source saturation<=.03) of all palettes/gradients/blockmeans and shared-channel dot noise, preserving intensity and random draw count. No family branches or classifier overrides. Test16 already-exposed sloth reference images using cached CV+frozen model outputs and same text/seeds. This cannot validate generalization or cure false lines. Original renderer default unchanged; no real training or reserved evaluation. Classifier line/curve probabilities saturate near1 across all30blocks, explicitly unreliable.

## D054 — Color ablation outcome

Mean RGBspread24.125->0.000,16exposeddevelopment images,32pairedrenders. Optionalconstraint eliminates inventedcolor but not falseinterference; black readabilityfallback appears on someglyphs. Do notpromote fullstylepass. Defaultpath preserved; constraint and recipe tests passed. Nextdiagnose featurepreprocessing/calibration on synthetic controls before touching real thresholds.

## D055 — Frozen feature inference audit

Replay48 saved synthetic test crops against stored descriptors, compare raw/calibrated scores for synthetic test,480 exposed real blocks and9 blank controls, inspect standardized linear contributions. No thresholds or weights changed. Training uses decision_function before sigmoid; live inference follows same rule. Initial audit import resolved canonical feature_bank without descriptors and failed before output creation; corrected module path to the historical staging package used for original inference, preserve errorlog. This is diagnostic reporting only, not calibration on real images.

## D056 — Synthetic nuisance confusion reproduced

Descriptor replay48/48 exact, blankslow, inferencecalibration correct. New synthetic dot-only controls40points cause32/32curve>.99;100points28/32straight>.99,32/32curve>.99, mediandot.111. Strong evidence for syntheticnegative coverage failure, not a realthresholdfix. Nextloop train newclassifier version on original synthetictraining plus dense-dot/texture hardnegatives, with independent selection/calibration/test seeds and ordinarybenchmark regression checks. Preserve oldmodels; no realtraining. See STYLE_FEATURE_AUDIT_01.md.

## D057 — Synthetic dot-negative LR revision protocol

Train new straight/curve bundles using original synthetictraining membership plus512 grayscale-dot-only crops, disjoint128selection/128calibration/128test seeds130000/140000/150000/160000. ThreeC candidates .01/.1/1, negativeweight4. Gate ordinaryselectionAP decline<=.02; minimize newnegative selectionFP among eligible. Calibrate on syntheticcalibration only. Preserve originalmodels; no automatic runtimepromotion, thresholds initiallyunset. Newtestnegatives used only afterselection; oldordinarytest previouslyexposed reporting. No realimages accessed. Dotclassifier unchanged because this experiment targets falseinterference, not allfeaturefailure modes.

## D058 — Negative revision outcomes, gate respected

Straightpasses ordinaryselectionAP gate; freshdotnegativeFP at.5 drops71.09->.78%, ordinarytestAP.8832->.8709. Curvecandidatesallfail selectionregressiongate; saveddiagnosticfallback isREJECTED, ordinarytestAP.8800->.8063 despiteFP97.66->.78%. Do notweaken gate or deploycurve. Straight stillrequires syntheticcalibration abstentionthresholds beforeintegration. Noactivejob, originalmodelsunchanged. See STYLE_FEATURE_NEGATIVES_01.md.

## D059 — Straight calibration and controlled development inference

Freeze acceptedstraight01 model; reconstruct only ordinarysyntheticcalibration plus128 newdotcalibration seeds150000–150127. Select nonoverlapping absent/present thresholds maximizingcoverage with empiricalNPV/precision>=.90 andsupport>=30, same historicalcriterion. These are in-samplecalibration estimates, not realconfidence. Storethresholdseparately withoutmutatingbundle. Then rerun pixelCV+straight inference on16exposedslothreferenceimages, compare pairedrenders with originalcurve/dot outputs and frozenrecipe parameters unchanged; botharmsuseachromaticconstraint. No realthresholdfitting or new-source claim.

## D060 — Real transfer failed despite synthetic gate

Syntheticthresholds .546983/.785389 fixedbeforeinference.16exposedslothimages:480/480blocksstraightpresent,295lineoperationsbotharms. Fourpairedrenders visuallyunchanged. Do notpromote straightrevision as realfix; syntheticdotcontrols insufficient. Next broaden synthetictexture/scale/antialiasing negatives and independentgeometrysupport; nevertune thresholds onreal. Allmodelhashesunchanged. See STYLE_STRAIGHT_REVISION_EVAL_01.md.

## D061 — Continuous geometric line evidence

Optional straightline count cap from longHoughsegments length>=.3width, maxgap2, localorthogonalcontrast>=20 on>=80%segment, duplicateedge suppression. Classifier stillrequiredpresent; neither scores nor thresholds changed. Synthetic32dotcontrols/32matchedlinepositives first; predeclare sensitivity>=.9,falsepositive<=.1 before exposed16image ablation. GeneraloriginalCVheuristic, no family branches/newpaper. Doesnotaddresscurves,shortlinesorbrokenlines; preserve defaultpath.

D061 firstoutcome: syntheticrecall.6875,falsepositives0; gateFAILED and no realablationran. Revised Hough proposal gap6/thresholdmax10,width16 to tolerate clutter-fragmentededges; retain pixelcontinuity80%, length.3width and unchangedgate. Rerun diagnosticseedset as DEVELOPMENT, not independenttest; preserve run01.

## D062 — Geometric support experiment fails synthetic gate

Run01 recall68.75%,run02 59.375%, both0/32dotfalsepositives; requiredrecall90%. Bothstopped before realablation, gateunchanged. Experimental support code is NOTenabled inpipeline; optionalrendercap defaultsabsent. Revised proposals didnotfixmisses. Next inspect syntheticfailedexamples and compare directintensityridge proposals rather than repeatedly looseninggate. No newpaperused. Userstatusupdate accuratelyreportedpartialprogress, noreliableendtoendsuccess.
\nD063: Background audit complete: 160 exact replays; font changes preserve background operations. Phase2 recipes amplify CryptBB dots2.67x, City linewidth4.03x, Courier lines4.08x and shift colors. DarkBay/outlined recipes zero, baseline mismatch predates Phase2. Rollback mixed; no universal promotion. Shared RNG changes glyph colors/jitter when background operations change. Next separate RNG streams and validate background controls. See STYLE_BACKGROUND_REGRESSION_01.md. No new paper.


D064: User changes sequence: Phase1 unchanged, Phase2 fonts (formerly3), optional Phase3 distribution refinement (formerly2) deferred pending assessment. New direct font runner bypasses all old recipe adjustments. Preserve old runs and numbering as historical. No claim old font renders followed new order.


D065: User requested all bank renders under revised order. Completed32 renders across8 sources, freshCV+frozenfont05 inference, cached image-hash-verified feature inference. No formerPhase2 recipes. Mean-font posterior selection, not inherited recipe-dependent search. Backgroundoperations unchanged32/32. Existingfit/reference only, no newunseenclaim. Wrongpixel/hollow/count/layout remain; no gatepass. See STYLE_DIRECT_BANK_01.md.


D066: User authorized Phase3 generation with colors locked, otherrecipecontrols retained.32renders/8sources completed; exactpalette/gradient restoration, glyphRGB and per-kindoperationcolors locked. No failures. Geometry/density may change pixelcolors through coverage; no pixelidentityclaim. Recipe-based only, no unrestrictedpixelrefiner. Visualoutcome mixed, Slothclutterregression. See STYLE_PHASE3_COLOR_LOCKED_01.md.


D067: Diagnostic-only colorlocked Sloth ablation,20renders,4pixel-exact baseline replays. Recipe raises26lines to75 and width1 to3; dots reduce5879 to2988. Colorassertionspass. No runtimecontrolchange because user said retainothercontrols. No gatepass/newfamily claim; see STYLE_PHASE3_CLUTTER_ABLATION_01.md.


D068: All-bank parameteraudit320renders,0failures,colorlocked,baselineexact. Strongestvisualcandidates Citylinewidth, tentativeCouriercurvecount andPymlexheight. CryptBBlinecontrols numericalonlyweak; no convincingsloth/roncuevaswinner; DarkBay/outlinedparameterszero. Spacingmostlyharmful, tiltmostlyinactive, curveamplitudeweak. Currentvaluesonly,4devsamples/source, reusedlowlevelmetricnotaccuracy. No automaticpromotion/usercontrolschange. See STYLE_PHASE3_PARAMETER_AUDIT_01.md.


D069: User defines activePhase3 as countonly, optionalrefinement outsidepipeline; automationpaused. TrainedoriginalsyntheticCNN2-8, no realtraining/OCR/type/VLM. Beststep500: synthetic89.29%,withheldfonts93.53%.212realinferences mixed, Courier/Roncuevas overcount; no generationpromotion. Split/label/checkpoint checks pass. See STYLE_CHARACTER_COUNT_01.md; countsupport limited2-8, possiblesyntheticlayoutshortcuts. Failedclipping fixture caughtbeforetraining and fixed.


D070 — User-requested DINO/CV/density count comparison completed. Synthetic-only threshold/checkpoint selection prevents real-family tuning. Reused cached frozen open-source DINO; trained density on synthetic glyph centers only. DINO53.87%, density38.99% vs priorCNN88.10% fresh synthetic exact; real212 mixed. Reason for rejection: attention follows noise/parts, density transfer fails. No promotion or automation restart. See STYLE_COUNT_HEATMAPS_01.md for provenance, deviations, validation and visual limitations.


D071 — User moved count into Phase1 and requested allfamilies Phase2preview. Used frozen countCNN meanposterior with4predeclared stratifiedquantiles, preserving uncertainty without asserting real calibration. Freshcountinference212references; CV plus cached pixel-matched features and freshfonts.32finalrenders pass glyphcount/visibility/background-operation checks. Real count andfontfailures documented; no silent family overrides or successgate. No retraining/optionalrefinement/automationrestart.


D072 — User drops unreliable length module and authorizes optional DINO parameter adjustment. Predeclared15 candidates (baseline and one-control low/high variants); fit first4 references and report next4, Pymlex2/2. All previously exposed development, not unseen. Frozen DINO CLS + attention-weighted patch means + spatial attention occupancy guide selection; CLS-only ablation isolates selection effect. No model training; baseline eligible. Exact glyph/drawing RGB, palettes/gradient, font/effect/text/count locked, failed visibility rejected. Historical scripts/artifacts preserved. Local trial dino_recipe_01 running; no promotion.


D072 outcome — Optional DINO search complete8families/30finalrenders. Mean objective improvement2.55%; separateCV improves4/8; attention changes selection1/8. Small/mixed visual effects; no promotion. Rejected new-operation color candidate and resumed; no constraint relaxed. Countdisabled; fixed5only. See STYLE_DINO_RECIPE_01.md.


D073 — Prior shared Phase3B trial completed90renders (CNN/transformer/GAN,300steps each,86fit30report). Mostly identity: CNN/GAN change2families, transformer4. Protected pixels/colors passed but visual benefit weak. User correctly requests family-isolated training; prior run preserved, not promoted. D074 — Eight separate GANs now initialized independently; no familyconditioning or shared learned weights. Larger96reference cohorts split72fit12selection12report inside existingfit pools, small sources proportional/Pymlex2/1/1. Two synthetic draws per fitting reference. Max1500steps, minimum500, validation every100, patience4; identity baseline eligible. DirectRGB residual projected to input colors, character mask exact. This replaces restrictive spatial-only edits without introducing RGB colors. Candidate color coverage can change. No countmodel, fixed5. Preserve exposure caveats; no pristine evaluation claim.


D075 — User declared Phase3B failed and stopped further experiments. D076 — User requests self-contained Phase1+Phase2 plus training guide, no training in this session. Bundled exact frozen assets and independent fontlicenses with relativepaths; exported96fitprofiles for6families,4Pymlex,16Sloth; provenance retained. Default renderer byteparity8/8. Added explicit newreference CV+frozeninference entrypoint, balanced21cell generation, stabletextpartition and crosspixel/evaluation audit. Tested336images, relocated21, deterministic42, rejectionofreference/evaloverlap. Unprovendiversitymode explicit, no silent baselinechange. No qualitygatepass; no100kcorpus/training claim.


D077 — Traced Courier numeric5 preview to white-background realreference but35syntheticblocks. Old colorvariance heuristic invalidly inferred rectangles. Replaced with conservative connected rectangular-color evidence; no familybranches.596profileaudit: Courier64/96triggered->0; DarkBay74/96supported; CryptBB17/96supported (recalllimitation). Fixtures/168generation checks pass. Release02/current modules fixed, v1/history preserved. BlockRNG isolated going forward; before/after randomstream transition disclosed. No classifiertraining or newpaper.


D078 — User starts V2 Tier1:5families9fonts, tiers2/3deferred. Firstmilestone presence/localization syntheticpilot + wholeimage ablation;63actualcrops/8grids with sharedCNN+positionawarelearnedpooling. No realtraining, no VLM, no classifier promotion. Exact scope/gates in V2_TIER1_PLAN.md.


D079 — Tier1pilot01 failed(nearlyallnegative/zero maskIoU). Retained failure; run02 addsbatchnorm,weighted positives,classificationweight3,32x64masks,single-grid ablation and freshseeds. Full/visible masks stored and audited. Run02multiview meanF1about0.70 but line/dotFPR~0.47/0.52; no clear multiviewgain, no gatepass. Next focus semantic/visibility/decoder/recipe quality before moredata. No classifierreplacement.


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


D084 outcome: independent curve search completed; synthetic gate=False. See V2_CURVE_SPECIALIST_RESULTS.md. No production promotion.


Curve v1 investigation: near-constant training predictions (~0.496; AP0.513 on 512 training examples), despite successful 16-image memorization control. Calibration tie chose0.99, explaining zero recall but not chance learning. See V2_CURVE_FAILURE_DIAGNOSIS.md for confirmed findings versus unproven training-design hypotheses. No production promotion; curve worker completed, dot worker remains independent. Monitor specialist status endpoints verified HTTP200 after narrow server allowlist fix. Further curriculum/localization ablations proposed, not yet run.


Started authorized finite two-stage stroke learnability trial: see V2_TWO_STAGE_STROKES.md. Reason: v1 curve classifier collapsed under whole-image supervision. Explicit masks teach localization; oracle versus predicted-mask shape tests isolate failure. Preserve dot worker, old runs, release, and paused automation. No robust success claimed.


User-requested10x3 raw RGB multicolour stroke retraining completed. Shared per-patch detector, stitched mask, no feature-pop, fresh synthetic seeds and font holdouts. See V2_STROKE_GRID_10X3.md for protocol/results. Failed diagnostic gate remains truthful; no production promotion or automatic loop. Prior full-dataset stroke inference exposed background/text mask contamination.


D085 outcome: independent dot search completed; synthetic gate=False. See V2_DOT_SPECIALIST_RESULTS.md. No production promotion.


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
