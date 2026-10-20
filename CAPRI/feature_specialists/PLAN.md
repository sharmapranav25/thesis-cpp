# Feature specialists: Tier 1 reset

2026-09-28. User authorizes independently trained classifiers, feature-specific pop, measured view selection, robust training. This supersedes the shared five-head pilot as the development direction. Keep all old runs and the current Phase1+fonts release. Five real inference cohorts and nine independent fonts remain the scope. No real classifier fitting, runtime VLM, target generator/fonts, PARSeq, Phase3 or continuous automation restart.

## Feature sequence and meaning

Start with curved interference (smooth non-text strokes). Then straight interference, dots, periodic grids/bands, background blocks/shapes, procedural texture, font effects and image-quality/distortion effects. Every feature requires a separate definition, generator/negative audit, training weights, pop candidates, view search and acceptance record. Font identity/geometry/color/density are not all interchangeable binary labels. Untouched specialists are queued, not implemented. This package currently implements CURVES ONLY.

## Curves corpus

6000 training parent groups,1000 selection,1000 calibration,500 each locked same-distribution/font/shape/rasterizer tests. Two images per group: same background/text/clutter with a smooth curve versus a straight interference stroke. 20000images total. Pair members stay in one split. Independent seed ranges; six train fonts, selection includes those plus seventh, locked font split uses last two. Text is random2..8alphanumeric and is nuisance, not OCR training material.

Three training curve recipes: sine, circular arc, quadratic Bezier. Uniform12orientation bins; variable scale, thickness,contrast,position,palette,background blocks/gradient/grids/dots, character rotation and occasional paired blur. Curves can appear in front of or behind text. Cubic Bezier is a heldout shape; OpenCV polylines is a heldout rasterizer (Pillow used for training). New rasterizer does not mean independent-source CAPTCHA data. No target-family generator used.

Paired counterfactual masks identify final visible changes. Reject and log targets with insufficient final contrast/visibility or nearly straight geometry; never silently label invisible requested curves positive. Conditions audited before training. Ambiguous heavily occluded cases excluded here are NOT claimed solved. Positive and negative support lengths/counts may differ; pairing reduces but does not eliminate generator shortcuts. No full independent library-source generalization claim.

## Pop and view search

Curve-pop candidate: label-independent grayscale local contrast magnitude using Gaussian local mean/std (noise floor), concatenated with originalRGB. Raw remains available; no groundtruth-derived pop. Both polarities treated symmetrically. Population statistics never fit from evaluation images.

Compare raw and RGB+pop for each of1x1,2x1,3x1,1x2,2x2,3x2,5x3,10x3. Same per-image batches/steps/seed, but not matched FLOPs. Spatial4x8 encoder pooling preserves structure instead of collapsing every crop to one average. Coordinates accompany crop representations. Select by selection-set AP, never locked test. Screen up to12epochs, patience4 after minimum6. Rank raw/pop jointly. Then test a combination of the top two distinct grids, with the pop choice of the top individual candidate. Final chosen configuration retrained independently with seeds431,432,433 up to24epochs; only these confirmation models count toward robustness.

## Gates fixed before training

G0: code/annotation tests; pair split/seed integrity; per-image visible support; nondegenerate curve geometry; diverse orientations; fonts hash-checked; original/vendor untouched. No training if checks fail.

G1: screening is configuration selection, not acceptance. Keep every result including failures and resource cost. Feature-pop is optional and retained only if supported by selection evidence.

G2: thresholds per confirmation seed chosen ONLY on calibration to maximize F1 subject to empirical FPR<=.05. No eligible threshold => failure, not silent relaxation. Locked tests: EACH confirmation seed and EACH test stratum must attain F1>=.90, recall>=.90,FPR<=.05. Also report95%Wilson FPR upper bound, AP, all counts and per-orientation/contrast breakdowns. These are engineering targets, not previously demonstrated performance. Consuming a locked test makes it development evidence for later revisions; never call it unseen again.

G3: synthetic acceptance alone does not authorize deployment. Frozen real inference + existing CV across five families; audited real accuracy still pending. Renderer controls/parameter estimation and actual reconstruction comparison need their own checks. A classifier score is not curve count/tilt/geometry. Current release stays unchanged.

## Execution

Run `train_curves.py --prepare-only` for corpus/audit, then `train_curves.py` to resume preparation or run finite search/confirmation. Run folder is append-only by experiment: existing protocol hash must match; interrupted checkpoints resumed by completed candidates only (incomplete candidate retrained with logged event). PID lock prevents duplicate jobs. StatusJSON and JSONL events update during work. Local monitor updated when canonical project is writable. Stop on failed corpus gate or recorded exception. No automated endless loop.

This is original engineering; no new paper directly used. Record exact external methods if adopted later. Classifier-only stage does not claim density/localization or real-time generation solved.
