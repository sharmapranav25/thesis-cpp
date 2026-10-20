# Curve specialist v1 failure investigation

2026-09-28. No production changes or retraining of the historical run. Dot specialist continues independently.

## Confirmed findings

- The failure begins during training, before real-image inference. Confirmation seed 431 training BCE stays near 0.693 (balanced chance baseline); selection AP peaks at about 0.507. Best screening AP is 0.510. View search did not solve it.
- Replaying the saved confirmation checkpoint on the first 512 training images gives AP 0.51255, scores 0.496124–0.496196, standard deviation 0.00000868. This is an almost constant predictor, not merely an overly strict evaluation gate.
- A controlled memorization check with the same architecture, RGB+pop, balanced first 16 training images and optimizer reaches 100% training accuracy by step 100 at both learning rates 0.0007 and 0.0001. Thus this path can compute gradients and fit these examples. This does not establish full-corpus learnability or generalization, and is not a validation score.
- Corpus code stores [negative, positive] and [0,1] in the same order; training applies the same shuffled indices to inputs and labels. No label-order mismatch was found in this review. Existing corpus audit checks stored alternating labels and disjoint paired groups.
- Calibration scans thresholds in increments of 0.01. Near-constant scores cannot provide useful separation. All eligible thresholds had zero F1 for seed 431; Python tuple maximization broke the tie in favor of 0.99. Therefore all locked examples were rejected, producing zero recall. The gate correctly failed. Threshold selection should explicitly report no useful operating point and use exact score breakpoints in a future version, rather than presenting 0.99 as meaningful calibration.

## Likely training-design weaknesses (not yet isolated causal results)

- Full nuisance difficulty from the start: curved glyphs, grid lines, random lines/dots, random colors, partial occlusion and blur. The actual non-text curve can occupy a small fraction of the image. Negatives legitimately contain curved letters, so merely detecting any curvature is insufficient.
- Synthetic target masks are used for visibility filtering but discarded from the saved training targets. Only one whole-image binary label supervises the network. It must discover localization and distinguish text from interference without explicit localization training.
- Feature-pop is generic local contrast; it highlights text and noise as well as the target. It is not a demonstrated curve-specific enhancement.
- Fine grids cut long curves into almost straight fragments. Pooling learns weights for separate crop tokens but has no explicit neighbor-to-neighbor tangent continuity objective. Coordinates alone do not demonstrate that the model combines fragments into a curve.
- Search repeats this weakly supervised setup across many grids before establishing a successful learnability/curriculum baseline. More parallelism does not repair that issue.

## Recommended next experiment

Create a new version, preserve v1. First demonstrate clean curve-versus-line learning, then add text, nuisance effects and occlusion in controlled ablations. Save synthetic visible masks and supervise localization/tangent geometry as auxiliary training targets; inference still uses images only. Include paired examples in minibatches and test an explicit paired ranking objective as an ablation. Keep full-image context alongside local detail; evaluate a geometry-aware pop against raw RGB. Stop before broad view search if training/selection remains at chance. These are proposed tests, not fixes already proven to work. Retire consumed locked splits to development and create fresh locked seeds before final evaluation.

## Evidence

Local investigation script: diagnose_curves.py. Raw diagnostic output: curve_diagnosis.json. Historical evidence: feature_specialists_v2/runs/curves_v1/confirm_431_history.json, view_selection.json, locked_results.json and confirm_431.pt.

No external paper was consulted or new paper-based method implemented for this investigation.
