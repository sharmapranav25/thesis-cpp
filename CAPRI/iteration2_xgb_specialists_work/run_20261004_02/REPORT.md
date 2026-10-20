# Specialist encoder with XGBoost replacements

All 107 image-feature outputs now use XGBoost. The exported frozen encoder executes no neural effect/text/font/length classifier and no CTC reader. CTC is preserved only in an optional separate diagnostic sidecar.

Encoder: 3,662,050 → **2,179,108** neural parameters. Exact FP32 feature equality passed (CNN 0, transformer 0 max absolute difference).

Current bundle size including tree files and the optional reader: **291.01 MiB**. Removing neural parameters does not remove the storage/compute cost of the tree ensemble.

Source checkpoint: specialist intensity merge epoch 4. The CNN/transformer stayed frozen; effects use their learned effect-branch queries, while text/font/length use their corresponding transformer queries. Synthetic train fits each booster; selection early-stops; calibration tunes effect/text presence thresholds. Font italic uses .5 across comparators. Seven binary text attributes have no ordinal targets and receive no invented intensity predictor.

The first pass stopped before any benchmark evaluation after selection showed severe ordinal regressions (font slant .579 versus neural .917). The second pass compares two fixed regression weighting choices, balanced and unweighted, per ordinal task. Choice uses higher selection within-one, then lower selection MAE, with the first candidate winning exact ties. Completed unchanged classification fits and balanced candidates are reused with hash checks. All first-pass evidence is retained; no benchmark or real result chooses the weighting.

## Overall comparison at the same .5 presence threshold

| Split / metric | Simple | Specialist neural | XGBoost replacement |
|---|---:|---:|---:|
| selection / Effect F1 | 0.4883 | 0.5387 | 0.5359 |
| selection / Text F1 | 0.5024 | 0.5457 | 0.5483 |
| selection / Effect intensity within ±1 | 0.8016 | 0.8167 | 0.8329 |
| selection / Length accuracy | 0.8450 | 0.8825 | 0.8690 |
| selection / Font category | 0.6925 | 0.7440 | 0.7375 |
| selection / Font ordinal mean within ±1 | 0.7785 | 0.7957 | 0.7710 |
| seen_test / Effect F1 | 0.4930 | 0.5403 | 0.5340 |
| seen_test / Text F1 | 0.5024 | 0.5424 | 0.5466 |
| seen_test / Effect intensity within ±1 | 0.8047 | 0.8196 | 0.8227 |
| seen_test / Length accuracy | 0.8490 | 0.8775 | 0.8795 |
| seen_test / Font category | 0.6870 | 0.7350 | 0.7380 |
| seen_test / Font ordinal mean within ±1 | 0.7869 | 0.7981 | 0.7718 |
| generalisation / Effect F1 | 0.4923 | 0.5424 | 0.5363 |
| generalisation / Text F1 | 0.5102 | 0.5470 | 0.5476 |
| generalisation / Effect intensity within ±1 | 0.8157 | 0.8263 | 0.8328 |
| generalisation / Length accuracy | 0.8532 | 0.8840 | 0.8820 |
| generalisation / Font category | 0.5656 | 0.5724 | 0.5832 |
| generalisation / Font ordinal mean within ±1 | 0.7567 | 0.7691 | 0.7665 |

## All font attributes

### selection

| Attribute / metric | Simple | Specialist neural | XGBoost | Change vs specialist (points) |
|---|---:|---:|---:|---:|
| Category accuracy | 69.25% | 74.40% | 73.75% | -0.65 |
| Stroke accuracy | 93.67% | 95.84% | 95.20% | -0.64 |
| Weight within ±1 | 80.35% | 82.60% | 80.80% | -1.80 |
| Width within ±1 | 70.65% | 73.00% | 72.85% | -0.15 |
| X-height within ±1 | 68.80% | 71.00% | 65.50% | -5.50 |
| Slant within ±1 | 91.60% | 91.70% | 89.25% | -2.45 |
| Italic F1 | 75.08% | 78.50% | 77.16% | -1.34 |

### seen_test

| Attribute / metric | Simple | Specialist neural | XGBoost | Change vs specialist (points) |
|---|---:|---:|---:|---:|
| Category accuracy | 68.70% | 73.50% | 73.80% | +0.30 |
| Stroke accuracy | 95.05% | 95.68% | 95.76% | +0.08 |
| Weight within ±1 | 80.15% | 81.75% | 79.95% | -1.80 |
| Width within ±1 | 73.70% | 75.10% | 75.10% | +0.00 |
| X-height within ±1 | 69.40% | 70.80% | 64.95% | -5.85 |
| Slant within ±1 | 91.50% | 91.60% | 88.70% | -2.90 |
| Italic F1 | 72.98% | 76.81% | 74.48% | -2.33 |

### generalisation

| Attribute / metric | Simple | Specialist neural | XGBoost | Change vs specialist (points) |
|---|---:|---:|---:|---:|
| Category accuracy | 56.56% | 57.24% | 58.32% | +1.08 |
| Stroke accuracy | 89.20% | 90.11% | 89.88% | -0.23 |
| Weight within ±1 | 80.12% | 81.88% | 80.56% | -1.32 |
| Width within ±1 | 72.04% | 72.56% | 73.36% | +0.80 |
| X-height within ±1 | 58.60% | 61.20% | 64.52% | +3.32 |
| Slant within ±1 | 91.92% | 92.00% | 88.16% | -3.84 |
| Italic F1 | 65.70% | 67.17% | 65.03% | -2.14 |

The within-one scores allow one ordinal-bin error; these are per-attribute metrics, not every-font-attribute-exact accuracy. Slant is heavily imbalanced and duplicates a text target. Stroke class 2 has no measured examples in the training corpus. These limitations remain.

## Calibrated presence comparison on reused benchmarks

| Benchmark / mean F1 | Simple | Specialist neural | XGBoost |
|---|---:|---:|---:|
| seen_test / Effects | 0.5854 | 0.6198 | 0.6127 |
| seen_test / Text | 0.5905 | 0.6312 | 0.6213 |
| generalisation / Effects | 0.5966 | 0.6283 | 0.6242 |
| generalisation / Text | 0.5893 | 0.6213 | 0.6166 |

All three presence methods use the same calibration rule: maximum F1 subject to FPR <= .10. Neural comparison uses saved reports, without rerunning or retuning those checkpoints. The simple calibrated presence report was previously recorded by iter2_xgb_01 for the identical baseline checkpoint. Its arithmetic differs slightly from the baseline original report; it is labelled separately from the .5-threshold table.

The seen_test/generalisation sets were already observed in prior experiments and are reused benchmarks. They were evaluated for this replacement after all heads were fitted and were never used for early stopping, configuration changes or per-head fallback. Every requested image-feature output is boosted even if weaker; all regressions remain in per_head_comparison.csv/json.

- selection: versus simple, 89/107 improve and 13/107 regress; versus specialist neural, 46/107 improve and 58/107 regress. Presence uses .5 on selection and equal calibration on benchmark per-head comparisons.
- seen_test: versus simple, 88/107 improve and 19/107 regress; versus specialist neural, 51/107 improve and 55/107 regress. Presence uses .5 on selection and equal calibration on benchmark per-head comparisons.
- generalisation: versus simple, 88/107 improve and 19/107 regress; versus specialist neural, 45/107 improve and 62/107 regress. Presence uses .5 on selection and equal calibration on benchmark per-head comparisons.

## Fit completion and inference checks

Two passes were fitting configurations, not two epochs. Each boosted model had a 2,000-round limit with 50-round early stopping. This was a limited comparison of fixed settings, not an exhaustive parameter search.

Final inference verification caught an italic output-decoding error: converting a probability to an integer before bool conversion produced false. Both deployed and batch inference now use probability >= .5 through the same decoder. Fitted boosters, cached vectors and synthetic benchmark scores were unchanged. Initial inference/profiles/previews are retained in inference_initial_italic_decode; all 3,599 corrected predictions and all 300 previews were regenerated.

Verification passed: 107 outputs, 107 saved boosters, exact benchmark-score replay, source-checkpoint hashes, all feature-cache hashes, correct task routing and no neural prediction modules. All 300 preview pixel hashes, character counts and sampled prototypes passed. Selected boosting rounds: median 323, maximum 1578; 0 models selected the 2,000-round limit.

Both workspace and published project CLI ran successfully; a real italic-positive example decoded true. Checks and provenance: verification.json, replacement_checks.json and bundle/checksums.json. The bundle includes the exact runtime source snapshot.


## Target-family previews

| Family | Generated | Strict readability | Mapping distance |
|---|---:|---:|---:|
| City | 100 | 81/100 | 0.0208 |
| CryptBB | 100 | 84/100 | 0.0837 |
| DarkBay | 100 | 78/100 | 0.0627 |

All 3,599 real images were inferred separately (CryptBB 600, City 2,000, DarkBay 999), using new boosted predictions. Only identity/source fields were reused; real answers and old predictions never fit a model. Renderer usability is separate from strict training readability and target resemblance. No CAPTCHA-solver training was launched.

The three corrected sheets were inspected against the existing target examples. City captures five digits and horizontal bands but has broad colour/font variation and extra curves. DarkBay often has six characters and block backgrounds but varies fonts, case, colour and layout beyond the target. CryptBB now produces more five-character strings and heavier glyphs than the specialist neural previews, yet still spreads across grids, monochrome noise, hollow glyphs and short strings. These are qualitative observations, not measured real-family accuracy. Renderer palette/position defaults and prediction errors both contribute; greater readability alone does not establish closer family resemblance.

![CryptBB previews](dataset/CryptBB_generated.jpg)

![DarkBay previews](dataset/DarkBay_generated.jpg)

![City previews](dataset/City_generated.jpg)

