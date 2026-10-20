# Curve regression debugging and controlled experiment05

## Observed evidence (post-hoc development benchmark51m)

03 curve AUC0.9088 versus04 specialists0.9060: ranking discrimination is similar despite0.5-threshold F1 regression. Negatives' mean curve score rises0.355->0.437. Scores are raw weighted-loss outputs, not calibrated confidence. A threshold-only explanation is insufficient without separate calibration testing; failed03 calibration is not deployed.

Specialist curve false positives: clean/text-only4/58(6.9%); straight-without-curve69/97(71.1%); grain-without-curve68/93(73.1%). Subgroups overlap other effects; these are associations, not proof of one feature causing confusion. Exact scores/seeds are in diagnostics.json and first12 error examples in curve_false_positives.png.

Local curve head AUC falls0.845(1x1)->0.807(10x3). Geometry audit restricted to30single-sine-curve images:340/341eligible positive10x3crops have centerline maximum residual<=1native pixel from fitted straight line;5x3:150/198;1x1:0/30.11small10x3crops have insufficient x extent and are reported separately. This supports local ambiguity but is not a general perceptual label audit: full masks, one simple curve type, no multi-curve disentanglement or visibility guarantee.

Frozen sensitivity test: forcing every local curve score to0.5 changes final curve probability by mean0.00221; F1 .72184->.72350. The fusion has a strong bypass through48appearance features, so explicit head scores contribute little in this probe. Learned grid weight .10-.16 per grid does not prove useful semantic fusion. Coarse-only/fine-only pooling at inference both slightly worsen F1; inference interventions shift activation distributions and do not establish retrained performance.

## Controlled retraining

05 uses EXACT04 specialist architecture, training data, initial RNGseed, minibatch order, optimizer,1800steps and global checkpoint-selection rule. InterventionA no_local_aux removes ALL local auxiliary loss but retains heads/global gradient paths. InterventionB coarse_curve_aux removes ONLY curve-loss terms in5x3/10x3grids, keeping all other terms and original averaging denominator; no change to global curve labels or fusion. It does not relabel small curves as straight. Other local features retain full grid supervision. Missing fine-curve supervision makes these heads exploratory, not complete deployable specialists.

Fresh frozen evaluation61m same,62m font shift,63m curve recipe shift;420each. Original03 and04checkpoints evaluated on the identical new images. Same generator/training seed remains a limitation. No real training;20exposed real fit references infer only for coarse-curve variant, existing CV profiles retained. No runtime VLM or external paper used. Preserve all histories/default pipeline. No automatic promotion, no PARSeq or Phase3. Automation paused.

## Fresh results

| Model | Same curve F1 / FPR | Font curve F1 / FPR | Recipe curve F1 / FPR |
|---|---|---|---|
| original03 | 0.741 / 0.388 | 0.702 / 0.371 | 0.672 / 0.329 |
| original04 | 0.754 / 0.456 | 0.689 / 0.480 | 0.705 / 0.443 |
| no_local_aux | 0.793 / 0.262 | 0.723 / 0.270 | 0.681 / 0.260 |
| coarse_curve_aux | 0.762 / 0.435 | 0.698 / 0.488 | 0.722 / 0.451 |

## What the intervention establishes

No-local-aux improves curve F1 .754->.793 and FPR .456->.262 on fresh same-recipe; font-shift .689->.723, but curve-recipe shift FALLS .705->.681. Fine-curve masking yields .762/.698/.722 across same/font/recipe, only modest changes. Thus ambiguous fine labels are real, but deleting them alone does not solve the behavior. Removing all auxiliary changes supervision for all five features, so do NOT attribute its gains exclusively to curve-loss gradient conflict. Single initialization, different selection checkpoints: suggestive controlled evidence, not robust causal proof. Other features can regress (no-aux fresh straight F1 .572 versus original04 .608); no model promotion.

A separate diagnostic uses420new synthetic calibration examples at60m, six training fonts, maximizing curve F1 on0.05..0.99 thresholds before application to61m/62m/63m. Original04 selectedthreshold0.75: fresh FPR falls .456->.224 but recall .962->.770 and F1 .754->.748. It largely exchanges false positives for misses. No-aux threshold0.52 gives F1 .787 versus fixed .793. Fine-masked threshold0.68 gives F1 .765 versus .762. Calibration is not a complete fix and is not deployed. Full thresholds/all-shift outcomes in curve_calibration.json.

Direction supported for next implementation: small grids should represent locally observable stroke fragments/geometry with unknown curvature when insufficient context, rather than always predict parent-effect identity. Larger/neighboring context should decide curvature. Make explicit specialist evidence useful in fusion and retest with feature-specific supervision ablations. This is proposed next work, not already implemented. Preserve original defaults; findings are synthetic, not established real curve accuracy.
