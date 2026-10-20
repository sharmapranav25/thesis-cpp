# CryptBB whole-reference analysis

**OBSERVED:** User stopped the five-dataset run and requested CryptBB only. All 600 registered, unique CryptBB references had already completed; their hash membership was checked against the historical fit+validation manifest and copied into a separate result directory. The prior exclusion of external eval images remains intact. All full-run workers were stopped. Other datasets were not completed.

[Prediction panels](../../runs/feature_bank_cryptbb_reference_01/predictions.png) · [Generated examples](../../runs/feature_bank_cryptbb_reference_01/cryptbb_generated.png) · [Counts](../../runs/feature_bank_cryptbb_reference_01/summaries.json) · [Recipes](../../runs/feature_bank_cryptbb_reference_01/generation_recipes.json).

## Predictions, not measured accuracy

| Feature | Images predicted present | Fraction of 18,000 bricks predicted present |
|---|---:|---:|
| Straight interference | 600/600 | 99.45% |
| Curved interference | 600/600 | 99.94% |
| Dot scatter | 367/600 | 11.56% |

Dots were predicted absent in 226 images and uncertain in 7. An image is positive if any brick is positive, absent only if every brick is absent, otherwise uncertain. This aggregation has not been calibrated at image level. Overlapping brick fractions are diagnostic counts, not image area or density estimates.

**OBSERVED:** Original curve/dot threshold selection rejected overlapping independently selected ranges despite feasible separated pairs. The new search considers all separated candidate pairs satisfying the same calibration precision/NPV >=.90 and support >=30 constraints, maximizing calibration coverage. Only saved synthetic calibration scores/labels were used; model weights and original bundles were preserved. New thresholds and model hashes are saved separately. These are in-sample calibration statistics, not established real-data confidence guarantees.

**OBSERVED by assistant:** Eight reference examples, selected by lowest pixel hash, show complex multicolor texture, blocks, grids and varied lettering. Predicted line/curve boxes cover almost every brick, including block-like textured areas. This saturation makes the current three-feature output insufficient to distinguish CryptBB styles. It suggests poor transfer and confusion with unsupported structures; without audited labels, an error rate cannot be assigned.

## Limited generated demonstration

**OBSERVED:** Produced 32 images using the existing Python text renderer, audited library fonts and the instrumented Pillow straight/curve/dot functions. Each recipe samples a reference's predicted positive combination and native dimensions; its median RGB supplies a flat background. Text is generic five-character text, not recognized or inferred from references. Density, tilt, font identity, texture, blocks, grid structure and letter transformations are not inferred. Uncertain features are not rendered and are not relabeled absent in prediction records.

All 32 images passed a per-glyph contribution check; **PASS on the gallery means text visibility only**. Assistant inspection of all 32 examples shows a substantial visual mismatch with the reference examples. They are feature-toggle demonstrations, not successful CryptBB recreation or verified family matching. PHP renderers were not used in this batch.

**Conclusion:** Descriptive inference and limited generation completed; visual-style matching failed. Do not proceed as though the line/curve predictions are reliable descriptions of CryptBB. Missing feature coverage and transfer validation must be addressed before claiming useful recreation. Real references have now guided generation and cannot be called untouched test inputs for this workflow. No target labels, generator source or target fonts were used, and no real accuracy was measured.

The descriptor padding-mask pooling was vectorized for this run. Five seeded shape/input checks produced bit-identical descriptors against the frozen training implementation. Earlier runs and source snapshots remain intact. Current artifacts are in the local staging workspace; this report does not assert publication into the canonical E: project.
