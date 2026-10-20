# Frozen classifier inference and generation: five previous families

Processed the same 128 style-reference + 40 reused evaluation images per family as Phase 1 run 05. This matched comparison covers 840 images, not every image in the full source datasets. Each image has 30 overlapping 10-across x 3-down blocks: 25,200 blocks and 75,600 classifier predictions.

Historical synthetic-trained Logistic Regression models and synthetic calibration thresholds were reused unchanged. All image identities were verified; all three model hashes match before and after every run. No real labels, fitting, recalibration or target-family generator code/fonts were used. Only the 128 reference profiles guide generation; evaluation predictions are summarized separately.

| Family | Images | Straight present | Curve present | Dots present | Generated | Render failures |
|---|---:|---:|---:|---:|---:|---:|
| cryptbb | 168 | 168 | 168 | 110 | 32 | 0 |
| city | 168 | 168 | 166 | 27 | 32 | 0 |
| courier | 168 | 168 | 168 | 9 | 32 | 0 |
| darkbay | 168 | 168 | 158 | 56 | 32 | 0 |
| outlined_text_captcha | 168 | 168 | 168 | 0 | 32 | 0 |

## Interpretation and limits

These are feature predictions, not audited prevalence or classifier accuracy. Any positive block makes the whole image positive; this aggregation is not calibrated on real images and can saturate. Positive-block fractions are not feature pixel densities.

Per-image joint states gate straight lines, curves and dots; all block probabilities and 3x10 maps are saved. Curves use generic geometry. Color/layout/background measurements remain shared CV rules. The other nine feature classes are not trained. The fixed five-character text and incomplete cross-block geometry remain limitations.

Generation uses the previous fit-only font/variant selection and identical seed/text schedules. The renderer also contains the generic rotation-containment fix, so comparison with run 05 is not a pure single-variable ablation. No Phase 2 training or generalization gate passage is claimed.

## Galleries and detailed predictions

[Five-family comparison](../runs/style_frozen_five_report_01/comparison.png)
- cryptbb: [comparison](../runs/style_frozen_cryptbb_01/comparison.png), [all outputs](../runs/style_frozen_cryptbb_01/all_generated.png), [block overlays](../runs/style_frozen_cryptbb_01/predictions.png), [partition summary](../runs/style_frozen_cryptbb_01/summary.json), [raw predictions](../runs/style_frozen_cryptbb_01/predictions.jsonl), [recipes](../runs/style_frozen_cryptbb_01/recipes.json).
- city: [comparison](../runs/style_frozen_city_01/comparison.png), [all outputs](../runs/style_frozen_city_01/all_generated.png), [block overlays](../runs/style_frozen_city_01/predictions.png), [partition summary](../runs/style_frozen_city_01/summary.json), [raw predictions](../runs/style_frozen_city_01/predictions.jsonl), [recipes](../runs/style_frozen_city_01/recipes.json).
- courier: [comparison](../runs/style_frozen_courier_01/comparison.png), [all outputs](../runs/style_frozen_courier_01/all_generated.png), [block overlays](../runs/style_frozen_courier_01/predictions.png), [partition summary](../runs/style_frozen_courier_01/summary.json), [raw predictions](../runs/style_frozen_courier_01/predictions.jsonl), [recipes](../runs/style_frozen_courier_01/recipes.json).
- darkbay: [comparison](../runs/style_frozen_darkbay_02/comparison.png), [all outputs](../runs/style_frozen_darkbay_02/all_generated.png), [block overlays](../runs/style_frozen_darkbay_02/predictions.png), [partition summary](../runs/style_frozen_darkbay_02/summary.json), [raw predictions](../runs/style_frozen_darkbay_02/predictions.jsonl), [recipes](../runs/style_frozen_darkbay_02/recipes.json).
- outlined_text_captcha: [comparison](../runs/style_frozen_outlined_text_captcha_01/comparison.png), [all outputs](../runs/style_frozen_outlined_text_captcha_01/all_generated.png), [block overlays](../runs/style_frozen_outlined_text_captcha_01/predictions.png), [partition summary](../runs/style_frozen_outlined_text_captcha_01/summary.json), [raw predictions](../runs/style_frozen_outlined_text_captcha_01/predictions.jsonl), [recipes](../runs/style_frozen_outlined_text_captcha_01/recipes.json).

Assistant visual review of the first-seed comparison: classifier gating reduces dot clutter in Courier and outlined examples, but typography, character count, interference thickness and coherent geometry remain mismatched. This review covers the displayed examples, not a human audit of all 160 images. All 160 passed the programmed rendering checks. DarkBay attempt 01 was interrupted by a Windows dashboard file lock; it remains preserved, and successful attempt 02 supplies this report. The monitor retry fix passed a simulated transient-lock check.
