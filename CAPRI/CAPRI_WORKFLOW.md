# CAPRI workflow

Agreed workflow for turning 100 unlabeled target CAPTCHA images into a synthetic domain and then 100,000 rendered training images. The first implementation is in `capri_darkbay_workflow/`; its run artifacts record completion and audit status. It does not replace the older frozen CAPRI package.

The purpose of multiple controllers is diversity around the target scheme. They share the same reconstruction objective but approach it differently. A controller does not have to outperform the others to belong in the mixture; preserve their complementary outputs instead of selecting one overall winner. Actual output diversity and downstream benefit still need measurement.

## 1 Hybrid inference and immediate refinement

The hybrid processes 100 images from one target scheme individually. For each image, save its raw inference, predicted CAPTCHA length, and renderer settings. Immediately clean and refine that image's settings in code, preserving both the raw and refined versions and the changes made.

Keep all 100 refined recipes. Do not collapse them into one average family recipe.

## 2 Scheme length rule

Collect the hybrid's predicted lengths across the 100 images.

- If one length has a clear majority, make it the scheme's default length.
- If there is no clear majority, give each distinct observed length equal probability, rather than using its observed frequency.

The first DarkBay run uses the proposed threshold of at least 60 of 100 votes. Fresh hybrid inference gives length 6 in 72 of 100 cases, so the code derives 6 as the default. It does not hardcode the known family length. Vote agreement is not calibrated accuracy. Apply the rule consistently through controller refinement and final generation; the current DarkBay executable stops if no majority exists rather than silently inventing a default.

On10October2026 the user requested bounded first6 controller examples per scheme. `controllers --preview` now applies this fallback for those previews using seeded shuffled cycles over the observed lengths, shared across all three arms. Full100-input reference distributions remain in the comparator. This preview integration does not enable uniform-length corpus generation; standard full runs retain the no-majority stop.

## 3 Three controllers

Send the same 100 refined recipes, with access to the target pixels, to all three controllers:

- Our custom controller, currently represented by MVP4.
- Our existing DINO controller, currently represented by DINO v2 in MVP7. This is not DINOv3.
- Our mixed custom and DINO controller, represented by MVP10, using both comparators and combinations of their renderer settings. This is the accepted third controller for diversity, even when it does not improve the aggregate similarity score.

Each controller refines the 100 recipes and produces one example image per final recipe. Preserve the final renderer settings and their connection to the input recipe and controller.

The result is 300 final recipes and 300 corresponding example renders, 100 from each controller. These are 300 recipe entries, not a guarantee of 300 distinct settings: the mixed controller can sometimes retain another controller's settings. Preserve provenance and measure duplication. SAM and Meta-Sim2 are not part of this three-controller workflow.

## 4 Build the synthetic domain

The 300 final recipes collectively define the CAPTCHA domain. Use the renderer to generate 100,000 fresh images from this mixture, with new synthetic strings and random seeds while following the scheme length rule.

The example images illustrate the recipes; they are not the source images to duplicate into the training corpus. Save the generated text labels and recipe provenance alongside the fresh renders.

The DarkBay run uses 333 or 334 images per recipe, with seeded remainder allocation: 33,334 custom, 33,333 DINO and 33,333 mixed images. Keep contributions from all three controllers rather than weighting solely by which has the best similarity score. Retry failed renders within the same recipe and record the rejection; never drop a recipe to improve the apparent validity rate.

## Constraints and remaining decisions

Target input is pixels only: no supplied target strings, character positions, masks, or texture labels. Hybrid predictions and pixel-derived estimates are allowed. Synthetic renderer labels are known. No VLMs.

The first implementation records every raw prediction and per-image cleanup. Cleanup uses highest-probability selection within predefined head groups, suppresses selected combinations as a simplifying heuristic, zeroes inactive levels, clamps generic renderer ranges and applies the model-derived length. These heuristics do not prove the removed combinations are absent in the target.

Each controller scores fresh renders using equal contributions from the corresponding target image's appearance and the full 100-image distribution. Custom and DINO use their own frozen comparators; mixed uses their calibrated equal-weight combination. Selection validates candidate rendering on three new synthetic seeds, against the same target references. This is not an unseen-target evaluation. Generic recovery is logged if no shortlisted recipe passes the renderer's completeness and usability guards.

The corpus is stored as 300 ZIP shards, one per recipe entry, with PNG images, synthetic text labels, seeds, recipe provenance and rejection logs. The final audit decodes every PNG, checks unique image hashes and seeds, verifies model-derived length and balanced allocation, and checks for copied target pixels. Renderer usability is not human legibility or solver accuracy. Reconstruction quality and downstream solver benefit remain separate questions.

## First completed run: DarkBay, 2026-10-05

The first run produced 100,000 distinct images from all 300 entries, with the allocation above. All have the model-derived length 6; no generated RGB hash matches a target input. There were 76 rejected render attempts, retried within their original recipes. All 300 controller entries passed pilot completeness/basic usability without generic recovery.

There are 179 distinct parameter signatures after excluding sampled text and seeds, not 300 distinct families. Custom and DINO chose identical parameters for 31/100 inputs; mixed matched custom for 67 and DINO for 54. All controller contributions remain in the mixture.

Only 36,253/100,000 images pass the stricter per-character readability guard. In the 300-image pilot, low local character contrast explains nearly all strict failures. This is an experimental corpus with an explicit quality flag, not a claim of training-ready quality or successful solver transfer. See `capri_darkbay_workflow/report.html`, `corpus_audit.json`, `dataset_manifest.json` and the scripts for evidence and reproduction.
