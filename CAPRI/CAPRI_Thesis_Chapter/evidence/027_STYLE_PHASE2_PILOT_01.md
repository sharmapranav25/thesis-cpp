# Phase 2 pilot 01 — recipe learning and image refinement

Exploratory run explicitly authorized after Phase 1 limitations were discussed. Historical Phase 1 gates were not rewritten as passes. Feature-classifier hashes remained unchanged; no real-label feature training or recalibration occurred.

## Protocol

Five families: 96 real fit images and 32 development images per family. Final evaluation used 40 additional historical validation images/family (hash positions 80:120), opened after a selection-freeze record was saved. These pools have historical exposure and are not claimed globally pristine. Images in the reserve did not train or select this pilot. All models were trained locally from scratch using existing open-source PyTorch/Pillow/OpenCV components and audited generic fonts.

Recipe search evaluated 32 bounded parameter vectors. CNN and transformer predictors trained for 250 steps to imitate search-selected recipe vectors, rather than backpropagate through the renderer. A 200-step local discriminator ranked the same candidates; this is adversarial feedback, not an end-to-end recipe GAN. Direct CNN, transformer and GAN models each trained for 400 steps. Direct refiners preserve native synthetic text pixels and a one-pixel safety band; fonts, text count and glyph shape cannot be corrected by these models.

## Reserved evaluation

Distance combines low-level color moments, multiscale gradient statistics and spatial color profiles. Positive improvement means lower distance than the unchanged classifier-guided baseline. It is not recognition accuracy or perceptual ground truth. Training and reporting use related low-level descriptors, so visual review is essential.

| Method | Mean-distance improvement | Families improved | Minimum diversity ratio | Render failures |
|---|---:|---:|---:|---:|
| recipe_search | 33.2% | 3/5 | 0.89 | 0 |
| recipe_cnn | 34.7% | 3/5 | 0.90 | 0 |
| recipe_transformer | 31.6% | 3/5 | 0.89 | 0 |
| recipe_discriminator | 36.5% | 3/5 | 0.86 | 0 |
| direct_cnn | 62.6% | 5/5 | 0.64 | 0 |
| direct_transformer | 61.5% | 5/5 | 0.51 | 0 |
| direct_gan | 61.9% | 5/5 | 0.69 | 0 |

| Family | Search recipe | CNN recipe | Transformer recipe | Discriminator recipe | Direct CNN | Direct transformer | Direct GAN |
|---|---:|---:|---:|---:|---:|---:|---:|
| cryptbb | 34.2% | 50.6% | 38.5% | 33.3% | 65.3% | 69.8% | 63.2% |
| city | 75.7% | 76.2% | 77.4% | 75.7% | 73.6% | 81.9% | 72.1% |
| courier | 21.9% | 23.6% | 25.5% | 36.2% | 62.7% | 49.1% | 62.4% |
| darkbay | 0.0% | -19.7% | -26.8% | -66.3% | 52.6% | 53.5% | 53.0% |
| outlined_text_captcha | 0.0% | -86.0% | -88.7% | 0.0% | 6.9% | 50.2% | 14.6% |

Total pilot wall time: 2.53 minutes. Numeric eligibility does not constitute a final success claim; visual review is recorded separately.

## Artifacts

[Complete metrics](../runs/style_phase2_pilot_01/results.json) · [Training log](../runs/style_phase2_pilot_01/training.jsonl) · [Frozen selections](../runs/style_phase2_pilot_01/selection_frozen.json) · [Candidate failures](../runs/style_phase2_pilot_01/search_failures.json)
- [cryptbb before/after gallery](../runs/style_phase2_pilot_01/cryptbb_comparison.png)
- [city before/after gallery](../runs/style_phase2_pilot_01/city_comparison.png)
- [courier before/after gallery](../runs/style_phase2_pilot_01/courier_comparison.png)
- [darkbay before/after gallery](../runs/style_phase2_pilot_01/darkbay_comparison.png)
- [outlined_text_captcha before/after gallery](../runs/style_phase2_pilot_01/outlined_text_captcha_comparison.png)

Each gallery shows references, unchanged baseline, search recipe, learned recipes, then direct refiners. These are first-seed examples, not selected best images. Full images, masks and recipes are retained by family/method. Checkpoints at direct-training steps 100/200/300/400 are retained; only fixed final step 400 was evaluated, with no reserve-driven checkpoint selection.

Unresolved limits: synthetic-trained feature predictions saturate on real images; five-character layout remains fixed; no real text labels, family fonts or copied reference patches were used. Recipes can only express effects supported by our renderer. Realistic appearance is not guaranteed by exact glyph-pixel preservation. No production or recognition-accuracy claim.

## Assistant visual review

Reviewed the first four seed-ordered outputs per method in all five galleries. Recipe methods are exploratory candidates: City gains stronger line structure, but color tint and font/layout mismatch remain; CryptBB/Courier results do not establish uniformly improved resemblance; DarkBay/outlined recipes largely preserve the baseline. No exact-resemblance claim.

Direct CNN, transformer and GAN refiners produce conspicuous pale halos around the protected characters and shift backgrounds too broadly. **Reject visual success for all three direct refiners in this pilot**, despite their 61–63% lower descriptor distance. Exact glyph-pixel preservation does not guarantee unchanged perceptual style. Numeric improvements alone are insufficient.

The final evaluation window is now exposed reporting data. Future fitting and selection must stay in fit/development pools; do not recycle these results as an untouched test. All checkpoints and this unsuccessful direct-refinement result remain preserved. The first end-to-end comparison is complete, not the entire research objective.
