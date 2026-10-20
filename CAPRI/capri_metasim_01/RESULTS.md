# CAPRI Meta-Sim-inspired trial results

Exploratory12step per-family adaptation, not full Meta-Sim reproduction. Existing seed731 hybrid frozen; no VLM or real labels. Source paper: https://arxiv.org/abs/1904.11621 .

| Family | DINO MMD reduction | Independent CV before | CV after | DINO delta95% bootstrap interval |
|---|---:|---:|---:|---|
| cryptbb | 3.79% | 1.862 | 2.185 | [-0.026103849709033965, 0.0037022337317466477] |
| city | 4.33% | 1.405 | 1.260 | [-0.035364757478237155, -0.010660524666309372] |
| courier | 7.20% | 0.774 | 0.514 | [-0.0356021985411644, -0.017177248001098645] |
| darkbay | 27.71% | 1.020 | 0.721 | [-0.12951883524656296, -0.08815990984439853] |
| outlined_text_captcha | 5.96% | 0.845 | 0.685 | [-0.05264037549495697, -0.029414717853069306] |

Lower is better for both distances. Bootstrap interval pertains to this chosen adapter and sampled data, not training-seed uncertainty. DINO is also the fitting objective, so improvements there are not independent quality evidence.

480 reference images total:48fit/16selection/32report per family, historically exposed. No report recipes feed generation. Font selection had prior broad exposure, hence no pristine holdout claim.64fresh paired generations/family=640saved images. All final glyph identity/geometry/RGB assertions and renderer visibility checks passed; see per-family fallback counters. Fonts, text length, palette, gradient, glyph layout were immutable.

Finite-difference plateaus remain: integer widths/counts or absent primitive effects can produce zero gradients. Twelve steps/one adapter seed is a bounded pilot, not exhaustive optimization. This does not measure PARSeq transfer or OCR utility. Do not promote automatically.

## Assistant visual review

All five deterministic first-six galleries inspected. Darkbay shows more rectangular clutter and fewer dots, but some new lines/blocks still do not resemble the target layout. Courier has fewer dots and more lines, with major font, stroke and spacing mismatch remaining. City changes are small and do not reproduce thick stepped bands. Outlined removes some noise but does not fix outlined glyphs. CryptBB adds some blocks/reduces clutter; independentCV worsens and the DINO bootstrap interval includes zero. Mixed limited evidence, not an overall visual pass.

Two Darkbay final candidates failed visibility checks and returned unchanged baselines; remaining families had zero final fallbacks. These are not hidden successes. No OCR validation conducted.
