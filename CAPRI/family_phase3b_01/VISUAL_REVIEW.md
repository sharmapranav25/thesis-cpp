# Assistant visual review

First four fixed-order report samples per family, or all available when fewer. No VLM in training/scoring. This is qualitative assistant inspection, not recognition accuracy.

- CryptBB: substantial background coverage changes and conspicuous borders/halos around the unchanged text. Reject visual improvement despite moment reduction.
- City: produces gray halos around glyphs rather than the target's stepped horizontal background bands. Reject.
- Courier: changes background areas and produces glyph-shaped contours. Baseline already contains block patterns in some examples before this GAN; these are not evidence of shared GAN weights. Reject.
- DarkBay: broad palette remains recognizable, but block/edge changes introduce outlines around glyphs. No clear improvement.
- Outlined: spills existing glyph colors across background regions and does not recover the correct outline font. Existing RGB membership passes but perceived color coverage changes markedly. Reject.
- Roncuevas: conspicuous thick contours around the protected glyphs; does not recover the family-specific interference/layout. Reject.
- Pymlex: only one report example; introduces glyph halos and fails the target visual style. Reject; tiny sample cannot establish generalization.
- Sloth: texture changes are small and the baseline hollow-font mismatch persists. No convincing gain.

All eight panels were inspected. Posthoc DINO results must be read alongside this visual rejection; training/selection moment improvement did not establish perceptual success. Background patterns present in the baseline remain possible because the general Phase1 renderer is shared; independent Phase3B weights do not repair all inherited errors.

The first six family results were completed before a small-cohort split assertion stopped the job at Pymlex. Fixed the four-image split to the predeclared2/1/1 and resumed only the two unfinished families. Previous results/checkpoints preserved. Final status elapsed seconds measures resumed segment, not total runtime.

No default promotion. A no-new-RGB assertion is not sufficient to guarantee unchanged perceived color. The reported exact character pixels are at192x64 after the same nearest-neighbor resize used for the baseline; glyph identities are synthetic labels, not independently recognized output text.
