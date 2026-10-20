# Simple-hybrid-assisted renderer adjustment

The frozen epoch-15 EMA of `iter2_baseline_15_20261003_04` was freshly run on CPU over all 3,599 target sources: CryptBB 600, DarkBay 999, City 2,000. The checkpoint SHA256 remained `ae73375ead99071d55c6394f7c4eb7c4d27c0a3285f8f15d8d84c9df43bc6b32`. Presence used the same raw 0.5 threshold as the earlier simple-hybrid previews. No old predictions or real answers entered the new inference.

For the adjustment trial, one reference per family was fixed using the first reference in each saved V1 manifest, matching the references inspected in the preceding manual demo. The current controller mapped the new simple-hybrid predictions to initial recipes, including its measured-label feedback mapper. Initial images and recipes were retained. Manual edits then used the same labelled captcha-effects canvas and text renderer as the preceding demo.

| Family | Useful starting information | Main manual corrections | Assessment |
|---|---|---|---|
| City | Five digits; monospace/non-italic font; weight, width, wide spacing; bands and pixelation | Palette, shallow band geometry, text size, rotation and vertical jitter | Clearly useful starting point; one adjustment pass |
| DarkBay | Six characters; non-italic, zero-slant text; weight and width controls | Tall rectangular background, alternating black/white colors, proportional sans font, digit allowance, spacing and geometry | Partly useful; background still required substantial manual work; one adjustment pass |
| CryptBB | Grid and border | Two digits became five mixed characters; replace thin italic/hollow style; rebuild fine pastel clutter and colored heavy glyphs; correct geometry | Little overall reduction in manual work; much of the recipe was replaced; two adjustment passes |

This is a qualitative effort assessment from one preselected example per family, not a timed blind study or a downstream recognition benchmark. The prior manual demo supplied familiarity with the target families and reusable drawing controls. Generated text is fresh rather than a reconstruction of the target answer. Renderer legibility checks passed for all three adjusted examples.

`predictions.jsonl` and `inference_report.json` contain the full-family inference. `*_starting_point.json` preserves the model inputs to rendering; `*_adjustments.json` records the manual changes. Each `*_comparison.png` shows target, simple-hybrid render and manually adjusted render from left to right.

These artifacts are separate from specialist training and do not alter its data, losses, checkpoint-selection rule, or renderer package.
