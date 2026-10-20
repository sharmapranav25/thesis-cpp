# Phase 3 font experiment 02 and external preview test

Continued synthetic-only training from the selected step-700 checkpoint to step 2,000. The synthetic selection set selected step 2,000. A fresh synthetic test range was used; both frozen models were then evaluated on the same 768 test images for a fair comparison.

| Frozen font model | Top-1 | Top-3 | Filled/hollow |
|---|---:|---:|---:|
| Run 01, step 700 | 45.70% | 77.47% | 89.06% |
| Run 02, step 2,000 | 56.25% | 84.77% | 90.23% |

These are synthetic closed-bank results, not real CAPTCHA font accuracy. No real images trained or calibrated either classifier. Original line/curve/dot models remained unchanged.

Ran frozen font inference and rendering search on all five families' existing fitting/development pools. Generated 160 more candidates, no rendering failures. The selected font/effect combinations were Courier Prime filled (CryptBB), Caveat bold (City), Courier Prime filled (Courier), Pixelify Sans filled (DarkBay), and Courier Prime bold (outlined). Assistant inspection of the all-family gallery: some serif/weight changes are visible; selection is not consistently correct, and outlined remains a clear failure. Lower embedding matching loss must not be interpreted as verified font matching.

## New external family: tiny frozen smoke test

Source: [pymlex/4-captcha](https://huggingface.co/datasets/pymlex/4-captcha), pinned revision `d36ce8b0cf06f2ffcdc459482a0c94b8f8380f78`. Dataset card declares GPL-3.0. Downloaded all 12 published clean preview images and source card; no labels, source generator, fonts or weights. The full archive is 3.04 GB; previews are not a random or representative sample.

Four upstream training previews guide inference aggregation only. Eight val/test previews are excluded from parameter selection. Both shared CV measurements and frozen line/curve/dot inference run on the images (360 blocks), then frozen Phase 2 CNN recipe prediction and frozen font model 02. Generated 16 samples at each phase (48 outputs), zero rendering failures, all model hashes unchanged. No fitting or recalibration on the new family.

Assistant visual review: the font model chose Pixelify Sans, which looks less like the reference typography. The five-character assumption and colored background bias also remain wrong for these examples. **This external smoke test does not pass visual matching.** It is useful evidence of a synthetic-to-real/font-domain gap, not an accuracy benchmark. Do not conceal this failure or claim the extra training solved font matching.

[Matched synthetic scores](../runs/style_phase3_font_02/matched_synthetic_test.json) · [Five-family font comparison](../runs/style_phase3_font_02/all_families.png) · [New-family comparison](../runs/style_phase3_new_family_01/comparison.png) · [Thesis paper register](THESIS_PAPERS.md)
