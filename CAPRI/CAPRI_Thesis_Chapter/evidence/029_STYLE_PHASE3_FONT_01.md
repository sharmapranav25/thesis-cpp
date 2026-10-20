# Phase 3 font experiment 01

Synthetic-only font training completed: 12 audited OFL fonts, 3,072 training images, 768 synthetic selection images and 768 separate synthetic test images. Checkpoint selected at step 700 using selection loss. No real image entered classifier training; the model was frozen before real inference.

Synthetic test: **50.52% top-1**, **81.64% top-3**, **92.19% filled/hollow accuracy**. Chance font top-1 is 8.33%. Per-font performance varies substantially; these numbers do not estimate accuracy on real CAPTCHAs.

Ran both previously frozen feature-classifier predictions and shared CV measurements as the foundation; new font matching inferred on 96 fitting/32 development images per family. Typography search chose candidate fonts/effects using fitting and development data only. The Phase 2 reserved evaluation set was not opened. Produced 32 candidates per family, 160 total, with no search or generation failures. Original feature-model hashes and frozen new font-model hash were verified.

| Family | Selected font | Effect | Width multiplier | Height multiplier |
|---|---|---|---:|---:|
| CryptBB | Courier Prime | filled | 1.1 | 1.25 |
| City | Caveat | filled | 1.1 | 1.25 |
| Courier | Balsamiq Sans | bold | 1.1 | 1.0 |
| DarkBay | Pixelify Sans | filled | 1.1 | 1.0 |
| Outlined | Courier Prime | filled | 1.1 | 1.25 |

Assistant inspection of CryptBB and outlined galleries: some serif lettering better resembles the broad CryptBB typography, but outlined remains a clear failure (filled letters instead of hollow outlines). Lower development matching loss is not sufficient evidence of visual success. No real font labels, exact-font identification, held-out real accuracy or Phase 3 completion claimed. Other galleries remain available for inspection.

[All-family gallery](../runs/style_phase3_font_01/all_families.png) · [Synthetic test](../runs/style_phase3_font_01/synthetic_test.json) · [Training log](../runs/style_phase3_font_01/training.json) · [Real development results](../runs/style_phase3_font_01/real_development_results.json) · [Thesis paper register](THESIS_PAPERS.md)

Further training is justified by the still-improving synthetic selection curve, not by tuning on synthetic test labels or reserved real evaluation. A separate run preserves this first result and uses a fresh synthetic test seed range.
