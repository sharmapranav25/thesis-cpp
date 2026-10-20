# Controller MVP v3

No VLMs, XGBoost, target recognition labels, target boxes or target recipe metadata. The existing neural hybrid supplies starting renderer recipes; its CTC output is not decoded. This iteration uses direct renderer search, not MetaSim2 or a trained adjustment policy.

## Completed results

All five experiments completed. Output audits verified 1,800 saved images, including 360 exact v2 baseline replays. None receives a close-reconstruction quality pass. See `report.html` for City, CryptBB, DarkBay and Outlined, and `SimpleCaptcha_run_01/report.html` for the user-selected Kaggle dataset.

Synthetic mask validation improved from 50.0% to 70.0% mean IoU; hollow text with interference improved from 25.4% to 57.4%. These are validation results, not real-target mask accuracy. DarkBay showed partial reconstruction improvement. City changed little, CryptBB remained poor, and Outlined exposed a failure where fragmented thin text improved the score while resemblance worsened.

The Kaggle Simple Captcha download contained 6,519 image entries and 6,518 unique pixel hashes. Images were extracted under hash-only names; answers and generator code were not used. `test_simple.py` applies the unchanged v1 → v2 → v3 procedure with a separate 256-image final holdout. This test failed visually: script lettering and blobs do not reproduce boxed capitals. The same checkpoint and search rules were used, without retraining or test-driven retuning.

Every full search logged two rejected proposals for the deliberately unsampled `caption` effect, originating in the background enumeration. The renderer whitelist prevented those candidates from producing selected outputs. Logs preserve these failures rather than hiding them.

## Changes

- Full-resolution U-Net trained from scratch on 4,096 independently generated renderer images. Training includes hollow text, crossing interference and multicolor glyphs. 512 validation images use separate font families. Checkpoint selection uses synthetic validation only.
- Pixel-only mask inference on both target and generated comparisons. Synthetic truth masks never enter candidate appearance scores. They supervise training and support separately labeled synthetic diagnostics.
- Typography descriptors add hole occupancy, erosion survival, stroke-distance quantiles and local image contrast to v2's component shape and layout measures. They do not read characters or identify exact fonts.
- Search can introduce all 38 sampled renderer effects, even when absent from hybrid predictions. Adding an effect updates the inherited `keep_only` filter so it is not silently discarded. Background search preserves selected glyph effects. Font, layout, length and palette controls remain available.

## Experiment

`train_mask.py` generates synthetic data and trains for 20 epochs. `prepare.py` builds diagnostic and fresh target manifests using pixel hashes. `checks.py` verifies effect reachability and descriptor contracts. `pipeline.py` runs the same protocol on Outlined, CryptBB, City and DarkBay. `audit.py FAMILY` verifies saved image hashes, exact v2 replay, paired seeds, target/synthetic separation, fresh split disjointness, unchanged source and frozen controls. `report.py` creates the local HTML report.

The paired ablation ranks identical 96 candidate renders with (1) old masks/old score, (2) new masks/old score and (3) new masks/new typography score. Full v3 continues from arm 3 through all-effects, background, font, layout, palette and refinement phases. That continuation has additional search budget; it is not a clean isolated measurement of renderer freedom.

Fit uses 18 synthetic renders per candidate, selection 36. All arms render the same 72 v2 test recipes and seeds. Changing length may change the generated random string. Fit/selection targets are reused from earlier versions. Previously viewed v2 test images are explicitly diagnostic. City, DarkBay and Outlined have separate 128-image fresh controller holdouts, excluding known prior controller/ablation/localization manifests. CryptBB has no remaining fresh images. No claim of globally untouched data is made.

Selection requires improved combined objective, no more than 5% text or background regression versus v2 under that judge, no loss of complete-render fraction, and at most 3 percentage points reduction in synthetic usable fraction. It does not guarantee test improvement. All candidate failures are recorded. No test-driven reruns or target-specific presets.

## Limits

Mask truth means visible original glyph ink after effects/occlusion; outline expansion and decorative shadows are excluded. A high synthetic IoU is not proof of real target localization. Shape descriptors can confuse holes in characters with hollow styling; multiple random strings reduce but do not eliminate that confound. The new score is handcrafted, not a learned font similarity model. Font-bank overlap with real targets is unknown; no target fonts were added. The renderer's available grammar, fixed hybrid character group and finite search budget constrain resemblance. Proxy improvements do not constitute a visual reconstruction pass.
