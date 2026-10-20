# Sequential validation protocol

Protocol established 2026-09-27 before new method runs. Thresholds below are engineering acceptance settings, not established CAPTCHA research benchmarks.

## Baseline gate

Replay all previously saved CryptBB cases using the same generic fonts and random seeds. Require identical RGB pixel hashes, labels and dimensions for every historical sample, unchanged vendor/original source hashes, and verified imports from the vendored directory. Record source defects separately: reproducing a defect is evidence of equivalence, not approval for production.

## Method gate

1. Declare the scope, inputs, fixed settings and measurable claims before running.
2. Validate the calculations against known artificial fixtures; fixtures are not target data.
3. Process all four real input pools, log failures and duplicates, and exclude exact pixel overlap with external evaluation pools.
4. Freeze estimates from fitting images before computing validation diagnostics. Keep manifest and code/config hashes.
5. Review representative and extreme cases visually; record image paths, sample-selection policy, reviewer, observations and limitations.
6. Mark a dataset PASS, FAIL or INCONCLUSIVE for the declared scope. Only four PASS results plus recorded visual review permit the next method. Do not redefine a failed acceptance threshold using the same validation images without a new protocol/version and disclosed validation reuse.

## Method 1 pre-run checks

- Known solid and two-color fixtures give exact counts, histogram mass and expected brightness; constant images have zero standard deviation.
- On every decoded input: all summaries finite, dimensions exact, histogram sum within 1e-10 of 1, frequent-bin mass in [0,1], no channel value outside [0,255].
- Cross-check RGB mean/std independently with Pillow ImageStat on 16 fixed, hash-selected images per dataset; absolute error <= 1e-6.
- No unreadable inputs silently skipped; fit/validation identities disjoint; no external evaluation overlap.
- Report dimension coverage, fit/validation histogram intersection and moment shifts. No universal cutoff is claimed for these descriptive diagnostics.
- A property may be measured correctly even when the fit pool does not represent a rare held-out style. Record rare/unseen modes rather than hiding them.
- Palette strips must display actual bin representative colors and the true covered pixel fractions, with omitted mass visible. Do not label colors as text/background.

## Regeneration versus measurement

The initial pixel study establishes what is observable. It does not infer a text alphabet, glyph font, segmentation, noise placement or distortion model. Any generator parameter connection needs a separate explicit check. A preserved hard-coded baseline remains visibly marked as such.

## Leakage and final scoring

Input paths may contain answers; never derive text from them. Do not open label files for fitting or validation gates. Hold out real labels for final downstream scoring after fixing model selection. Pixel equality does not rule out near-duplicate/cropped/recompressed leakage. Internal visual review may influence development, so it cannot be presented as an untouched final test.
