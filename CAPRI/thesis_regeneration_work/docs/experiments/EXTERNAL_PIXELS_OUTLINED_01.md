# EXTERNAL-PIXELS-OUTLINED-01 — public dataset probe

Date: 2026-09-27 America/Los_Angeles (download receipt is 2026-09-28 UTC).

## Question and scope

Does the unchanged Method 1 implementation produce correct descriptive measurements on a newly acquired public CAPTCHA collection? This is a fifth, exploratory dataset, separate from the four primary thesis pools. It is not a test of complete CAPTCHA regeneration, and does not skip ahead to later methods.

## Source and acquisition

- Dataset: [muddokon/outlined-text-captcha](https://huggingface.co/datasets/muddokon/outlined-text-captcha).
- Pinned repository revision: `6c6a6a6911ae5b3a288378497cb9139e430dee7b`.
- Dataset metadata declares MIT. Original ownership and real-world collection provenance were not independently verified; do not describe this as a proven independent real-world benchmark.
- Downloaded upstream `train.zip` only, **6,512,469 bytes**, SHA-256 `6a05e6824d2b4279be0a0cdd697a9fbc7105e17738987a5112721329ec43cf6c`; this matches Hugging Face's LFS object hash. Upstream validation/test ZIPs were not downloaded.
- Selection was a convenience choice from public search results with a declared license and small archive, rather than a statistically random draw from all available datasets. It was not selected by measured performance.
- [Download receipt](../../data/external/outlined_text_captcha/source/download_receipt.json) and [pixel-extraction receipt](../../data/external/outlined_text_captcha/extraction_receipt.json).

## Process

1. Download the pinned archive and verify byte count/hash.
2. Decode the **800 image members** without executing archive content or using labels. All archive members were images. Save RGB images under opaque member-index/content-hash names; no original filename is supplied to the feature method.
3. Check decoded-pixel identity against all four previous input pools and the declared CryptBB/DarkBay evaluation pools. No exact overlap was found. Near-duplicates or common generator ancestry are not ruled out.
4. Run the existing `study_pixels.run_dataset()` and `pixel_statistics.measure()` unchanged. Code hashes match the four-dataset Method 1 run.
5. Deduplicate and split by the existing hash/seed policy: **800 unique images; 640 fitting / 160 validation**. No image sampling limit was applied within the downloaded training split.
6. Freeze the fitting profile, then measure held-out images and produce review panels. Assistant vision inspected both panels; no transcription was used to fit parameters.
7. Run a separate spatial negative control to test what the metrics cannot establish.

## Commands

Executed in `C:/Users/pranav/Documents/ChatGPT/thesis/thesis_regeneration_work`; all artifacts are subsequently published under the matching `E:/pranav/Final sem/sourcefree_synth` paths.

```powershell
.\scripts\download_external_captcha.ps1
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B scripts/study_external_captcha.py
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B scripts/pixel_spatial_control.py
```

Reruns need new output paths; existing archives/images/results are not overwritten. The first Python HTTPS attempt failed because the interpreter's SSL module was unavailable. The PowerShell sandbox then blocked network sockets; the approved PowerShell download succeeded. These were acquisition-environment issues, not feature-method failures.

## Quantitative observations

| Measurement | Result | Evidence |
|---|---|---|
| Canvas | 200x50 for all 800 images | Per-image measurements and `result.json` |
| Exact duplicates | 0 | `split.json` |
| Numerical checks | All pass | `result.json` |
| Independent Pillow cross-check | Maximum moment error about 1.05e-12 over 16 checked images | `independent_checks` in `result.json` |
| Fitting mean RGB | Approximately [145.009, 145.009, 148.776] | `frozen_fit_profile.json` |
| Mean fitting brightness proxy | 145.438 on 0–255 scale | `frozen_fit_profile.json` |
| Mean within-image brightness std | 61.298 | `frozen_fit_profile.json` |
| Fit/validation mean RGB histogram intersection | 0.993505 | `result.json` |
| Measurement run wall time | 24.80 seconds, including exclusion inventory | `result.json` |
| Runtime model/VLM calls and training | None | Source code and environment receipt |

The histogram score is color-distribution overlap, **not 99.35% recognition accuracy or regeneration quality**. Full evidence: [run results](../../runs/external_pixels_outlined_01/outlined_text_captcha/result.json), [environment and code hashes](../../runs/external_pixels_outlined_01/environment.json), [frozen fitting profile](../../runs/external_pixels_outlined_01/outlined_text_captcha/frozen_fit_profile.json), and [split](../../runs/external_pixels_outlined_01/outlined_text_captcha/split.json).

## Assistant visual review

Opened [12 fitting display slots](../../runs/external_pixels_outlined_01/outlined_text_captcha/fit_review.png) and [9 validation display slots](../../runs/external_pixels_outlined_01/outlined_text_captcha/validation_review.png); selection is deterministic and includes fitting extremes. OBSERVED: gray left-to-right shading, thin dark or blue outline-like text, and a much thicker dark crossing stroke. No character strings were fed to the analysis.

The strips correctly summarize dominant gray color mass, but their top six bins often omit thin blue details. The omitted fraction is explicitly displayed; the complete 512-bin histogram retains the full quantized color distribution. The claim that an outline or crossing stroke exists is the reviewer's qualitative observation, **not a capability demonstrated by Method 1**. Inferring their geometry remains later work.

## Spatial negative control

For four hash-selected validation images, randomly permuted whole RGB pixels using fixed seeds. The result visibly destroys glyphs, the gradient's spatial direction and the crossing stroke while preserving dimensions and all global color counts.

All four original/shuffled pairs have **identical histograms**, intersection approximately 1.0, and moment differences below 3e-13. This is a successful negative control demonstrating that Method 1 cannot evaluate spatial structure. The shuffled outputs are control images, not generated CAPTCHAs or proposed training samples.

![Original versus shuffled control](../../runs/external_pixels_outlined_01/spatial_negative_control/comparison.png)

[Control results](../../runs/external_pixels_outlined_01/spatial_negative_control/result.json).

## Verdict

**PASS for applying the unchanged pixel-statistics method to this external dataset. INSUFFICIENT for discovering the full CAPTCHA rendering rules or validating regeneration.** The next eligible method remains edges and texture analysis on the four primary datasets; this external set can be retained as an additional regression probe. Do not use this result to mark Methods 2–6 complete.

Source attribution is for the dataset. The algorithm's research provenance remains R1 in [METHODS.md](../../METHODS.md); no additional paper-derived algorithm was introduced in this probe.
