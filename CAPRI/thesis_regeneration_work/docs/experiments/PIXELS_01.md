# PIXELS-01 — native pixel statistics on four datasets

Date: 2026-09-27. Reviewer: assistant image-viewing capability. Quantitative scope: all input images; qualitative scope: the deterministic panels linked below. Protocol was specified in `docs/VALIDATION.md` before execution.

## Commands

Executed in `C:/Users/pranav/Documents/ChatGPT/thesis/thesis_regeneration_work`; publish the resulting files into the matching `E:/pranav/Final sem/sourcefree_synth` paths.

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B -m unittest discover -s tests -p test_pixel_statistics.py -v
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B scripts/study_pixels.py --baseline runs/baseline_replay_01 --out runs/pixel_statistics_01
```

All four fixture tests passed: solid image, two known colors, equal-image rather than equal-pixel weighting, and joint width/height sampling. Runtime for fixtures: 0.175 seconds. Real-data environment, implementation/protocol/config hashes and CPU settings are in [environment.json](../../runs/pixel_statistics_01/environment.json).

## Results

| Dataset | Input files | Unique | Fit / validation | Histogram intersection | Runtime seconds | Scoped gate |
|---|---:|---:|---|---:|---:|---|
| CryptBB | 600 | 600 | 480 / 120 | 0.959161 | 5.78 | PASS |
| City | 2,000 | 1,999 | 1,599 / 400 | 0.999043 | 43.99 | PASS |
| Courier | 2,510 | 2,510 | 2,008 / 502 | 0.922870 | 59.58 | PASS |
| DarkBay | 799 | 799 | 639 / 160 | 0.917605 | 4.63 | PASS |

Histogram intersection is descriptive overlap of average RGB distributions. It is not an accuracy percentage, a significance test, or a criterion for claiming successful CAPTCHA regeneration. Every image passed finite/range/histogram-mass checks. Independent Pillow calculations agreed within 1e-6 on 16 selected images per dataset. Fitting profiles remained unchanged after validation. The specified external exclusion checks found no exact decoded-pixel overlap.

## Visual findings

All eight fit/validation panels were opened and inspected, up to 21 displayed slots per dataset (some fitting extremes repeat a representative selection). The reviewer did not transcribe text or supply character labels to the pipeline.

| Dataset and artifacts | OBSERVED in sampled images | What remains unresolved |
|---|---|---|
| CryptBB: [fit](../../runs/pixel_statistics_01/cryptbb/fit_review.png), [validation](../../runs/pixel_statistics_01/cryptbb/validation_review.png) | Light dominant color mass, many colored details and mixed background patterns; palette strips show substantial omitted mass | A six-bin palette is incomplete; global colors cannot tell text from clutter |
| City: [fit](../../runs/pixel_statistics_01/city/fit_review.png), [validation](../../runs/pixel_statistics_01/city/validation_review.png) | Predominantly white/light gray with smaller dark details; top six bins cover about 99% of sampled pixels | Spatial structure and character identities are not captured |
| Courier: [fit](../../runs/pixel_statistics_01/courier/fit_review.png), [validation](../../runs/pixel_statistics_01/courier/validation_review.png) | Both light and dark dominant colors, colored strokes and clutter; high-entropy example has substantial residual color mass | Family mean brightness hides strong within-family variation |
| DarkBay: [fit](../../runs/pixel_statistics_01/darkbay/fit_review.png), [validation](../../runs/pixel_statistics_01/darkbay/validation_review.png) | Muted block colors with dark/light details; the palette shows dominant color groups without assigning roles | Block layout, font and character count are not recovered |

Palette strips agree qualitatively with displayed color mass. Gray remainder rectangles are explicitly labeled, not interpreted as a measured gray color. Measurements use native pixels even where display thumbnails are resized. Visual checking is sampled; numerical checks cover all images.

## Measured discoveries and limitations

- City has one exact duplicate. Its fitting dimensions include 318x79, 324x81, and one 788x198 outlier. These are measurements, not an inferred explanation for capture/resizing differences.
- Courier has 25 joint width/height modes in fitting, and validation contains an additional 201x56 mode. Its cause is UNRESOLVED. A sampler fitted only on fitting images cannot reproduce that unseen exact size; no validation-informed mode was added.
- CryptBB fitting/validation dimensions are 200x60; DarkBay is 200x50.
- Width/height sampling was validated as preserving observed joint modes. Color-to-text/background mapping remains UNRESOLVED, and no complete regenerated-family fidelity claim is made.

See each `runs/pixel_statistics_01/<dataset>/result.json`, `frozen_fit_profile.json`, `validation_profile.json`, `input_receipts.json`, `split.json` and per-image JSONL/NPZ outputs for exact evidence.

## Verdict

**Method 1 PASS on all four datasets for correct descriptive measurements and joint canvas sampling.** This does not pass segmentation, alphabet inference, complete style recovery or downstream recognition. It permits Method 2 to begin under the sequential protocol. Assistant vision was used for qualitative review only; no VLM ran in the measurement code.
