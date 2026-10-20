# Method 1 — Pixel statistics

Status: PASS on all four datasets for the declared measurement scope; [experiment record](../experiments/PIXELS_01.md). Scope: whole-image descriptive measurements, not character recognition or background segmentation.

## Definitions

1. Decode with EXIF orientation and composite transparency over white, matching existing image ingestion. Report any alpha use.
2. Preserve native dimensions. Record width/height distribution instead of inferring it from a manually supplied family constant.
3. Measure RGB means and population standard deviations over all pixels.
4. Compute display-space brightness `Y = 0.299 R + 0.587 G + 0.114 B`; this is a conventional luma-like proxy, not physical luminance. Record quantiles 0, 5, 25, 50, 75, 95, 100% and standard deviation.
5. Quantize each RGB channel into eight bins (32 levels/bin); count all 512 joint bins, then normalize by pixel count. The normalized intersection between two histograms is `sum(min(p_i, q_i))`, ranging from 0 to 1. This follows the histogram-comparison idea in R1; binning is our fixed choice.
6. Display the six most frequent occupied bins using the mean RGB of pixels in each bin; retain actual weights and omitted mass. Mean colors within a bin are summaries, not necessarily exact individual pixels. Call them palette summaries, never text/background labels.
7. Aggregate per-image statistics and normalized histograms with equal image weight. Report distinct dimension modes and validation modes absent from fitting.

## Experimental design

Read all images in all four configured input pools. Deduplicate by dimensions plus decoded RGB identity. Deterministically shuffle sorted image hashes with seed 42 and use 80% fitting / 20% validation. Keep all original file receipts so duplicate removal is auditable. Fitting summaries are written and hashed before validation summaries. Audit external evaluation overlaps without reading labels.

Compare independent Pillow ImageStat calculations on 16 hash-selected images per dataset. Use artificial solid/two-color fixtures to validate expected moments and histogram masses. Numerical gate details: [VALIDATION.md](../VALIDATION.md).

Representative visual panels use evenly spaced fitting hash order plus darkest/brightest/highest-color-entropy cases. Validation panels use a separate deterministic selection. Report the sampled image identities. A visual panel is not inspection of every image.

## Generator interface

Dimension distributions can directly inform canvas sampling. Color histograms are appearance evidence but do not establish where to place each color. Palette-to-background/text assignment is intentionally unresolved at this stage and must not be smuggled in as a measured rule. No alphabet, length, font or distortion rule is inferred by Method 1.

## Research

R1 in [METHODS.md](../../METHODS.md): Swain and Ballard (1991), DOI 10.1007/BF00130487. Used only for the color histogram/intersection idea from the verified abstract; not a reproduction of the full paper. Fixtures, RGB quantization and validation protocol are this project's engineering decisions.

## Results

See [PIXELS-01](../experiments/PIXELS_01.md), [raw summary](../../runs/pixel_statistics_01/summary.json) and [visual review](../../runs/pixel_statistics_01/visual_review.json). One City duplicate was removed; Courier includes a validation-only canvas mode, retained as a limitation. No complete-family regeneration claim.
