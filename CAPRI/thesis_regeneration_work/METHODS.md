# Methods and research provenance

This is the method registry for the thesis. Planned methods are not completed experiments. Each method needs its own detailed record under `docs/methods/` and results on all four datasets before the next method begins. Current execution state is in [STATUS.md](docs/STATUS.md).

| Order | Method | Intended observations | Main limitation | State |
|---|---|---|---|---|
| 0 | Imported baseline | Reproducibility of existing generator behavior | Hard-coded text rules; broad random styles | PASS: 128 exact historical matches |
| 1 | Pixel statistics | Dimensions, RGB distributions, brightness, contrast, palette mass | No spatial/semantic distinction between glyph and noise | PASS for measurements on all four datasets |
| 2 | Edges and texture | Edge density/orientation, line/grid periodicity, local texture | Edges can belong to glyphs or clutter | Not started |
| 3 | Segmentation | Candidate foreground masks, components, spacing and count hypotheses | Touching/split glyphs and noise; no automatic semantic truth | Not started |
| 4 | Fit-and-compare rendering | Renderer settings supported by image similarity | Proxy mismatch and unavailable glyph styles | Not started |
| 5 | Conventional OCR | Tentative alphabet and length evidence | Domain errors and correlated/confident mistakes | Not started |
| 6 | Image embeddings + clustering | Image style groups and coverage | Learned similarity may ignore character details | Not started |

## Method 1: pixel statistics

Use native decoded RGB pixels, not resized thumbnails, for exact dimensions and distributions. Compute per-image normalized joint RGB histograms, channel moments, luminance quantiles/standard deviation, and frequent occupied color bins. Aggregate with equal image weight so high-resolution images do not dominate. Retain dimension and color variation rather than replacing a family with one average image.

Histogram overlap is an appearance diagnostic. It contains no character identity or spatial arrangement. Report whole-image palette colors without claiming that the most frequent color is necessarily background or that other colors are glyphs.

This limitation was tested directly in the [external public-data probe](docs/experiments/EXTERNAL_PIXELS_OUTLINED_01.md): four randomly pixel-permuted images retained exactly the same histograms while losing visible CAPTCHA structure. Method 1 also passed its existing numerical checks on all 800 external images without changing its code/settings.

### Research actually consulted

**R1. Michael J. Swain and Dana H. Ballard, “Color indexing,” International Journal of Computer Vision 7, 11–32 (1991). DOI: [10.1007/BF00130487](https://doi.org/10.1007/BF00130487).** Publisher [abstract and bibliographic record](https://link.springer.com/article/10.1007/BF00130487) read 2026-09-27. The abstract motivates color histograms and histogram intersection as efficient appearance comparisons. This project adopts that limited idea; its RGB binning, dataset splitting and validation thresholds are our choices, not paper claims. The full publisher text was not available; an attempted university PDF redirected to authentication. Do not describe this as a full reproduction of the paper.

**Considered but not adopted:** Reinhard et al., “Color Transfer between Images” (2001), DOI [10.1109/38.946629](https://doi.org/10.1109/38.946629). A PDF retrieval failed and no implementation is attributed to it. It is not evidence that color transfer would solve CAPTCHA style recovery.

Simple dimensions and descriptive statistics are standard calculations, not a claimed research novelty. Detailed definitions, checks and limitations belong in [01_PIXEL_STATISTICS.md](docs/methods/01_PIXEL_STATISTICS.md).

## Later-method literature

No paper is yet claimed as used for Methods 2–6. Research will be read when each method becomes eligible. Each adopted paper entry must include authors/year/title/URL or DOI, exactly which idea is used, deviations, where it appears in code, and whether the full paper or only an abstract was read. Model weights also need version, source, training-data caveats and measured compute.

## Interpreting outcomes

A method can pass measurement validation while leaving full image regeneration imperfect. State those outcomes separately. Unsupported semantic conclusions are failures or unresolved questions, not measured rules. A sampled assistant visual review is qualitative evidence, not dataset-wide ground truth or OCR accuracy.
