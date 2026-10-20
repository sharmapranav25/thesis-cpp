# Region demonstration 02: ten across, three down

**OBSERVED:** At the user's request, the demonstration now divides each image into **10 columns and 3 rows: 30 base bricks**. Each crop extends by 12.5% of its brick width/height on each available side (rounded upward to whole pixels). Neighboring crops therefore overlap by approximately 25%; boundary crops are clipped to the source bounds. The previous two-scale grid is not the current requested configuration.

Open the [interactive explorer](../../runs/feature_bank_regions_02/explore.html). Select an image and a brick to see its location, its crop, and which neighboring crops contain shared pixels. Orange marks the selected crop, blue marks geometric neighbors, and gray marks base boundaries. These overlays are display-only and never enter saved crops.

## What ran

**OBSERVED:** One image per real dataset was selected by lowest decoded-pixel hash from that dataset's previously frozen 40-image validation sample. The same selection rule selected one existing successful pilot image per synthetic backend. No visual cherry-picking, classifier output, annotation files, CAPTCHA strings from real images, or filename semantics were used. Dataset names choose the five review pools; the crop function receives only image dimensions and fixed geometry.

Ten images produced 300 native-resolution crops: 150 evaluation-only crops and 150 pilot-only synthetic crops. Every feature label remains **unknown**, which must not be interpreted as absent. No full-image label is inherited. Pilot crops are not an accepted training corpus.

**OBSERVED:** Twelve tests passed, including exact native-pixel extraction, full coverage, exact overlap reassembly, odd-sized canvases, the requested 10-by-3 geometry, and rejection of cross-partition base groups or duplicate parents. The ten actual image exports independently passed pixel/coverage/reassembly checks. These are geometry and data-integrity tests, not classifier accuracy tests.

## Avoiding misleading results

- Split synthetic **base generation groups before cropping**. Keep all crops, paired renders, font/effect variants and replay variants of a base example in one partition. Duplicate parent images also cannot cross partitions. The group-partition validator enforces the supplied group identities; generating correct group identities remains the corpus builder's responsibility.
- Target images and user-audited feature labels stay out of training, calibration, threshold choice and model selection. These datasets have already been visually inspected; they cannot honestly be called untouched final tests.
- Shared pixels establish geometric adjacency only. They do not prove that two marks are the same line or that a curve continues between pieces. Connecting detected features remains unimplemented.
- Smaller pieces may erase the context needed to distinguish glyph strokes from interference. Retain the whole image for context and allow uncertain observations. More pieces are not automatically more accurate.
- Future synthetic crop labels need final visible-effect masks and trusted local evidence. A requested effect, or a positive whole-image label, does not establish a positive crop. Absence also requires reliable negative evidence.
- Assistant visual review is part of the research process. It is not a runtime classifier, independently collected ground truth, or a substitute for the required user audit.

## Visual review

**OBSERVED by assistant:** The coarse predecessor panels showed marks and glyphs crossing cut boundaries; overlap preserves portions on both sides. The new denser panels make those fragments smaller. This motivates retaining position and global context; it does not establish improved feature recognition. No settings were chosen by measured target accuracy.

[CryptBB](../../runs/feature_bank_regions_02/cryptbb/overview.png) · [City](../../runs/feature_bank_regions_02/city/overview.png) · [Courier](../../runs/feature_bank_regions_02/courier/overview.png) · [DarkBay](../../runs/feature_bank_regions_02/darkbay/overview.png) · [Outlined text](../../runs/feature_bank_regions_02/outlined_text_captcha/overview.png)

## Reproduce and status

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B scripts/demonstrate_regions.py --columns 10 --rows 3 --out runs/feature_bank_regions_NEW
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
```

**PROPOSED next:** finish renderer clipping/visibility gates, then construct grouped synthetic examples with crop-specific labels and train local feature classifiers. Aggregate their evidence using position and continuity without counting overlapping crops as independent votes. Evaluate local labels separately from whole-image labels; the existing 200-image audit is not automatically crop ground truth.

**Status:** geometry demonstration complete; bank capability gates still partial, training and feature matching not started. No research result or new algorithm gate is claimed. See [protocol](../../runs/feature_bank_regions_02/protocol.json), [crop manifest](../../runs/feature_bank_regions_02/manifest.json), [checks](../../runs/feature_bank_regions_02/status.json), and [implementation](../../feature_bank/regions.py).
