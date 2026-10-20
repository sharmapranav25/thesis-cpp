# Feature Bank and CAPTCHA Feature Classification — Implementation Plan

## Execution update — 2026-09-28 UTC

**OBSERVED:** Stage 1 input/selection checks completed across five datasets: 6,708 historical unique images verified, 40 validation images per dataset selected, and 200 assistant feature-label drafts saved. User audit remains pending. Stage 2 initial pilots generated and reviewed 320 images; a separate 64-image MobiCMS random-tape repair achieved exact replay. **Stage 2 is PARTIAL; corpus/classifier/matching gates have not passed.** No model was trained.

- [Pilot results, failures and next gate](feature_bank/PILOT_01.md)
- [Resources and acquisition](feature_bank/RESOURCE_ACQUISITION_01.md)
- [Feature definitions and user annotation audit](feature_bank/ANNOTATION_PROTOCOL.md)
- [Adapter architecture and capability-check limits](feature_bank/ADAPTERS.md)
- [Feature-to-function coverage and unsupported cases](feature_bank/MAPPING_COVERAGE.md)
- [User review page](../runs/feature_bank_01/annotation_review.html)

The dated planning statements below describe the pre-implementation baseline. Original planning text is retained in [the archive](archive/FEATURE_BANK_CLASSIFICATION_PLAN_before_feature_bank_stage2_20260928.md).


Date: 2026-09-27.

Status: planning and handoff document. The feature-bank experiment described here has not been implemented or trained. Existing baseline and pixel-statistics results are separate experiments.

## Start here: handoff to the implementing session

Implement this plan inside `E:\pranav\Final sem\sourcefree_synth`, as a parallel experiment. Do not edit the imported repository at `E:\pranav\Final sem\CaptchaThesis` or rewrite historical results. Read the linked architecture, status, issues, dataset register and resource survey before implementing. Inspect any applicable `AGENTS.md` instructions and verify the current environment; the observations below are a dated snapshot.

The user explicitly chose:

1. **Parallel experiment:** retain the original six-method investigation and its completed results.
2. **PHP + Python:** use shortlisted PHP resources and useful existing Python rendering components.
3. **Endpoint: bank and classifiers only:** include feature-to-function matching; defer complete target-style regeneration and downstream OCR training.
4. **Evaluation:** assistant-drafted feature annotations, audited by the user, for 40 images per dataset across five datasets. No CAPTCHA transcription is needed.

Build and validate stages in order. Show intermediate images, feature views, detected evidence and candidate bank examples. Record failures rather than calling generic generation a successful recreation of a target family. Do not ask the user to decide these already selected scope choices again.

This document defines future implementation and experiments. Saving it did not install libraries, change generator code, run training or generate a new corpus. The next session must keep completed, proposed and failed work distinct. User auditing is required before calling the real-image annotations audited; pending annotations must not be silently treated as approved.

### Existing context to read

- [Architecture](../ARCHITECTURE.md)
- [Current execution status](STATUS.md)
- [Original method registry and references](../METHODS.md)
- [Dataset register](DATASETS.md)
- [Validation protocol](VALIDATION.md)
- [Issues and decisions](ISSUES.md)
- [Imported generator test: generic output versus target-style recreation](experiments/IMPORTED_ALL_DATASETS_01.md)
- [PHP resource survey and published examples](research/PHP_LIBRARY_SURVEY_2026-09-27.md)
- [Frozen imported-source provenance](../vendor/PROVENANCE.json)

## 1. Objective, boundaries and deliverables

Build a system that examines an unfamiliar CAPTCHA image, identifies supported visual features, and returns the rendering functions and resources capable of producing similar features.

The experiment ends with:

1. A verified catalog of rendering assets and capabilities.
2. Synthetic examples with reliable feature labels.
3. Classical computer-vision measurements and trained feature classifiers.
4. A mapping from observations to ranked rendering capabilities.
5. Evaluation on all five available datasets.
6. Documentation, visual examples, failures and reproducible experiment records.

Full target-style generation, automatic recipe optimization and OCR training are deferred. Rendering is still required here to create classifier training examples and test bank capabilities.

Retain these constraints:

- Unlabeled target images only for analysis; no target generator source or target fonts.
- No VLM, GAN or pretrained image embedding model in the runtime pipeline.
- Assistant visual review may draft evaluation annotations; the user audits them. This means the research process includes VLM assistance, although the implemented runtime does not.
- Evaluation annotations never enter training, calibration or model selection.
- CPU-first implementation. The target single RTX A4000/16 GB budget is unnecessary for the initial classifiers; physical GPU availability has not been verified for this plan.
- Preserve source snapshots, original data, exclusions and historical results.
- All numerical budgets and acceptance thresholds below are **PROPOSED experimental settings**, not established research results.

### Current implementation facts

**READ:** There are reusable rendering functions, but no completed capability catalog or trained feature-bank classifier in the inspected project. The original Methods 2–6 remain unimplemented according to [STATUS.md](STATUS.md).

**READ:** The existing [features.py](../sourcefree_synth/features.py) mixes color, edge, layout and texture descriptors and has an optional DINO path. Reuse must be explicit; this file is not an isolated classical-feature experiment, and its DINO path is outside this experiment.

**READ:** The independent [renderer.py](../sourcefree_synth/renderer.py) and the frozen imported [background functions](../vendor/imported_repo/system1_synthetic_generator/previously_produced_generators/gen_gateonly/backgrounds.py), [noise functions](../vendor/imported_repo/system1_synthetic_generator/previously_produced_generators/gen_gateonly/noise_layers.py) and [text renderer](../vendor/imported_repo/system1_synthetic_generator/previously_produced_generators/gen_gateonly/text_renderer.py) are starting resources, not verified interchangeable capabilities.

**OBSERVED environment snapshot:** the existing interpreter `E:\pranav\model 1\.venv\Scripts\python.exe` has scikit-learn 1.6.1, scikit-image 0.24.0, NumPy 2.0.2, Pillow 11.3.0 and PyTorch 2.5.1+cu121 package metadata; importing OpenCV reported 4.13.0. PHP and Composer were not found by `Get-Command`. This does not prove they are absent from disk. Do not upgrade the shared environment blindly; record and isolate experiment dependencies.

## 2. CAPTCHA resources, datasets and feature-bank construction

### 2.1 Rendering resources

Use four PHP projects initially, plus the existing Python resources. These are candidates awaiting local capability tests, not already validated integrations.

| Resource | Intended contribution | Initial decision |
|---|---|---|
| [Gregwar/Captcha](https://github.com/Gregwar/Captcha) | Front/background lines, distortion, scatter, configurable text and backgrounds | Primary PHP renderer |
| [Securimage](https://github.com/dapphp/securimage/blob/master/securimage.php) | Curved interference, dot-like noise, text deformation and optional boxes | Primary PHP renderer |
| [MobiCMS](https://github.com/mobicms/captcha) | Font diversity and per-character font mixing | Typography resource |
| [simple-php-captcha](https://github.com/yasirmturk/simple-php-captcha/blob/master/simple-php-captcha.php) | Textured backgrounds, text rotation and shadows | Background/shadow resource |
| Existing Python rendering components linked above | Explicit backgrounds, shapes, noise, text layers and intermediate outputs | Instrumented reference components |
| [Laminas Captcha Image](https://docs.laminas.dev/laminas-captcha/adapters/#laminascaptchaimage) | Another implementation of distortion, dots and lines | Withheld renderer for generalization testing |

Keep S1SYPHOS as a documented alternative to Gregwar, rather than including both initially. Their shared ancestry makes them unsuitable as independent training/test sources. Keep mews/captcha and Yii on the expansion list. See the [survey](research/PHP_LIBRARY_SURVEY_2026-09-27.md) for evidence and limits of the inspection.

Before importing each project:

- Resolve and record an exact source revision and dependency lock.
- Record licenses separately for code, fonts and background assets.
- Preserve original source; record any instrumentation changes separately.
- Verify runtime compatibility, image decoding, text-label integrity and repeatability.
- If a resource cannot expose trustworthy effect labels, retain it only for qualitative/stress testing.
- Locate or provision an isolated PHP runtime before PHP adapter tests. Record compatibility failures and any adapter changes.

### 2.2 Target datasets

Use existing manifests and exclusions. Do not resplit casually or read text from filenames.

| Dataset | Previously recorded unique images | Role | Audited feature-label sample |
|---|---:|---|---:|
| CryptBB | 600 | Primary target appearance benchmark | 40 |
| City | 1,999 | Primary target appearance benchmark | 40 |
| Courier Market | 2,510 | Primary target appearance benchmark | 40 |
| DarkBay | 799 | Primary target appearance benchmark | 40 |
| Downloaded outlined-text CAPTCHA | 800 | Additional public-data benchmark | 40 |

Locations are in [datasets.json](../configs/datasets.json) and [DATASETS.md](DATASETS.md). The external data is under `data/external/outlined_text_captcha/images/` relative to the project. Deduplication and split evidence are in the existing pixel-study run manifests and their [primary](experiments/PIXELS_01.md) and [external](experiments/EXTERNAL_PIXELS_OUTLINED_01.md) reports. Verify manifests rather than treating the table as a fresh filesystem count.

Select 40 images per dataset reproducibly from its existing validation partition, independently of classifier predictions. Preserve their hashes and selected IDs. Excluded CryptBB/DarkBay evaluation pools remain excluded.

The downloaded dataset's real-world provenance is unverified. Existing datasets have already been inspected during development, so this experiment must not call them a pristine, previously unseen final test.

### 2.3 What a bank entry contains

| Entry type | Examples | Required information |
|---|---|---|
| Assets | Fonts, generic background textures | Source, hash, license, supported characters, inspected visual properties |
| Operations | Draw dots, draw curves, create gradient, render hollow glyph | Function, verified controls, parameter ranges, input/output type, examples |
| Complete renderers | A library producing a finished image | Supported effects, coupled controls, output limitations, provenance |

Each operation record includes:

- Stable capability ID and implementation revision.
- Supported visual features.
- Verified parameters, units and bounds.
- Whether it returns a complete image, an RGBA layer, or a mask.
- Whether it changes other features at the same time.
- Whether actual random choices and masks are available.
- Reproducibility status and known failure cases.
- Links to controlled example renders.

Do not assume complete PHP renderers expose independently composable layers. A complete-image renderer remains a complete-image entry unless an adapter demonstrably exposes its layers.

### 2.4 Initial feature vocabulary

Use multiple independent feature labels because several features can coexist.

| Feature | Meaning | Important distinction |
|---|---|---|
| Straight interference | Added, approximately straight strokes | Character strokes are hard negatives |
| Curved interference | Added curved or wavy strokes | Curved glyphs are hard negatives |
| Dot scatter | Small isolated noise marks | Punctuation and dotted fonts are different cases |
| Decorative shapes | Added geometric decorations | A character O is not automatically circular noise |
| Periodic grid | Repeated intersecting or parallel structure | Several unrelated lines do not establish a grid |
| Background texture | Repeated or irregular surface texture | Distinguish from foreground clutter where possible |
| Background gradient | Smooth spatial color/brightness change | Whole-image brightness alone is insufficient |
| Background blocks | Patch-like regions of different colors | Avoid treating every character as a color block |
| Hollow lettering | Letter strokes drawn around an unfilled interior | Different from a filled letter with a border |
| Glyph border | Additional contour around filled lettering | Different from hollow lettering |
| Glyph shadow | Offset duplicate-like structure associated with text | Different from unrelated neighboring strokes |
| Nonlinear text deformation | Visibly bent or deformed lettering | Original font design may make the cause ambiguous |

Record dimensions, colors, contrast, candidate stroke widths and spatial density as measurements rather than forcing them into arbitrary yes/no classes.

Observations describe visible evidence. They cannot prove which historical drawing operation produced an image. Ambiguous cases receive an `uncertain` label. Font decorations, separate overlays and intrinsic character shapes must not be conflated just because they contain similar contours.

### 2.5 Capability verification and synthetic corpus

First produce a 64-image pilot per renderer containing clean examples, individual effects and combinations. Inspect every pilot image and retain failures.

Then create a first supervised corpus capped at 24,000 accepted images:

- 20% clean examples and hard negatives.
- 40% examples emphasizing one effect.
- 40% combinations of two to four effects.
- Balance contributions across eligible backends.
- Vary strings, fonts, colors, dimensions, effect position and strength.
- Use generic ASCII letters/digits and lengths 3–8 as experimental diversity, not inferred target rules.

Record requested settings, actual operations and visible effects separately. A requested line that was skipped, clipped or completely hidden must not become an unquestioned positive visual label.

Generate masks or paired renders with/without an effect where feasible. Capture random choices once and replay them for paired examples; merely resetting a seed may change downstream randomness when an operation is removed.

For each trained feature require at least 1,000 known positive and 1,000 known negative training examples. Missing coverage leaves the feature unsupported rather than triggering invented labels. Respect the corpus cap and report gaps; do not silently expand the budget.

Split synthetic data into:

- 60% training.
- 15% model selection.
- 10% calibration and decision thresholds.
- 15% final synthetic testing.

Keep paired renders, crops and variants of a base example together. Deduplicate across partitions. Add separate tests for unseen font families, withheld combinations and the withheld renderer. Account for shared implementation ancestry; a renamed fork is not an independent source.

## 3. Image analysis and classification algorithms

### 3.1 Image views

Keep original RGB pixels and native dimensions. For fixed-size descriptors, resize proportionally into a 256 × 96 canvas and pad. Record the scale and valid-image mask; exclude padding from statistics.

Create eight views:

1. RGB.
2. Grayscale.
3. Grayscale with CLAHE local contrast enhancement.
4. Canny edges.
5. White top-hat response.
6. Black top-hat response.
7. Adaptive threshold, one polarity.
8. Adaptive threshold, opposite polarity.

These are feature-enhancing views, not independent votes or ground truth. Use the whole image plus overlapping regions at two scales. Retain region coordinates and summarize local measurements with their maximum, median and upper quantile. Preserve whole-image continuity measurements for lines and grids.

Do not decide that a feature is absent because fewer than half the regions contain it. Avoid resizing or filtering that silently erases the feature being measured; retain native-resolution diagnostic views.

### 3.2 Descriptor families

| Descriptor family | Intended evidence |
|---|---|
| Native RGB/HSV statistics | Color distributions, brightness and contrast |
| HOG | Local edge orientation and shape |
| Edge connectivity and Hough measurements | Long straight structures and their directions |
| Connected-component measurements | Candidate dots, small objects, elongation and spacing |
| Local texture statistics and autocorrelation | Texture and periodicity |
| Morphological responses | Small bright/dark marks, approximate stroke scales and contour structure |
| Spatial summaries | Where evidence appears and how broadly it is distributed |

Use HOG with nine orientation bins, 8 × 8 cells, 2 × 2 blocks and L2-Hys normalization. Compute it on grayscale and contrast-enhanced grayscale.

HOG's basis is supported by [Dalal and Triggs, 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf). That paper concerns human detection; its results do not establish CAPTCHA feature accuracy. The implementation reference is [scikit-image HOG](https://scikit-image.org/docs/0.24.x/api/skimage.feature.html#skimage.feature.hog). The abstract and relevant descriptor documentation were consulted for this proposal; this is not a reproduction of the paper's evaluation.

### 3.3 Algorithms

| Stage | Algorithm | Purpose |
|---|---|---|
| A | Classical CV rules | Establish what direct measurements can already detect |
| B | Independent binary logistic-regression models | Interpretable supervised baseline for each feature |
| C | Independent Extra Trees classifiers | Test nonlinear combinations of the same descriptors |

Train one binary model per feature. Exclude examples with unknown labels for that feature; do not encode unknown as absent.

Logistic regression defaults:

- Standardize descriptors using training data only.
- L2 regularization, `liblinear`, balanced class weights.
- Test `C = 0.1, 1, 10` on the synthetic selection partition.
- Maximum 2,000 iterations; nonconvergence is recorded as failure.

Extra Trees defaults:

- 256 trees.
- Balanced class weights and `max_features="sqrt"`.
- Compare maximum depths 16 and 32, and minimum leaf sizes 2 and 5.
- Fixed random seeds.
- At most four CPU workers, with feature models trained sequentially.

These algorithms are available in the inspected scikit-learn 1.6.1 environment. References: [logistic regression](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html) and [Extra Trees](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

Select configurations using synthetic average precision. Prefer logistic regression when its average precision is within 0.01 of the best tree configuration. Real evaluation labels do not determine the winner. Record configuration selection and per-feature outcomes.

No CNN is included in this first experiment. A neural classifier would require a subsequent plan justified by measured failures of these baselines.

### 3.4 Confidence and abstention

Calibrate learned scores with sigmoid calibration on data disjoint from model fitting. See [calibration documentation](https://scikit-learn.org/stable/modules/calibration.html); use APIs compatible with the pinned local dependency version.

For each feature derive two thresholds on the calibration partition:

- A positive threshold targeting at least 90% positive precision.
- A negative threshold targeting at least 90% negative predictive value.
- Scores between them produce `uncertain`.

Require at least 30 supporting examples for a threshold estimate. If valid separated thresholds cannot be obtained, keep the feature uncertain rather than manufacturing confidence.

Calibration on synthetic images does not guarantee calibrated confidence on real images. Report that limitation and measure real-image calibration where audited labels permit it.

## 4. Mapping observations to the feature bank

### 4.1 Separate learned recognition from explicit capability knowledge

The classifier learns: “This image contains evidence of curved interference.”

The catalog records: “These verified functions can draw curved interference, with these controls.”

The feature-to-function relationship is explicitly documented and tested. The model does not need to guess API names or identify the original CAPTCHA library.

```text
CAPTCHA image
    -> Original image + feature-enhancing views + overlapping regions
    -> CV measurements + independent feature classifiers
    -> Feature observations, uncertainty and approximate parameter evidence
    -> Filter verified catalog entries
    -> Rank compatible operations and example settings
    -> Return up to three candidates per supported feature
```

### 4.2 Initial mapping rules

| Observation | Candidate bank capabilities | Evidence used to narrow candidates |
|---|---|---|
| Straight interference | Gregwar line controls; verified Python straight-line operation | Direction, width, length and image location |
| Curved interference | Securimage curved lines; Python arc/sine operations | Curvature, span, thickness and density |
| Dot scatter | Securimage noise; Python dot operation | Mark size, density and contrast |
| Decorative shapes | Verified Python shape operations | Contour shape, size, fill and position |
| Periodic grid | Python background grid or stroke-grid operation | Spacing, orientation and stroke width |
| Texture | Licensed texture assets and supported background renderers | Local texture descriptors and contrast |
| Gradient | Verified Python gradient operation | Direction, color endpoints and smoothness |
| Blocks | Verified Python tiled/patch background operation | Patch size distribution and colors |
| Hollow lettering | Verified hollow-glyph operation or audited suitable font asset | Interior/background relationship and contour thickness |
| Glyph border | Verified border operation | Contour width and color contrast |
| Shadow | simple-php-captcha or verified Python shadow operation | Offset, contrast and softness |
| Text deformation | Gregwar/Securimage distortion or verified Python warp | Deformation descriptors and permitted strength range |

These are proposed mappings. Catalog membership requires capability tests. A documented parameter is not automatically a functioning parameter. For example, a renderer with a fixed one-pixel line must not be advertised as supporting arbitrary line thickness.

### 4.3 Candidate ranking

For each positive feature:

1. Filter to verified entries supporting that feature.
2. Remove entries incompatible with required output type or reliably estimated parameter bounds.
3. Compare observed descriptors with the entry's controlled example descriptors.
4. Rank by standardized descriptor distance.
5. Break ties using fewer unwanted side effects, then stable capability ID.
6. Return the best three with example images, evidence and limitations.

Fit descriptor scaling using the synthetic bank only. Exclude feature dimensions that are unavailable or unreliable. Parameter hints use normalized units where appropriate, such as stroke width divided by image height. Preserve broad ranges when estimates are uncertain.

For uncertain features, return explicitly tentative candidates. For unsupported features, return a bank gap. Never substitute an unrelated capability silently.

This phase does not choose a final whole-image recipe. Combining operations and optimizing their joint appearance belongs to the later generation experiment. Several implementations may be equally valid; the goal is capability matching, not recovery of the original source code.

### 4.4 Minimum interfaces

Add an isolated experiment namespace with these capabilities:

- `bank audit`: inspect resources, provenance and controls.
- `bank atlas`: render controlled examples and build descriptor records.
- `features train`: train and calibrate the specified classifiers.
- `features inspect`: emit observations and visual evidence for images.
- `features match`: return ranked bank candidates.
- `features evaluate`: produce synthetic and audited real-image reports.

Main records:

- **Capability:** identity, source revision, supported features, controls, output type, examples and verification status.
- **Observation:** feature ID, score, present/absent/uncertain state, evidence regions and measurement ranges.
- **Match:** observation ID, capability ID, example settings, ranking evidence and limitations.

Keep existing CLI behavior intact. Store each experiment in a new run directory. Version feature definitions, descriptor settings, manifests, models and bank records so an observation or match can be reproduced.

## 5. Validation, execution order and documentation

### 5.1 Evaluation annotations

For the 200 selected target images:

1. Draft feature labels without showing classifier predictions.
2. Record `present`, `absent` or `uncertain` for every feature.
3. Have the user audit the drafts.
4. Preserve both draft and audited labels.
5. Record disagreements and ambiguous examples.
6. Freeze annotations before real-image scoring.

These are assistant-assisted, user-audited annotations, not independently collected human ground truth. No CAPTCHA transcription is required. Never treat an unaudited draft as a verified real-image label.

### 5.2 Required tests

Bank integrity:

- Invalid parameters fail explicitly.
- Same recorded configuration and randomness reproduce output.
- Feature toggles affect the intended feature.
- Actual effects agree with saved labels and masks.
- Text is not silently replaced, clipped or rendered with an unrecorded fallback.
- Unchanged source snapshots retain their hashes.

Classification:

- Clean letters resembling lines, dots or circles are hard negatives.
- Localized features are detected without majority voting.
- Mixed features, low contrast, small images and both color polarities are tested.
- Unseen fonts, unseen combinations and withheld-renderer results are separate.
- Unknown labels and abstentions are handled explicitly.
- Split leakage and duplicate images are checked.

Mapping:

- Every returned capability exists and has passed capability checks.
- Unsupported ranges are rejected.
- Whole-image renderers are not presented as isolated layer functions.
- Equivalent implementations can both be valid matches.
- Removing a capability produces a visible coverage gap.
- Retrieval is not scored as original-library identification.

### 5.3 Metrics and acceptance

Report per-feature precision, recall, F1 and average precision; macro averages, class counts and uncertainty intervals; abstention rate and accuracy on non-abstained predictions; false-positive rates on clean/hard-negative cases; results per dataset and withheld source; runtime, memory, rejection counts and annotation disagreements.

Initial exploratory targets:

- Synthetic test macro F1: at least 0.85.
- Withheld-renderer macro F1: at least 0.75 on shared, verifiable features.
- Audited real-image macro F1: at least 0.75 on sufficiently represented features.
- Decision coverage: at least 70%, alongside reported selective accuracy.

A feature with insufficient positive or negative examples receives **insufficient evidence**, not a pass. A good macro score cannot conceal a failing individual feature or dataset. Report support counts and uncertainty for rare features, particularly with only 40 annotated images per dataset.

Mapping acceptance requires valid, compatible capabilities and traceable ranking evidence. A low descriptor distance alone does not establish successful CAPTCHA recreation. If only a partial feature set is supported, report the completed subsystem and unsupported cases explicitly; do not declare the entire intended coverage achieved.

### 5.4 Execution order

1. Freeze dataset references, feature definitions and annotation protocol.
2. Audit resources and produce controlled pilot galleries.
3. Build the verified catalog and labeled synthetic corpus.
4. Evaluate the classical CV baseline across all five datasets.
5. Train/evaluate logistic regression across all five.
6. Train/evaluate Extra Trees across all five.
7. Freeze selected classifiers and evaluate feature-to-bank matching.
8. Publish the comparison, failure analysis and recommendation for a later generation phase.

Complete and document each stage's five-dataset review before moving to the next algorithm. Gate failures remain visible; do not tune on the audited evaluation labels to manufacture a pass. Pending user audits may allow independent bank/infrastructure work, but do not permit claiming an audited evaluation or passed gate.

### 5.5 Markdown records

This master planning document should link to separate records for resource inventory/provenance/licenses; feature definitions and annotation examples; adapter architecture and capability checks; descriptors and classifier methods; synthetic corpus construction/splits; results/visual reviews/failures; and feature-to-function mappings/unsupported capabilities.

Every result must distinguish **READ**, **OBSERVED**, **INFERRED** and **PROPOSED** statements. Link claims about local behavior to source files, run artifacts and tests. Link external claims to primary documentation or papers.

For each research citation record the idea adopted, what was actually read, implementation differences and evidence limits. Alongside HOG and the software documentation, [Jaderberg et al., 2014](https://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14c/jaderberg14c.pdf) supports varied, layered synthetic text rendering. The relevant synthetic-data-engine section was consulted. Its scene-text results do not demonstrate that this proposed CAPTCHA classifier or mapping system will work; this plan does not adopt its use of real image crops as rendering assets.

The outcome of this experiment is a documented answer to:

> Which CAPTCHA features can we recognize reliably, and which verified rendering resources can express each one?


## User clarification and region demonstration — 2026-09-28

The intended recognition path is local bricks → spatially connected feature evidence → library functions → later style approximation. The requested demonstration uses **10 pieces across and 3 down**, with overlapping crops and whole-image context retained. [Region demonstration and integrity protocol](feature_bank/REGIONS_02.md) · [Interactive explorer](../runs/feature_bank_regions_02/explore.html). This implements geometry only; no classifier, matching, or reconstruction gate has passed. Synthetic base groups must be split before cropping, and labels must describe visible evidence within each crop.


## Local-label mechanism pilot — 2026-09-28

[Small labeled-brick pilot and revisions](feature_bank/LABELED_BRICKS_03.md): 64 images / 1,920 local crops from eight synthetic base groups; straight lines, curves and dots have final contribution masks and conservative local labels. Effect-color shortcuts found in the first design were corrected before training. Only 1,139 unique crops; no training or full-corpus gate passed. Unknown features and ambiguous fragments remain unknown/uncertain. Earlier revisions are retained.


## Candidate corpus and parameter proposal — 2026-09-28

[Label repair and expanded corpus](feature_bank/CORPUS_CANDIDATE_01.md): 5,600 synthetic parents, 168,000 bricks, 119,292 retained unique crops; grouped partitions, withheld font/combination, two quality rejections preserved. Integrity/replay checks passed; corpus acceptance remains partial and no training started. [Density and tilt after classification](feature_bank/FEATURE_PARAMETERS.md) records the user-proposed extension, its feature-specific definitions and validation gates; estimators are not yet implemented.
