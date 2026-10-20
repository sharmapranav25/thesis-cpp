# City renderer controller pilot

The controller reduced held-out appearance distance by **8.02% versus the simple hybrid starting settings** and **7.28% versus the specialist starting settings**. This is a useful proof that an inexpensive feedback loop can improve the renderer distribution. It has not established better font accuracy, and the generated images still differ substantially from City.

The main run took 234.7 seconds (3.91 minutes) on an RTX A4000. The additional diagnostic renders and verification took 16.7 seconds. These times cover execution, not setup or development. No neural weights were trained or updated, and no CAPTCHA solver was trained. The simple and specialist checkpoints and the cached feedback model passed the unchanged-hash check.

| Starting settings | Appearance distance ↓ | DINO distance ↓ | Statistics distance ↓ | Independent audit ↓ | Usable | Strict readability |
|---|---:|---:|---:|---:|---:|---:|
| Simple hybrid start | 0.42837 | 0.37130 | 0.51398 | 2.8465 | 100.0% | 85.4% |
| Specialist hybrid start | 0.42495 | 0.35886 | 0.52410 | 2.8490 | 100.0% | 76.0% |
| Specialist with controller | 0.39400 | 0.34034 | 0.47449 | 2.2789 | 100.0% | 77.1% |

Each generated group contains 96 fresh images, compared with 512 held-out City images. The adjusted specialist and its unadjusted baseline use the same 96 strings and seeds. Every main group has 96 unique images, 96 unique strings and complete glyph labels. Usable and strict readability are renderer checks on known synthetic glyphs; they are not OCR success rates. The full controller passes strict readability for 74/96 images, versus 73/96 for the specialist start and 82/96 for the simple start.

The independent grayscale and edge audit fell by 19.94% versus the simple start and 20.01% versus the specialist start. It was excluded from controller selection. A paired bootstrap with 200 resamples put the change in the primary appearance distance at -0.04358 to -0.01903 (95% interval; after minus before). This interval covers sampling variation for one fixed controller and one run, not uncertainty across searches or target families.

![Pilot metrics](metrics.png)

The target pool contained 2,000 City images and 1 exact pixel duplicate. After deduplication, the seeded split used 256 images for fitting, 128 for selection and 512 for the final test; the remaining images were unused. Both starting profiles were built from predictions for the same 256 fitting images. Pixel hashes prove that the three subsets are disjoint. Test scores were computed after saving the selected controls. No real answers or target font files contributed to the objective.

The search used 15 legal renderer controls, eight rounds of eight candidates and 24 common recipes per candidate: **1,536 search renders**. Recipe preparation uses the existing measured-label mapper and performs additional renderer calls; validation, final evaluation and diagnostics also add renders. Thus 1,536 is the search budget, not the total number of renderer calls. The three best fitting candidates were evaluated with 48 fresh recipes against the separate selection subset. Candidates had to stay within five percentage points of neutral usability and strict readability. The third-ranked fitting candidate won selection; one higher-ranked candidate failed readability. The test did not trigger further tuning.

Appearance feedback combined 60% frozen DINO ViT-S/8 feature distance and 40% colour, intensity and projection statistics, using multi-kernel maximum mean discrepancy. The existing DINO model has about 21.7 million parameters, but it was loaded from the local cache and kept frozen. Normalisation and kernel scales came only from the fitting subset. A population search adjusted global settings; it did not learn an image-conditioned controller. This is a small experiment inspired by [Meta-Sim](https://arxiv.org/abs/1904.11621), rather than a reproduction of its neural training method.

The selected controls narrowed text to 0.756 times its initial horizontal scale and reduced the nominal size to 0.943 times its initial value. They added 0.055 spacing and 0.050 vertical jitter, changed the variable-weight fraction by +0.137, lowered requested band strength by two levels, multiplied band slope by 0.873 and kept curves with probability 0.821. Background colours were blended 19.6% toward a light gray, and foreground colours 3.2% toward a dark gray. Rounding selected no font replacement and no additional synthetic bold stroke; the probability of forcing effect palettes to gray was zero.

All 96 paired samples retained their starting font lists and individual glyph font keys. Category, stroke class, x height, slant and italic choices therefore stayed unchanged. Weight and width labels changed because they are measured from the rendered glyphs. A positive variable-weight fraction adjustment does not imply heavier final glyphs when size and width change together.

| Rendered font attribute | Simple start | Specialist start | Adjusted specialist |
|---|---:|---:|---:|
| Mean image median weight level (0–10) | 4.302 | 4.448 | 4.000 |
| Mean image median width level (0–10) | 5.677 | 6.062 | 3.281 |
| Mean image median x height level (0–10) | 5.594 | 5.323 | 5.323 |
| Mean image median slant level (0–10) | 0.000 | 0.000 | 0.000 |
| Italic images | 4.2% | 0.0% | 0.0% |

These are synthetic font measurements, not errors against real City font labels. The typography question remains unresolved: the trial mainly changed geometry within the existing font choices. In the first 12 consecutive test examples, coloured backgrounds, pastel bands, outlines and varied font styles remain conspicuous beside City's gray bands and dark digits. Some adjusted text is narrower, but the overall family resemblance is still weak. No examples were selected for visual quality.

![Held-out references and renderer examples](comparison.jpg)

After the main result was frozen, three additional groups applied only its colour, text or effect controls to the same 96 specialist recipes. These are descriptive checks on the already-scored test subset, with no search, retuning or new selection.

| Frozen controls applied | Appearance distance ↓ | Reduction from specialist start | Strict readability | Independent audit ↓ |
|---|---:|---:|---:|---:|
| Colour only | 0.42355 | 0.33% | 79.2% | 2.5086 |
| Text only | 0.41283 | 2.85% | 69.8% | 2.7552 |
| Effects only | 0.40858 | 3.85% | 81.2% | 2.7745 |
| All controls | 0.39400 | 7.28% | 77.1% | 2.2789 |

The text-only group reduced the appearance proxy by 2.85%, while strict readability fell from 73/96 to 67/96. That does not demonstrate an all-font accuracy gain. Renderer operations share a seeded random stream, so changing effect count or level can also change later colour and style draws. The ablations describe which settings were applied; they cannot cleanly partition the gain into independent causes.

The pilot added foreground/background colour hooks and a band-slope hook within its worker processes. These were extensions to the exposed controls; original renderer source files were not edited. Labels and readability were measured after each render. Neutral controls reproduced the original renderer exactly in the saved replay check. A stronger next experiment should improve feedback for typography and family style, and compare it with stronger random search or other controllers at equal renderer budgets. The current result does not justify a large GAN training run yet. CryptBB, DarkBay, real OCR accuracy and labelled real font accuracy remain untested here.

The saved evidence is [results.json](results.json), [diagnostics.json](diagnostics.json), [selected_controls.json](selected_controls.json), [verification.json](verification.json), [protocol.json](protocol.json) and [environment.json](environment.json). Run the experiment with the existing project Python environment and `launch/pilot.py --out <new directory>`. It uses local project code, font assets, cached model predictions and cached DINO weights. Run `launch/diagnostics.py --out <completed directory>` for the fixed-control checks, and `launch/write_report.py --out <completed directory>` to regenerate this report. The fitted controller applies to City and has not been installed as the production default.
