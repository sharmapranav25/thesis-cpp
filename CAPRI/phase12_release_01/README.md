# Phase 1 + Phase 2 generation release 01

Start here. This is the supported self-contained generation entrypoint for the next session. It contains frozen models, independent fonts with licenses, CV/descriptor/rendering code, and eight saved family profiles. **No PARSeq training is performed here.** No count predictor, DINO, GAN/refiner, target-family generator, OCR or runtime VLM is invoked.

The package can be moved as a unit. Python plus the dependencies in `requirements.txt` are required. `environment-tested.json` records the tested environment, including CUDA version; choose an appropriate PyTorch wheel for your hardware. Existing tested interpreter on this machine: `E:/pranav/model 1/.venv/Scripts/python.exe`. This is only an interpreter path, not a dependency on Model1 training code or weights. CPU generation works; profiling supports CPU/CUDA. Do not install dependencies over an existing working environment unnecessarily.

From this folder, with the selected Python environment:

```powershell
python pipeline.py verify
python pipeline.py list
python pipeline.py generate --family cryptbb --partition train --count 210 --seed 1001 --out ../previews/cryptbb_01 --recipes --masks
```

Output paths must not exist. Failure never silently overwrites an experiment. Inspect `preview.png` and `status.json` first. Generate a preview for each family before bulk generation. Choices: `cryptbb`, `city`, `courier`, `darkbay`, `outlined_text_captcha`, `roncuevas`, `pymlex`, `sloth`.

## Generation policy

- Exactly21 cells: numeric/alphabetic/alphanumeric, each length2–8. Requested counts are balanced to within one image per cell;210 gives10each,100000 gives4761or4762each.
- Numeric includes leading zeros. Alphabetic covers upper/lower/mixed case. Alphanumeric requires at least one letter AND one digit. All62 ASCII alphanumeric symbols are eligible. Case subtypes are randomly sampled, not separate equally sized quotas.
- These are generic priors; no real length/type inference or forced family whitelist.
- `--mode baseline` (default) reproduces the accepted Phase1+Phase2 behavior with the saved selected font/effect. `--mode diverse` explicitly adds bounded height/spacing/noise variation and chooses another independently sourced bank font30%of the time. This is a proposed training diversity arm, NOT a proven improvement. It does not implement the earlier proposed30%fully generic family-agnostic branch; that remains a separate future experiment.
- Real references contribute style measurements only. Synthetic labels are freshly generated. No copied reference crops, actual real labels or target-family generator code are used.
- Each text is assigned to train/validation/test by stable hash (90/5/5). Repeated text within one partition is allowed; identical text cannot cross partitions. Seeds control variation. Two-digit numeric vocabulary has only100possible strings, so its validation/test label pools are small. Do not claim image independence just from text disjointness; run the pixel audit too.
- Built-in rendering rejects clipping/low individual glyph visibility; generation retries up to100per sample. Every retry is logged. Failure aborts instead of emitting blank images or wrong labels.

## Bulk corpus example (next session)

```powershell
python pipeline.py generate --family cryptbb --partition train --count 100000 --seed 1001 --mode diverse --out ../corpora/cryptbb_train_01
python pipeline.py generate --family cryptbb --partition validation --count 5040 --seed 2001 --mode diverse --out ../corpora/cryptbb_validation_01
python pipeline.py generate --family cryptbb --partition test --count 5040 --seed 3001 --mode diverse --out ../corpora/cryptbb_synthetic_test_01
python pipeline.py audit --corpora ../corpora/cryptbb_train_01 ../corpora/cryptbb_validation_01 ../corpora/cryptbb_synthetic_test_01
```

Repeat one family at a time; do not pool family corpora by accident. Generation is CPU-bound; full100k throughput has not been benchmarked. Bulk corpus and PARSeq training have NOT been run as part of packaging. `--recipes` stores full drawing operations and can be large; default manifest records reproducible seeds, selected profile/font, controls, labels and pixel hashes. `--masks` stores glyph masks when needed. `labels.csv` has `filename,label` with image paths relative to the corpus directory. `manifest.jsonl` provides the audit data. Preserve the package integrity hash alongside the corpus.

## New/revised family profiling

Create an explicit reference manifest containing only approved style-fit images:

```json
{"partition":"style_fit","images":["relative/path/to/reference1.png","relative/path/to/reference2.png"]}
```

Paths are relative to that manifest, or absolute. Do not point it at an unsplit whole evaluation folder.

```powershell
python pipeline.py profile --references references.json --name new_family --out ../profiles/new_family_01 --device cpu --variant 0
python pipeline.py generate --profile ../profiles/new_family_01/profile.json --partition train --count 210 --seed 1001 --out ../previews/new_family_01
```

This performs fresh native3rows×10columns overlapping-brick descriptors, three frozen calibrated feature classifiers, fresh global/block CV, and frozen font inference. Variant0is the conservative generic default; existing profiles retain their historical fit-selected variant. No new-family search or tuning against evaluation happens automatically. Filename/text is not an inference input. Duplicate reference pixels are deduplicated.

## Real evaluation boundary

An evaluation manifest uses `{"partition":"evaluation","images":[...]}`. Add `--evaluation-manifest evaluation.json` to `audit`. It rejects pixel overlap with the reference hashes used to generate those corpora. It does NOT establish that old experiments never saw those images, establish trustworthy ground truth, or audit prior model-selection exposure. Consult `TRAINING_HANDOFF.md`. Saved profiles are historically exposed development profiles, not a clean new test split. `reference_path_provenance` and source paths are metadata only; generation never opens them. Regenerate a profile from explicit fitting references if a new split is required.

## Honest limitations

Only straight interference, curved interference and dot scatter have learned feature models. Their real-domain predictions can saturate; any positive block currently makes the whole image positive. Other style measurements and cross-block reconciliation are heuristics, not a trained relational transformer. Tilt is a proxy, not verified glyph rotation. Fonts/outline estimates are unreliable for several families. The historical Phase1quality gate was not passed; this is a usable experimental implementation, not a claim of accurate family reproduction. Pymlex has4profile images and Sloth16; the other six have96in this release.

Hard-failed Phase3 variants and the count model are intentionally absent. Their historical directories remain preserved outside this release. Do not silently reactivate them. Do not fix a real-data mismatch by importing a target's generator/font or tuning on final labels. Record limitations/failures rather than inventing results.

## Files and maintenance

`pipeline.py` is the entrypoint. `assets/` contains the three classifiers, thresholds, font model, fonts and license notices. `profiles/` contains measurements and frozen inference provenance. `PROVENANCE.json` maps copied files to original source and SHA256. `INTEGRITY.json` verifies shipped files. Source paths in provenance do not create runtime dependencies. This is a snapshot release: make a new version for changes, retain this version for reproducibility. `VALIDATION.md` records executed checks; it is distinct from a real recognition evaluation.
