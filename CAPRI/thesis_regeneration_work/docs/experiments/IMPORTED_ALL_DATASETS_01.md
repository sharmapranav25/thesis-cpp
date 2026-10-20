# IMPORTED-ALL-DATASETS-01 — fresh C1/C2 rendering test

Date: 2026-09-27 America/Los_Angeles. Requested after the five-dataset A/B preview.

## What was actually tested

The test imported `E:/pranav/Final sem/CaptchaThesis/experiments/gen_ablation/generate_lmdb.py` and its `gen_gateonly` dependencies directly from the original repository. It called `generate_one()` for C1 random difficulty and C2 curriculum, requesting 16 images per mode in each of five dataset contexts: **160 saved images**. This is an execution and internal-label test of an existing generic renderer. It is **not** regeneration learned from reference images. Evidence: [module paths, source hashes and harness identity](../../runs/imported_all_datasets_01/source_receipt.json), [summary](../../runs/imported_all_datasets_01/summary.json), and [test harness](../../scripts/test_imported_generation.py).

## Family configuration: what the name means

READ from [original families.py](<E:/pranav/Final sem/CaptchaThesis/experiments/gen_ablation/families.py:18>):

| Family | Allowed characters | Length | Dimensions recorded in FamilySpec |
|---|---|---:|---|
| CryptBB | 0–9, A–Z, a–z | 5 | 200 x 60 |
| City | 0–9 | 5 | 318 x 79, 324 x 81, 788 x 198 |
| DarkBay | 0–9 and A–Z excluding I | 6 | 200 x 50 |
| MGM | 0–9, A–Z, a–z | 5 | 120 x 40 |
| Courier | No family entry | — | — |
| Downloaded outlined-text | No family entry | — | — |

READ: [generate_one()](<E:/pranav/Final sem/CaptchaThesis/experiments/gen_ablation/generate_lmdb.py:57>) uses the supplied alphabet and length. It does **not** consume `FAMILY_DIMENSIONS` or any real image. The visual schema is sampled independently. [UniversalCaptchaGenerator](<E:/pranav/Final sem/CaptchaThesis/system1_synthetic_generator/previously_produced_generators/gen_gateonly/generator.py:50>) explicitly describes generic generation without targeting particular families. Therefore the tested family configuration is a text constraint, not a font/background/noise/spacing recipe.

The alphabet files live under `CaptchaThesis/system5_clean_baseline/charsets/`. `families.py` calls them canonical evaluation-protocol definitions: these are supplied text priors, not characters discovered from unlabeled pixels. No individual evaluation labels or reference-image transcriptions were used in this run. This baseline must not be presented as discovery under the image-only thesis constraint.

MGM was not tested because it is not one of the five available image collections in this comparison. For Courier and outlined-text, the harness explicitly used the imported generic mixed-case alphanumeric alphabet and left text length to the imported schema; it did not invent target-specific rules. Those 64 images demonstrate generic rendering only. See each cell's `support`, `charset`, and `length_policy` in [summary.json](../../runs/imported_all_datasets_01/summary.json).

## Execution and environment adaptation

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B scripts/test_imported_generation.py --out runs/imported_all_datasets_01 --count 16
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B scripts/verify_imported_instrumentation.py
```

Executed in `C:/Users/pranav/Documents/ChatGPT/thesis/thesis_regeneration_work`, then published to `E:/pranav/Final sem/sourcefree_synth`. Python 3.9.13, Pillow 11.3.0, NumPy 2.0.2. The ten generation/measurement cells took 12.22 seconds in total; this timing excludes subsequent visual review, publication and observer-equivalence checking. No training or VLM runtime calls occurred. [Environment and timings](../../runs/imported_all_datasets_01/summary.json).

The original font whitelist accepted **zero** fonts even with the local generic font directory configured. The harness mapped whitelist font filenames to 12 installed generic fonts, checked full alphanumeric glyph coverage and recorded hashes. Mapping was in memory; original source/whitelist files were unchanged. This was an environment adaptation, so it is not an as-is-environment success. [Preflight](../../runs/imported_all_datasets_01/as_is_preflight.json), [font mapping](../../runs/imported_all_datasets_01/font_path_mapping.json), [source receipt](../../runs/imported_all_datasets_01/source_receipt.json).

## Results and limits

| Dataset context | C1 / C2 images | Matching reference dimension, C1 / C2 | Color-histogram overlap, C1 / C2 |
|---|---:|---:|---:|
| CryptBB | 16 / 16 | 0 / 0 | 0.68 / 0.52 |
| City | 16 / 16 | 0 / 0 | 0.39 / 0.44 |
| Courier — generic fallback | 16 / 16 | 0 / 0 | 0.53 / 0.55 |
| DarkBay | 16 / 16 | 0 / 0 | 0.17 / 0.19 |
| Outlined-text — generic fallback | 16 / 16 | 1 / 0 | 0.25 / 0.26 |

READ from [per-case results collected in summary.json](../../runs/imported_all_datasets_01/summary.json): every requested image was saved and reopened. No exported/internal-label mismatch, stale-image return, or returned-image degeneracy flag occurred in this small run. There were no exact duplicate outputs within any of the ten cells. These checks do not prove text readability or rule out the historical stale-state bug in other runs.

**83 of 160 calls** changed the requested schema during retry. This preserves the original retry behavior; it also limits treating C1/C2 as strictly controlled difficulty distributions. Per-output `*_trace.json` files preserve every attempted schema and quality metadata; `result.json` records which attempt supplied the final image. The [generated report](../../runs/imported_all_datasets_01/REPORT.md) contains the counts for each cell.

The histogram metric is equal-image-weighted global RGB overlap with the existing reference validation profile. It is **not** accuracy, readability, a similarity pass threshold, or evidence of matching spatial structure. Only 1/160 generated canvases had a width/height pair present in its corresponding validation collection. No generated image was resized before measurement; proportional resizing is only for previews. Measurement definition: [pixel_statistics.py](../../sourcefree_synth/pixel_statistics.py).

An independent re-execution without any stage or `generate()` observers reproduced all 32 labels, dimensions and pixel hashes in the CryptBB C1 and City C2 cells. This supports observer noninterference for those two cells only. [Verification receipt](../../runs/imported_all_datasets_01/observer_equivalence.json).

## Qualitative review using assistant vision

The assistant opened the combined A/B panel, all ten 16-image output sheets, and two detailed process panels. This is qualitative visual review, not an OCR accuracy experiment. Review had no influence on this run's parameters or selection: A/B uses the first generated sample in each mode and two previously selected validation references.

OBSERVED in the [A/B panel](../../runs/imported_all_datasets_01/all_datasets_ab.png) and each dataset's C1/C2 `preview.png`:

- CryptBB: generated styles vary between clean backgrounds, grids and heavy clutter; the reference pair has dense speckles and colored tiles/grid structure. Some generated examples share individual ingredients, but no matching distribution is established.
- City: the configured five-digit text constraint is followed internally, while colored/thick/outlined glyphs and dark or grid backgrounds differ from the reference's thin dark digits and gray horizontal bands.
- Courier: generic fallback outputs include letters and variable lengths, whereas the displayed reference pair contains four digits with crossing strokes. This observation is not a new automatic family rule and was not used to change generation.
- DarkBay: six-character text is constrained by the supplied alphabet, but the generic outputs vary broadly in fonts, backgrounds and clutter instead of consistently reproducing the reference's rectangular tonal patches and mixed light/dark text.
- Downloaded outlined-text: the displayed generated pair uses filled glyphs without the reference pair's thin outlines and prominent dark crossing line. Occasional outlined generic glyphs elsewhere in the batch do not establish a learned target style.

Some outputs have glyphs that look nearly filled-in despite the internal gate passing: examples include [City C2 sample 012](../../runs/imported_all_datasets_01/city/c2_curriculum/012.png), [Courier C2 sample 012](../../runs/imported_all_datasets_01/courier/c2_curriculum/012.png), and [DarkBay C1 sample 012](../../runs/imported_all_datasets_01/darkbay/c1_random/012.png). This is an observed readability concern, not a quantified error rate.

Process panels show background → background noise → text → foreground noise → post-processing → returned image. Opened examples: [CryptBB C2 sample 001](../../runs/imported_all_datasets_01/cryptbb/c2_curriculum/001_process.png), [outlined-text C1 sample 001](../../runs/imported_all_datasets_01/outlined_text_captcha/c1_random/001_process.png). Panels capture the selected final attempt, while retry traces preserve all attempted schemas; they are not a movie of every retry.

## Verdict

**Execution and internal-label checks pass for this bounded run after font-path adaptation. Dataset regeneration is not achieved by the tested C1/C2 path.** It has no mechanism here that converts reference-image features into a family rendering recipe. Courier and downloaded outlined-text have no imported family configuration. Successful file generation must not be reported as successful regeneration.

This run did not test the separate VLM-based code-generation workflow, learn C3 weights, train OCR, fix the imported renderer, or advance Methods 2–6. Those are different tasks. Future feature-to-renderer work must be assessed separately from this generic baseline.
