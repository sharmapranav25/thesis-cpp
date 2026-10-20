# Sourcefree Synth

An experimental generator fitted from **unlabeled real images**, using generic fonts and procedural rendering. Two fitting modes share the same renderer:

| Mode | During fitting | During generation |
|---|---|---|
| `no-vlm` | Image statistics and bounded parameter search; optional frozen DINOv2-small features | CPU Pillow + NumPy |
| `light-vlm` | Same search, with local Qwen2-VL-2B providing bounded style hypotheses | CPU Pillow + NumPy; VLM no longer needed |

This is a new, standalone project. It does not import the target CAPTCHA source, target fonts, source-trained recognizers, or the legacy generators. Existing files under `CaptchaThesis` and your earlier attempts are unchanged.

**Status:** CPU pipeline and its invariants are tested with artificial fixtures. No real-data fitting, bulk generation, recognizer training or model download was performed during implementation. The optional VLM/DINO paths require a separate runtime test. Style-distance scores are diagnostics, not evidence of recognition accuracy.

## Start here on this machine

Open PowerShell in `E:\pranav\Final sem\sourcefree_synth`.

```powershell
$py = 'E:\pranav\model 1\.venv\Scripts\python.exe'
& $py -B -m sourcefree_synth doctor
```

The existing interpreter has the CPU dependencies. For a different environment, install `requirements.txt` there; install `requirements-models.txt` only for the optional model paths. Do not upgrade your shared research environment unnecessarily.

### 1. Read-only inventory and overlap check

```powershell
$real = 'E:\pranav\CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001\CryptBBCaptchas(unsolved)07272023'
$eval = 'E:\pranav\model 1\data\eval'
& $py -B -m sourcefree_synth inspect --real-dir $real --exclude-dir $eval
```

The pipeline reads image pixels, not labels.csv or filename text. Identical decoded images are deduplicated before splitting. Any exact decoded-pixel overlap with an excluded directory stops the run. Supplying an evaluation directory is strongly recommended; omitting it means no external overlap check is possible. This check does **not** prove the absence of cropped, altered or near-duplicate images.

### 2. First generation test: no VLM, 64 outputs

```powershell
.\preview.ps1 -RealDir $real -ExcludeDir $eval -Mode no-vlm -Count 64
```

This command actually runs fitting and generation. It creates fresh, uniquely named `runs\..._fit` and `runs\..._preview` folders. It never replaces an existing run. The wrapper uses 12 trials per style and 16 probes per trial: a small functionality/visual test, **not** the full experiment.

Open these files after the run:

- `..._fit\real_validation.png`: the real images kept out of fitting.
- `..._preview\preview.png`: synthetic output with known labels.
- `..._fit\report.json`: fit/validation style gaps, real-vs-real reference and fresh-seed acceptance.
- `..._preview\status.json`: must say `complete` before the dataset is used.

**Text priors are explicit and unresolved.** The defaults sample ASCII letters/digits and lengths `4,5,6,7` uniformly. They are generic hypotheses, not recovered facts about the target. The pipeline does not infer length from connected components or read the real evaluation labels. If an independent, permitted specification supplies an alphabet/length constraint, use `-Charset` / `-Lengths`; otherwise keep the uncertainty and compare hypotheses later. Do not copy restrictions from the target source code or labeled evaluation set.

Example of declaring an independently justified prior (not a recommended target preset):

```powershell
.\preview.ps1 -RealDir $real -ExcludeDir $eval -Lengths '4,5,6' -Charset '0123456789'
```

If PowerShell blocks local scripts, use the equivalent Python commands below instead of changing your machine-wide execution policy.

### 3. Fit and generate separately

```powershell
& $py -B -m sourcefree_synth fit --real-dir $real --exclude-dir $eval `
  --out runs\experiment_a_fit --mode no-vlm --trials 24 --samples 32 --clusters 3

& $py -B -m sourcefree_synth generate --config runs\experiment_a_fit\generator.json `
  --out runs\experiment_a_preview --count 64 --seed 123

& $py -B -m sourcefree_synth audit --run runs\experiment_a_preview
```

Check each command's exit status before continuing. Each output path must be new. Generation refuses a config whose adjacent fitting `status.json` indicates failure. Large pools should wait until visual quality, acceptance and prior choices are reviewed. To scale later, use a new output folder and a larger `--count`; fitting need not be repeated.

### 4. Lightweight VLM variant

```powershell
.\preview.ps1 -RealDir $real -ExcludeDir $eval -Mode light-vlm -Count 64
```

This uses **Qwen/Qwen2-VL-2B-Instruct**, CUDA, BF16 where supported, one image per call, at most 256 visual tokens, and 160 output tokens. It selects 24 diverse **fitting** images by default. The model only suggests background type, distortion severity, outline, shadow and color variation. It neither transcribes images nor supplies synthetic labels. Invalid responses are logged; if all responses fail, fitting stops rather than silently becoming a no-VLM run.

Weights must already be cached under `E:\hf_cache\hub`, or supply `-VlmModel 'E:\path\to\local\model-snapshot'`. No network requests to an inference service are made. Missing cached weights cause an error. Public weight downloads require the Python CLI's explicit `--allow-download` option. No model downloads or fine-tuning are automatic.

The VLM is unloaded before optional DINO fitting. Peak allocated VRAM is reported in `vlm_observations.json`. This configuration is designed for the 16 GB budget, but its actual peak and behavior on this machine still need to be measured.

### 5. Optional learned image features, in either mode

Add `-Features dinov2` to the wrapper or `--features dinov2` to `fit`. This loads frozen `facebook/dinov2-small` locally, preserves the full image with square padding and adds mean patch embeddings to the statistical metrics. Inference uses batches of eight. It does not use your source-trained OCR models. Start with `pixels` for the first execution test; compare learned features as a separate ablation.

## What the code does

1. Decode and hash all input and excluded images. Deduplicate; deterministically split by pixel identity, independent of filenames (80% fitting, 20% validation).
2. Measure per-image color palettes, color histograms, edge distributions, spatial edge layout and texture autocorrelation. Cluster broad style modes using fitting images only. No character crop bank, inpainting or real background is used.
3. In `light-vlm` mode, collect validated style hypotheses from representative fitting images. Suggestions only bias the candidate search; they cannot inject code, labels or unbounded parameters.
4. Search background, typography, geometry and joint parameters in stages. Probe each candidate with the same chosen strings and seeds. Use distribution means/spreads plus symmetric nearest-feature coverage; penalize rejected renders. Retain a small Pareto shortlist near the best training objective for each style.
5. Freeze `generator.json` **before** computing validation diagnostics. Store observed style proportions and joint per-image palette/canvas profiles. No validation labels or validation-based candidate selection occur.
6. Choose a fresh string, style, candidate and profile; draw all pixels procedurally at native resolution. Apply global warps consistently to text, background and masks.
7. Reject empty, clipped, excessively overlapping, crushed or low-contrast characters. Retries retain the same text, parameters and palette profile. Exhaustion raises an error; it never returns a stale image or switches to an unrelated generator.
8. Write known labels, image hashes, masks, rendering parameters, font hashes, sample/render seeds and quality records. `audit` verifies structural integrity independently.

The empirical profile colors are coarse statistics; actual real-image pixels, glyph crops and filenames do not enter rendering. Because rendering rejection changes the accepted distribution, the report includes fresh-seed acceptance and whole-mixture diagnostics. Failed runs remain visible for debugging and must not be used for training.

## Outputs

```text
..._fit/
  split.json               # image paths/hashes for audit, fit/validation membership
  fonts.json               # generic fonts accepted/rejected and hashes
  trials.jsonl             # every candidate, scores, acceptance and failure examples
  vlm_observations.json    # light-vlm only: raw replies, parsed hints, peak VRAM
  generator.json           # frozen configuration; self-contained except font files
  report.json              # image-only diagnostics, limitations, no recognition claims
  fit_preview.png          # small diagnostic renders created by fit
  real_validation.png      # separate real-image panel
  status.json              # running/complete/failed, elapsed time and errors

..._preview/
  images/                  # PNG files
  masks/                   # union character masks after warp
  character_masks/         # optional --char-masks
  labels.csv               # filename,label (filenames relative to this run folder)
  samples.jsonl            # aligned metadata and per-character quality records
  generator.json           # copied frozen config
  preview.png              # first 64 images with chosen strings
  status.json              # requested/written counts and completion state
```

Font files are referenced, not bundled. Default discovery uses a small allowlist of generic Windows fonts or DejaVu/Liberation on Linux. All selected fonts must cover the requested charset. There is no silent fallback to Pillow's default font. `--font-dir` accepts an explicitly generic collection; responsibility for its provenance remains with the experiment owner. Do not supply target font files.

## Validation and limitations

```powershell
& $py -B -m unittest discover -s tests -v
```

The tests use temporary artificial images and make no network/model calls. They cover pixel-based leakage checks, filename-invariant splits, retry isolation, deterministic image/label alignment, low-contrast rejection, schema validation, and an end-to-end fit/generate/audit run including tamper detection.

This version has **no demonstrated downstream accuracy**. In particular:

- It uses a Pillow renderer; PHP-GD equivalence and renderer-backend comparisons are not implemented.
- Generic glyph shapes can still be the main mismatch. More real examples or a small VLM cannot guarantee recovery of an unavailable font.
- Segmentation-free descriptors are deliberately imperfect proxies. A lower style gap is not necessarily better recognition.
- Alphabet/length recovery, VLM pseudo-label anchors, OCR training and checkpoint selection are not implemented. Priors are visible CLI inputs instead of hidden scheme-specific constants.
- Quality gates detect geometric/contrast failures; they do not certify that humans or recognizers can read every assigned string.
- If the search cannot find viable configurations or a generation sample exhausts retries, the run stops with diagnostics. Do not remove gates to get a full dataset; inspect the renderer/search mismatch first.
- Existing evaluation sets have a history of reuse. Keep their labels outside this pipeline; a final unbiased research claim may require a fresh test set.

For the actual comparison, hold real-image split, text prior, generic font collection, search budget, synthetic count, recognizer initialization and training schedule constant across modes. Score a separately trained recognizer on held-out labeled real data only after fixing model selection. That experiment is subsequent work.

## Implementation files

| File | Purpose |
|---|---|
| `sourcefree_synth/data.py` | Image-only ingestion, hashes, exclusions, split and output protection |
| `sourcefree_synth/fonts.py` | Generic font discovery, character coverage and immutable font checks |
| `sourcefree_synth/features.py` | Pixel/DINO descriptors, style clustering and distribution metrics |
| `sourcefree_synth/renderer.py` | Procedural rendering, masks, quality gates and isolated retries |
| `sourcefree_synth/vlm.py` | Optional local 2B style observer with strict JSON schema |
| `sourcefree_synth/pipeline.py` | Parameter search, freeze, diagnostics and dataset generation |
| `sourcefree_synth/__main__.py` | `doctor`, `inspect`, `fit`, `generate`, `audit` commands |
| `preview.ps1` | Small fit + 64-image generation + audit workflow |
| `tests/test_invariants.py` | Artificial-fixture validation |

## Design provenance

This implementation carries forward the procedural rendering and bounded search ideas inspected in `CaptchaThesis/experiments/gen_ablation/` and the user's earlier generic renderer work. It does not import those generators or their configurations. It specifically avoids the observed stale-image retry path, component-count-derived labels, noisy OCR crop-bank labels, real-background ghost artifacts and real-label checkpoint selection.

Optional model APIs follow primary documentation:

- [Qwen2-VL-2B model card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Transformers Qwen2-VL documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen2_vl)
- [DINOv2-small model card](https://huggingface.co/facebook/dinov2-small)
