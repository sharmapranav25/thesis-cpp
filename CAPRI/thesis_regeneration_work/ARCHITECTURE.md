# Architecture

## Research objective

Estimate reproducible CAPTCHA appearance properties from unlabeled real images, then use those estimates to control synthetic rendering. Progress from simple measurements to more complex methods while separating measured properties, hypotheses and remaining hard-coded priors. Neither visual similarity nor passing structural tests establishes downstream OCR accuracy.

## Execution paths

```text
Historical imported repository (read-only)
  -> byte-identical vendor snapshot + SHA-256 provenance
  -> legacy probe with Windows font-path adaptation
  -> baseline replay -> exact pixel/label/dimension comparison

Unlabeled image pools
  -> decode -> pixel identity -> deduplicate -> external overlap check
  -> deterministic development / validation split
  -> method 1 -> four-dataset gate + visual review
  -> method 2 -> four-dataset gate + visual review
  -> method 3 -> four-dataset gate + visual review
  -> method 4 -> four-dataset gate + visual review
  -> method 5 -> four-dataset gate + visual review
  -> method 6 -> four-dataset gate + visual review

Supported observations -> explicit renderer adapter -> synthetic samples
  -> stage images + parameters + labels + rejection logs + validation
```

## Components and boundaries

| Component | Location | Responsibility |
|---|---|---|
| Frozen imported renderer | `vendor/imported_repo/system1_synthetic_generator/previously_produced_generators/gen_gateonly/` | Original background, noise, typography, distortion and quality code |
| Frozen saved CryptBB renderers | `vendor/imported_repo/system1_synthetic_generator/system1_production/final_generators/` | Replay three historically VLM-authored Python generators; no inference-time VLM |
| Original ablation driver | `vendor/imported_repo/experiments/gen_ablation/` | C1/C2 recipe/text sampling and hard-coded family contracts |
| Import receipt | `vendor/PROVENANCE.json` | Original absolute paths, hashes and modification policy |
| Historical probe | `scripts/legacy_probe.py` | Unchanged copy of the previous instrumented test harness |
| Reproduction runner | `scripts/reproduce_baseline.py` | Redirect imports to vendor; compare with saved historical results |
| Dataset locations | `configs/datasets.json` | Input/exclusion paths only; no alphabet or length rules |
| Existing experimental implementation | `sourcefree_synth/{pipeline,renderer,features,vlm}.py` | Prior independent approach, retained for comparison; not the imported baseline |

Existing `features.py` mixes pixel, edge and texture descriptors and optionally embeddings. It is NOT a valid isolated Method 1 experiment. New staged methods must isolate their own measurements and record any reuse.

## Parameters and provenance

Every derived parameter must be tagged as MEASURED, INFERRED, FIXED EXPERIMENT SETTING, or UNRESOLVED. An empirical mode is not automatically the background; a connected component is not automatically a character. Missing characters in OCR predictions do not prove alphabet exclusions.

Original family alphabet/length constants are allowed for baseline reproduction only. Discovery methods must not import them as facts. Generic rendering operations and explicit algorithm thresholds remain programmed assumptions; recovering a parameter distribution does not recover the unknown source code.

## Compute and information constraints

- Target budget: one RTX A4000, 16 GB VRAM; actual model peak memory must be measured when models are introduced.
- Start with CPU NumPy/Pillow. Do not download models or train recognizers in the baseline or initial measurement stages.
- Fit on unlabeled real pixels. Do not parse labels from paths, filenames or CSVs. Audit manifests retain source paths for traceability but they are not model inputs.
- CryptBB and DarkBay external evaluation images are used only for exact overlap checks; their labels remain unread. City/Courier have no declared external evaluation set.
- Internal validation images are held out from fitting. Repeated development against their diagnostics makes them development validation, not a pristine final test set.
- Assistant image viewing is qualitative VLM-assisted evaluation. It is not inference inside the generator. Changes motivated by this review must be recorded; the overall research process is therefore not claimed to be free of VLM assistance.

## Failure behavior

Preserve imported source bytes. Document original retry/schema substitution and stale-label risks. Any fixed-schema, failure-closed generation behavior belongs in a separately identified adapter, with its own validation. Never silently relax gates, discard bad samples, invent labels or proceed to a later method after an unresolved method gate.

## Storage and reproduction

User-facing project: `E:\pranav\Final sem\sourcefree_synth`. Temporary implementation staging may occur under the writable thesis workspace; publish verified files back to the user-facing folder. Keep each run in a new `runs/` subdirectory. Preserve earlier runs and documentation revisions. See [METHODS.md](METHODS.md), [VALIDATION.md](docs/VALIDATION.md), and [EXPERIMENTS.md](docs/EXPERIMENTS.md).
