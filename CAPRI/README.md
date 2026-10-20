# CAPRI research workspace

Completed DarkBay experiment record: [DarkBay through Model 2](docs/DARKBAY_THROUGH_MODEL_2.md), covering the 100-image hybrid/controller pipeline, both datasets, reader training, validation, real evaluation and synthetic-calibrated shoot-rate analysis through 6 October 2026. This later record supersedes the preliminary outcome statements below; historical protocols remain preserved.

Start here for the maintained pixel-only controller workflow. The active code is now in `src/capri_controller`, with four selected checkpoints in `models/active` and one entry point, `capri.py`. Existing experiment folders are historical evidence and remain at their recorded paths.

## Where things belong

| Location | Purpose |
|---|---|
| [src/capri_controller](src/capri_controller/) | Maintained hybrid, custom, DINO and mixed controller integration |
| [models/registry.json](models/registry.json) | Selected checkpoint paths, roles, original locations and SHA256 hashes |
| [configs/local.json](configs/local.json) | Machine-specific Python, renderer, feature-model source and run paths |
| [configs/targets](configs/targets/) | Pixel input manifests; no target text labels |
| `runs/` | Destination for future runs; never put new experiment outputs inside `src/` |
| [capri_darkbay_workflow/report.html](capri_darkbay_workflow/report.html) | Completed DarkBay run and comparisons, preserved in place |
| [docs/MAINTENANCE_LOG.md](docs/MAINTENANCE_LOG.md) | What the cleanup changed, validation and remaining limitations |
| [docs/WORKSPACE_CATALOG.md](docs/WORKSPACE_CATALOG.md) | Generated index of all workspace folders and their sizes |
| [docs/CODE_AND_MODELS.md](docs/CODE_AND_MODELS.md) | Architecture, model selection and operating instructions |
| [archive/root_logs_20261005](archive/root_logs_20261005/) | Archived root-level logs; exact move manifest in `maintenance/` |

## Commands

Run from this directory using the existing environment. The environment folder happens to be named `experiments_xgb`; the maintained controllers do not use XGBoost.

```powershell
$capriPython = 'E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe'
& $capriPython capri.py verify
& $capriPython capri.py status
& $capriPython capri.py models
& $capriPython capri.py report --run capri_darkbay_workflow
& $capriPython capri.py audit --run capri_darkbay_workflow
```

For a new run, use a new output directory. Controller fitting is expensive; `status`, `models`, `verify`, `inventory` and `report` do not train or fit controllers.

```powershell
& $capriPython capri.py controllers --out runs/darkbay_next
& $capriPython capri.py generate --run runs/darkbay_next --pilot --workers 8
& $capriPython capri.py generate --run runs/darkbay_next --workers 8
& $capriPython capri.py report --run runs/darkbay_next
& $capriPython capri.py inventory
```

The default target manifest contains the same 100 DarkBay pixel references. `controllers --targets path/to/manifest.json` selects another manifest with a `fit` list of `source` and `pixel_sha256` objects. Standard full runs require 100 unique inputs and a model-derived length majority of at least 60/100; they fail explicitly without a majority. The bounded preview exception is described below.

For a visual preview, `controllers --preview --out runs/<new-name> --targets <manifest>` uses all100 references but refines only the first6 inputs, preserving custom/DINO/mixed outputs. In this preview mode only, a missing majority uses seeded, equally weighted cycles across the distinct model-predicted lengths, as specified in `CAPRI_WORKFLOW.md`. This never uses known scheme lengths or target strings. Preview runs cannot export a corpus. `capri.py preview-smaller-hybrid` orchestrates the user-approved first6 examples from the existing five smaller-hybrid sets and reuses the verified exact-input DarkBay outputs; it refuses to overwrite its dated output. Standard full controller/corpus runs still require a majority.

## Current evidence

DarkBay: 100,000 distinct labeled images, 300 recipe entries, 179 parameter combinations, balanced across the three controllers. The default length of 6 came from 72/100 raw hybrid votes. Only 36.3% pass the strict readability check; the corpus is experimental, with no established downstream solver benefit. See [CAPRI_WORKFLOW.md](CAPRI_WORKFLOW.md).

The four active weight copies have verified hashes. Original checkpoints and experiment scripts are retained for reproduction. Historical XGBoost, SAM, Meta-Sim2 and DINOv3 experiments are not selected by the maintained workflow. Frozen `CAPRI` packages are separate research releases, not this entry point.

## PARSeq reader experiment

DS2 curriculum job: `capri.py darkbay-ds2`. Its dataset, protocol, launch receipt, stage logs and `status.json` are under `runs/Darkbay_ds2_noFont-intel`. It generates100k fresh renders, ranks with the DS1 epoch20 ensemble, holds out5000 synthetic images, and trains seeds48/123/456 for20epochs with Model2's adapted curriculum and augmentation. It saves each epoch and selects by synthetic validation, then evaluates on historical real DarkBay. Check status before starting; the running lock prevents duplicate orchestration. See the maintenance log for protocol differences from DS1.

The training dataset is named **Darkbay_ds1_noFont-intel** (100,000 synthetic images, without font intelligence). Its identity and existing storage paths are recorded in `capri_darkbay_workflow/dataset_identity.json`.

The 100k DarkBay corpus has a fixed20-epoch PARSeq run with seeds48/123/456 and final-checkpoint selection. Data/configs/status are in `runs/parseq_darkbay_100k_20261005`. Training was launched on 2026-10-05 with random initialization after the user requested a start with the pretrained checkpoint unavailable. This initialization differs from the thesis's pretrained recipe. Read the run's `status.json` for live state; do not launch a duplicate job. The guarded pretrained launcher, for a separately planned run, is:

```powershell
& $capriPython capri.py train-parseq --checkpoint 'path/to/scene-text-base.pth'
```

`--from-scratch` is a separate explicit alternative and changes the initialization protocol. See the maintenance log and saved `protocol.json` before launching.

All three reader runs have now completed20epochs. Real DarkBay evaluation on900 images excluding the100 controller inputs gives54.67%,48.78%,52.67% exact case-sensitive accuracy (mean52.04%). This is historical-data evaluation, not a newly untouched benchmark. [Evaluation report](runs/parseq_darkbay_100k_20261005/evaluation_darkbay/report.html). Predictions were frozen before labels were scored; no checkpoint was selected based on these results.
