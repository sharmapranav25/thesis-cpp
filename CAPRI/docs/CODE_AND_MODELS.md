# Code and model management

The maintained controller implementation lives in `src/capri_controller`. This separates reusable code from old run directories without relocating evidence or breaking historical scripts. Use the root `capri.py` entry point for new work.

## Code structure

| Module | Responsibility |
|---|---|
| `cli.py` | Explicit inspection, fit, generation, audit and reporting commands |
| `runtime.py` | Central configuration, model resolution, JSON and streaming hashes |
| `pipeline.py` | Per-image hybrid inference, cleanup, model-voted length and three-controller refinement |
| `controllers.py` | Custom, DINO and calibrated mixed scoring |
| `features/` | Shared pixel descriptors, inferred regions, custom and DINO comparisons |
| `networks/` | Frozen custom encoder and mask architectures; checkpoint-compatible output dimensions |
| `rendering/` | Generic renderer controls and complete glyph controls |
| `artifacts.py` | Input RGB-hash verification and saved render labels |
| `corpus.py` | Balanced, resumable ZIP-shard generation and full image audit |
| `reporting.py` | Lightweight HTML overview from saved artifacts |
| `inventory.py` | Metadata catalog for folders and historical checkpoints |
| `vendor/dino/` | Existing official DINO source with its original Apache license |

Package-local imports are explicit and relative. `runtime.py` is the only path bridge to the existing external renderer and feature-model source repositories. Those dependencies and the font bank remain external; this cleanup does not make the project standalone or portable to another computer automatically. Edit `configs/local.json` on another host, verify all dependencies and review absolute image/font paths in saved recipes.

## Selected models

| Registry name | Purpose | Selection |
|---|---|---|
| `hybrid` | Initial feature labels and length votes | Original simple hybrid, epoch 15 EMA |
| `custom_encoder` | Appearance features for the custom controller | Existing frozen custom CNN encoder |
| `text_mask` | Infer text regions from pixels | Existing synthetic-trained mask network |
| `dino` | Attention and appearance features | Official original DINO ViT-S/8 |

The phrase “DINO controller v2” describes our controller iteration. It does **not** mean DINOv2 pretrained weights. The mixed controller combines custom and DINO scores; it has no additional trained checkpoint. No VLM or XGBoost model is loaded by this workflow. The hybrid architecture has a CTC branch, but the controller does not decode target strings.

`models/registry.json` is authoritative for active weight paths and expected hashes. Copies in `models/active` are byte-identical to the recorded originals. Retaining the originals is deliberate: old experiments still depend on them. Do not infer model quality from filenames such as `best.pt`, timestamps or file size.

`models/historical_inventory.json` lists workspace checkpoint locations and sizes. Except for the active copies, its hashes are not computed and its contents have not been audited for duplication or fitness. External training checkpoints are not recursively inventoried; the selected external source checkpoint is recorded in the active registry. The inventory is not authorization to delete training states, specialist teachers or frozen releases.

## Operation and provenance

Use `capri.py verify` after changing machines or selected weights. It checks active file hashes and the two external source directories; the regression tests provide stronger evidence of actual inference and rendering behavior. It does not automatically download models or install packages.

Use a fresh `runs/<name>` directory for controller fitting. The CLI rejects an existing output directory. Corpus generation resumes completed shards only when their stored checkpoint-independent recipe manifest hash and ZIP hash match the current run. Each image keeps its chosen synthetic text, render seed, controller and recipe ID, and basic/strict validity flags.

The model decides the default target length through raw vote aggregation. The 60% threshold and renderer cleanup are engineering rules, not human target labels. Standard full runs stop if there is no majority. Target strings, positions and texture labels are never supplied.

The opt-in `controllers --preview` path refines the first6 inputs against all100 references. Only this bounded path implements the documented no-majority fallback: seeded shuffled cycles of distinct observed model lengths, with one assigned length shared by all controller candidates/arms for each input. It changes no comparator, objective, weights or candidate search. `generate` refuses preview runs. `smaller_hybrid_preview.py` provides the dated `preview-smaller-hybrid` orchestration, exact-input/checkpoint-verified DarkBay reuse, per-scheme audits and a comparison report. Raw inference evidence is now saved before the standard no-majority failure as well.

The 100,000-image allocation is 333 or 334 images per entry. Rejections are retried within the same entry. The stricter readability flag is diagnostic, not an export requirement. Filtering to strict-only images would change the mixture and must be reported.

The new `report` command writes `overview.html`; it does not rerender or replace the detailed historical `report.html`. Re-run `inventory` to refresh the catalog after adding experiments or moving artifacts. Update [MAINTENANCE_LOG.md](MAINTENANCE_LOG.md) whenever maintained code, selected models, locations or reproduction commands change.

## Historical reproduction

Original MVP folders and the original three DarkBay scripts remain in place. They reproduce the old run and are not the maintenance target for new features. Their hashes are preserved in `maintenance/extraction_20261005.json`. Changes in the new package include direct imports, registry-based paths, a new-run overwrite guard, removal of the old DarkBay-specific vote assertion, a generic gallery heading, and exclusion of sampled text from the parameter-diversity count. Rendering and comparator calculations were preserved and regression checked.

Root-level logs were moved to `archive/root_logs_20261005`; `maintenance/root_log_moves_20261005.json` records every original path, destination, byte size and SHA256. Old scripts can create new logs at their old destinations on rerun. No historical model, corpus, source directory, release or ZIP package was deleted.

## Model 3 adaptive reader experiment

`capri.py darkbay-ds3` runs the separate curriculum-only experiment on the existing Model2 LMDB and split. Numeric protocol: [MODEL_3_ADAPTIVE_PROTOCOL.md](MODEL_3_ADAPTIVE_PROTOCOL.md). Implementation is under `src/capri_controller/training/ds3*.py`; outputs and state are under `runs/Darkbay_model3_adaptive_20261006`. Inspect status before invoking; the run lock prevents duplicate orchestration. Readiness receipts are required before production. All baseline artifacts and selected controller models remain unchanged.
