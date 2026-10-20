# Current status

Updated: 2026-09-27.

| Stage | Status | Evidence |
|---|---|---|
| Documentation and frozen import | Published to the synth folder | `../ARCHITECTURE.md`, `../METHODS.md`, `../vendor/PROVENANCE.json` |
| Historical baseline reproduction | PASS: 128/128 exact matches; visual weaknesses documented | [Baseline record](experiments/BASELINE_REPLAY_01.md) |
| Method 1: pixel statistics | PASS on four datasets for measurement scope; full regeneration unresolved | [PIXELS-01](experiments/PIXELS_01.md) |
| External public-data probe | PASS on 800 additional images; spatial negative control confirms limitations | [External pixel study](experiments/EXTERNAL_PIXELS_OUTLINED_01.md) |
| Fresh imported C1/C2 test | 160 images generated; internal checks pass after font mapping; dataset regeneration not achieved | [Five-dataset generation test](experiments/IMPORTED_ALL_DATASETS_01.md) |
| Methods 2–6 | Not started; gated | `../METHODS.md` |

No downstream training, model download or local VLM execution is part of these initial stages.

The user requested the public-data probe before Method 2 implementation. Initial Method 2 literature exploration occurred, but no Method 2 code or gate is complete. Next eligible primary method: edges and texture analysis, validated on all four primary datasets before segmentation.
