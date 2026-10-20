# Experiment ledger

Every run has a new output directory; preserve failures. Record command, date, code/config/input identities, seeds, environment, runtime, coverage, metrics, qualitative review, verdict and limitations. Links below are to executable evidence, not reconstructed claims.

| ID | Purpose | Status | Evidence |
|---|---|---|---|
| HIST-SAVED | Original saved-generator test | Historical, previously executed | `E:/pranav/Final sem/imported_regeneration_test/cryptbb_saved_01` |
| HIST-C1C2 | Original portable C1/C2 test | Historical, previously executed | `E:/pranav/Final sem/imported_regeneration_test/cryptbb_universal_portable_01` |
| BASELINE-REPLAY-01 | Replay historical results from frozen local import | PASS: 128 exact matches | [Experiment record](experiments/BASELINE_REPLAY_01.md) |
| PIXELS-01 | Native pixel measurements on all four input pools | PASS within measurement scope | [Experiment record](experiments/PIXELS_01.md) |
| EXTERNAL-PIXELS-OUTLINED-01 | Unchanged Method 1 on 800 public images + spatial negative control | PASS for measurements; insufficient for regeneration | [External experiment](experiments/EXTERNAL_PIXELS_OUTLINED_01.md) |
| IMPORTED-ALL-DATASETS-01 | Fresh original-repo C1/C2 batches: 3 supported text contracts + 2 generic fallback contexts | 160 outputs; internal checks pass; family visual regeneration not achieved | [Generation test](experiments/IMPORTED_ALL_DATASETS_01.md) |

The old independent renderer run `E:/pranav/Final sem/sourcefree_synth/runs/cryptbb_no_vlm_20260927_01_preview` is separate from the imported baseline. It has known visual distortion weaknesses and is not evidence that the imported generator fits real data.
