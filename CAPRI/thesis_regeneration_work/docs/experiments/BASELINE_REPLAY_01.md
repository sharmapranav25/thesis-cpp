# BASELINE-REPLAY-01 — imported generator reproduction

Date: 2026-09-27. Scope: establish that the imported implementation produces the same results after copying into this project.

## Command and environment

Executed from the local staging copy `C:/Users/pranav/Documents/ChatGPT/thesis/thesis_regeneration_work`:

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B scripts/reproduce_baseline.py --out runs/baseline_replay_01
```

Published without changing result bytes to `E:/pranav/Final sem/sourcefree_synth/runs/baseline_replay_01`.
Python 3.9.13, Pillow 11.3.0, NumPy 2.0.2; CPU rendering. No VLM calls or training.

## Numerical outcome: PASS for exact reproduction

All **128/128** images matched historical RGB pixel hashes, requested labels and dimensions: three saved generators x two font environments x 16 samples = 96; portable C1/C2 x 16 = 32. All 25 original/vendor snapshot files retained their hashes. Import-location checks confirmed runtime modules came from the vendor copy.

- [Full reproduction receipt](../../runs/baseline_replay_01/equivalence.json)
- [Generated report](../../runs/baseline_replay_01/REPORT.md)
- [Source provenance](../../vendor/PROVENANCE.json)

Both portable C1/C2 cases found 12 local generic fonts and returned no label mismatch or degeneracy flag in this small replay. Recipe substitutions remained present (C1: 11/16; C2: 8/16), exactly as before. This does not show that larger runs are free of the previously demonstrated stale-image bug.

## Assistant visual review

Reviewer: assistant native image-viewing capability; qualitative, sampled comparison, not OCR ground truth. Reviewed on 2026-09-27:

| Artifact | Observation | Interpretation |
|---|---|---|
| [C1 preview](../../runs/baseline_replay_01/universal/c1_random_whitelist_paths_mapped/preview.png) | Wide variation in canvas size, grids, colors, typography and distortion | Preserved generic style sampler; not a fitted CryptBB generator |
| [C2 preview](../../runs/baseline_replay_01/universal/c2_curriculum_whitelist_paths_mapped/preview.png) | Similar variation, including dark backgrounds, outlines and crowded/overlapping-looking text | Numerical gate does not certify perceptual readability |
| [Saved cluster 0, mapped font](../../runs/baseline_replay_01/saved/cluster_0_same_font_local_path/preview.png) | Straight, colored bold text on sparse speckle; yellow/cyan can be low contrast | Reproduced but not faithful to observed real variation |
| [Saved cluster 1, mapped font](../../runs/baseline_replay_01/saved/cluster_1_same_font_local_path/preview.png) | Saturated rectangular background patches compete with text; some right-edge glyph clipping | Retain as historical result; not approved for training |
| [Saved cluster 2, mapped font](../../runs/baseline_replay_01/saved/cluster_2_same_font_local_path/preview.png) | Strong speckle and frequent right-edge clipping of large glyphs | Known rendering limitation preserved by exact replay |
| [Real CryptBB comparison](../../runs/pixel_statistics_01/cryptbb/fit_review.png) | Light backgrounds with several clutter types and varied colored glyph shapes | Visual mismatch remains despite baseline reproducibility |

The three default-font saved cases were checked numerically against all historical outputs but were not individually reviewed visually in this replay. Process panels for the first two C1/C2 outputs are retained beside each preview.

## Verdict and next step

Baseline equivalence **PASS**. Family fidelity / training readiness **NOT ESTABLISHED**. Proceed to Method 1's scoped measurement checks; do not present baseline reproduction as successful image-based rule discovery.
