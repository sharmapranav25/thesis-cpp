# DarkBay ensemble shoot-rate results

Training dataset: **Darkbay_ds1_noFont-intel** — the complete 100,000-image synthetic corpus, without font intelligence. Dataset identity is recorded in `capri_darkbay_workflow/dataset_identity.json`.

Recorded 2026-10-05. Three PARSeq seed predictions (48, 123, 456), whole-string majority vote with confidence tie-break. Accuracy is correct answers divided by attempted CAPTCHAs. Evaluation pool: 900 real DarkBay images, excluding the 100 controller references.

| Threshold | Correct / attempts | Accuracy |
|---|---:|---:|
| 0.99 | 211 / 269 | 78.4% |
| 0.95 | 267 / 358 | 74.6% |
| 0.90 | 293 / 398 | 73.6% |
| 0.85 | 312 / 439 | 71.1% |
| 0.80 | 332 / 478 | 69.5% |
| 0.70 | 340 / 499 | 68.1% |
| 0.60 | 421 / 645 | 65.3% |
| 0.50 | 451 / 749 | 60.2% |
| 0.40 | 468 / 807 | 58.0% |
| 0.30 | 491 / 871 | 56.4% |
| 0.20 | 495 / 897 | 55.2% |
| 0.10 | 496 / 900 | 55.1% |
| No filtering | 496 / 900 | 55.1% |

Source: `runs/parseq_darkbay_100k_20261005/shoot_rate_analysis/summary.json`. This is a descriptive sweep on historical test data, not independent threshold calibration.

## Font intelligence integration note

Confirmed by the user: these results are the baseline **without font intelligence**. Font intelligence will be integrated next. Preserve these measured results separately so the effect of that integration can be compared against this baseline. No integration or new training was performed as part of recording this note.
