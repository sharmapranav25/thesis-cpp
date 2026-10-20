# Phase 3 contrast robustness run 04

Run 04 completed in 80.14 seconds. Synthetic selection chose step3600, before fresh test reporting. Training remained synthetic-only; the historical feature classifiers were unchanged. All160 real-development renders completed without failures. No reserved Phase2 evaluation images used.

## Matched frozen comparison

Fresh seeds80000–80063, 12fonts, 768images at each gain; same images evaluated by both fixed models using four-view inference. This differs from full-image-only training-script test metrics. No checkpoint selected using this comparison.

| Contrast gain | Font top1:03 →04 | Hollow/filled:03 →04 |
|---|---:|---:|
| 1.0 | 51.82% → 50.00% | 95.31% → 91.93% |
| 0.65 | 50.26% → 52.86% | 93.36% → 89.84% |
| 0.35 | 48.05% → 49.74% | 80.73% → 82.81% |
| 0.15 | 20.96% → 26.30% | 39.71% → 68.75% |

## Decision and limits

Do not promote run04 as a universal replacement. Low-contrast robustness improves but normal-contrast scores regress. Assistant inspected four outlined real-development examples and their baseline/adjusted renders: the chosen typography remains filled, not hollow. Mean real fit hollow probability fell from .236 in run03 to .064 in run04. These are uncalibrated scores, not real accuracy. Phase3 remains unpassed.

The contrast augmentation is a local extension of the documented DeepFont-inspired approach, not a paper reproduction. See [paper register](THESIS_PAPERS.md). No new paper used. Historical files remain unchanged. Next investigation should test outline geometry/scale and segmentation rather than assuming additional contrast training fixes the mismatch. New independent-family evaluation remains outstanding for these revisions; do not call the loop complete.