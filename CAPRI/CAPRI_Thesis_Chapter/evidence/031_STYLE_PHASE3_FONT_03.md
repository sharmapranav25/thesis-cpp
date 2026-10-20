# Phase 3 font robustness experiment 03

Completed 2026-09-28. Synthetic-only warm-start training from run 02 continued to step 3200; checkpoint 2800 selected by synthetic selection loss. No real labels, runtime VLM, target-family font files or generator code used. Historical feature classifiers remained frozen.

## Method and paper
DeepFont-inspired synthetic clutter augmentation and multi-view training alignment; original small CNN, not a reproduction. Training cycles whole/left/middle/right views. All synthetic partitions alternate standard and 3x clutter. See [thesis paper register](THESIS_PAPERS.md) for primary sources, actual borrowed ideas and deviations.

## Matched test
Both fixed models evaluated on the same 768 synthetic images, seeds 40000–40063 across 12 fonts. This harder test differs from earlier runs' headline tests.

| Measure | Run 02 | Run 03 |
|---|---:|---:|
| Font top-1 | 45.96% | 54.56% |
| Font top-3 | 74.09% | 84.24% |
| Hollow/filled | 85.55% | 89.97% |

Evidence: `runs/style_phase3_font_03/matched_synthetic_test.json`. These are synthetic accuracies, not verified real-font accuracy. The test did not select the checkpoint.

## Real development results
CV measurements plus frozen feature inference were retained; new frozen font inference and typography search used existing 96 fit / 32 development images per family. No Phase 2 reserved window accessed. Generated 160 images with zero render failures. Selected fonts: CryptBB/Courier/outlined Courier Prime, City Noto Sans, DarkBay Oswald.

Assistant inspected the first four baseline/adjusted renders per family in all_families.png. Typography changes are visible, but outlined still selects bold filled letters and fails the intended hollow appearance. This inspection is modest qualitative review, not independent real accuracy. Development losses are model-dependent search diagnostics, not proof of generalization. The current script still generates five characters, an unresolved limitation.

## Outcome
Synthetic robustness improved on the matched test. Phase 3 has NOT passed. No new independent-family test of run 03 has yet been completed; the previous pymlex preview test applies to run 02 and is now exposed development evidence.
