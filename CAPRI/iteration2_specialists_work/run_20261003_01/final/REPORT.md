# Iteration-2 specialist results

Selected checkpoint: `C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_specialists_work\run_20261003_01\final\hybrid.pt`.
SHA256: `198f89c05877a2e5053215ed53cc3e9be33d4a2a9dabc94f6e7842af415f6ccf`.

Completed: five presence specialists × four 20k-draw epochs, eight full-100k presence merge epochs, three intensity specialists × four 20k-draw epochs, and four full-100k intensity merge epochs. Selected intensity merge epoch 4. Training settings stayed fixed throughout.

Presence grouping: 5 groups, 61 assigned tasks. Every presence specialist starts from the original frozen baseline; intensity specialists start from presence merge epoch 8. Student merges use ground truth plus assigned-output distillation.

On selection, 94/107 scored outputs improved and 8/107 regressed. Seven binary text features have no intensity target and are explicitly excluded from scored ordinal comparisons.

| Selection metric | Baseline | Final |
|---|---:|---:|
| Effect presence F1 | 0.4883 | 0.5387 |
| Text presence F1 | 0.5024 | 0.5457 |
| Effect positive-level within-one | 0.8016 | 0.8167 |
| CTC exact | 0.4160 | 0.4465 |
| Character error rate | 0.2993 | 0.2670 |
| Length accuracy | 0.8450 | 0.8825 |
| Font category accuracy | 0.6925 | 0.7440 |
| Font ordinal within-one | 0.7785 | 0.7957 |

These selection presence values use the same 0.5 threshold. Calibration thresholds are fitted separately to maximize F1 subject to FPR <= .10. Checkpoint choice uses selection only with the preservation gates recorded before launch. The comparison is against the saved 15-epoch baseline; the specialist pipeline received additional training. No equal-compute continued-simple control was run, so this does not isolate specialization from the benefit of more updates.

## All font attributes on selection

| Font attribute / metric | Simple | Specialist | Change (points) |
|---|---:|---:|---:|
| Category accuracy | 69.25% | 74.40% | +5.15 |
| Stroke accuracy | 93.67% | 95.84% | +2.17 |
| Weight within ±1 | 80.35% | 82.60% | +2.25 |
| Width within ±1 | 70.65% | 73.00% | +2.35 |
| X-height within ±1 | 68.80% | 71.00% | +2.20 |
| Slant within ±1 | 91.60% | 91.70% | +0.10 |
| Italic F1 | 75.08% | 78.50% | +3.42 |

The ordinal scores permit an error of one bin; they are not exact-value accuracy. These are per-attribute scores, not the proportion of images with every font attribute correct. Slant is heavily imbalanced and its text/font targets duplicate each other; its high score should not be read as strong evidence of useful slant inference.

## Reused benchmark comparison

| Benchmark / metric | Simple | Specialist | Change (points) |
|---|---:|---:|---:|
| seen_test / Effect F1 at .5 | 49.30% | 54.03% | +4.73 |
| seen_test / Text F1 at .5 | 50.24% | 54.24% | +4.00 |
| seen_test / Effect intensity within ±1 | 80.47% | 81.96% | +1.49 |
| seen_test / CTC exact | 40.65% | 47.05% | +6.40 |
| seen_test / Length accuracy | 84.90% | 87.75% | +2.85 |
| generalisation / Effect F1 at .5 | 49.23% | 54.24% | +5.01 |
| generalisation / Text F1 at .5 | 51.02% | 54.70% | +3.68 |
| generalisation / Effect intensity within ±1 | 81.57% | 82.63% | +1.06 |
| generalisation / CTC exact | 46.96% | 50.44% | +3.48 |
| generalisation / Length accuracy | 85.32% | 88.40% | +3.08 |

These are reused benchmark sets already observed in previous experiments, not untouched validation evidence. Tests are not used for selection. Historical baseline results are read from its saved final report; the baseline was not scored again. Presence summary F1 in the table above uses .5 for both models. Final per-head F1/recall/FPR in the JSON uses calibration thresholds. Intensity scores use measured positive levels; font ordinal zero is valid.

### Font attributes: seen_test

| Font attribute / metric | Simple | Specialist | Change (points) |
|---|---:|---:|---:|
| Category accuracy | 68.70% | 73.50% | +4.80 |
| Stroke accuracy | 95.05% | 95.68% | +0.63 |
| Weight within ±1 | 80.15% | 81.75% | +1.60 |
| Width within ±1 | 73.70% | 75.10% | +1.40 |
| X-height within ±1 | 69.40% | 70.80% | +1.40 |
| Slant within ±1 | 91.50% | 91.60% | +0.10 |
| Italic F1 | 72.98% | 76.81% | +3.83 |

### Font attributes: generalisation

| Font attribute / metric | Simple | Specialist | Change (points) |
|---|---:|---:|---:|
| Category accuracy | 56.56% | 57.24% | +0.68 |
| Stroke accuracy | 89.20% | 90.11% | +0.91 |
| Weight within ±1 | 80.12% | 81.88% | +1.76 |
| Width within ±1 | 72.04% | 72.56% | +0.52 |
| X-height within ±1 | 58.60% | 61.20% | +2.60 |
| Slant within ±1 | 91.92% | 92.00% | +0.08 |
| Italic F1 | 65.70% | 67.17% | +1.47 |

Font category improves 4.80 points on seen_test but only 0.68 points on held-out-font generalisation (56.56% → 57.24%). Synthetic selection gains are clearer than the evidence for category transfer to new fonts. The real families do not have measured font ground truth, so no real-family font-accuracy claim is made.

| Family | Predictions | Previews | Strong readability passed | Mapping distance |
|---|---:|---:|---:|---:|
| City | 2000 | 100 | 77/100 | 0.0236 |
| CryptBB | 600 | 100 | 68/100 | 0.1064 |
| DarkBay | 999 | 100 | 61/100 | 0.0503 |

All 300 PNGs pass renderer usability, exact requested character counts, unique pixel hashes, saved image integrity, and unchanged sampled-prototype checks. The original six-attempt run failed only CryptBB row 25/prototype 415. Retrying the same prototype succeeded on zero-based attempt 8; the original failed report/labels and exact retry seed are preserved. No predictions, training settings, sampled prototypes or checkpoint weights were altered. Completion now rejects partial preview reports rather than treating file existence as success.

## Visual assessment

The three sheets were inspected against saved real reference examples. This is qualitative visual review, not a blinded similarity score. City has predominantly five digits and bands, but excessive font/palette/curves variation. DarkBay often has six characters and blocks/inverted regions, but font families, case mix, block geometry and colours remain far from the reference. CryptBB is the weakest: many short strings and thin/narrow glyphs with noisy grids/blobs, lacking the heavy multicolour mixed-font target appearance.

Whole-set predictions show the upstream problem: CryptBB length 2–4 in 487/600 images and predicted font weight 1–2 in 456/600. City is digits in all 2,000 predictions, but font categories remain spread across five classes. DarkBay blocks are requested in 75 previews and measured in 74, an additional renderer/mapping discrepancy; decoy characters are requested in five and measured in four. Correct mapping alone cannot fix the wrong predicted count/font/content or infer unspecified colours and geometry.

Renderer usability is not proof of target-family resemblance. Strict training readability passes only City 77/100, CryptBB 68/100, DarkBay 61/100, so these previews are not an approved solver-training corpus. No solver training was launched.

![CryptBB previews](../dataset/CryptBB_generated.jpg)

![DarkBay previews](../dataset/DarkBay_generated.jpg)

![City previews](../dataset/City_generated.jpg)

The static corpus lacks high measured thin-stroke and occlusion levels and usually varies one major text appearance property at a time. These coverage limitations remain explicit. All 53 presence pools contain diverse font categories and other features (`pool_diversity.json`).

Full per-head numbers, regressions, teacher checkpoint routes, label counts and reusable commands are retained in the run artifacts. `merged_teacher_comparison.csv` compares each phase student against the actual assigned teacher score, including baseline/start fallback. Raw/EMA and complete resume checkpoints are saved for every phase. The original baseline hash was verified unchanged.
