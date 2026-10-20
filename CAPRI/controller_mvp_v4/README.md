# Controller MVP v4: diversity transfer experiment

## Completed result

Implementation and matched training are complete. **The primary transfer test did not succeed.** Exact accuracy, averaged over the two fixed seeds:

| Evaluation | Images | Hybrid | Single | Diverse |
|---|---:|---:|---:|---:|
| SimpleCaptcha, primary | 999 | 0.00% | 0.05% | 0.00% |
| DarkBay | 39 | 0.00% | 0.00% | 1.28% |
| Outlined | 128 | 0.00% | 0.00% | 0.00% |
| CryptBB, supplementary | 350 | 0.00% | 0.00% | 0.29% |

Diverse versus single character error rates: SimpleCaptcha 79.57% vs 79.58%; DarkBay 46.79% vs 50.64%; Outlined 97.97% vs 95.78%; supplementary CryptBB 75.37% vs 77.14%. These are poor absolute recognition results. They do not establish that diversity is ineffective in general: the selector has weak typography coverage and the readers also remain imperfect on synthetic validation.

City's 79 reserved images and the original 24 CryptBB diagnostic images have no verified answers. The CryptBB supplement was added after finding this; it uses the same frozen readers and all eligible native-size labeled CSV images, excluding known controller hashes and duplicates. It is exploratory, not the primary test. One of the 1,000 reserved Kaggle images has a filename outside the publisher's documented five-character format; it was excluded before scoring. Original filename/CSV annotations are accepted as supplied, matched by decoded-pixel hash, without human or OCR re-annotation. Model weights and controller configurations were not changed after answer access.

This iteration retains 16 target-like renderer configurations per family rather than one winner. They are parameterized schemes in the existing renderer, not copies of external CAPTCHA libraries. No VLMs or target answers are used in controller selection, synthetic generation, reader training or checkpoint selection.

## Controller

`collections4.py` starts from the v3 candidate histories, crosses typography settings with background/interference settings, and evaluates candidates against the existing unlabeled fit and selection pixels. Candidates must be complete, at least 90% usable, have synthetic predicted-mask IoU at least .30 and median predicted/true ink-area ratio between .4 and 2.5. Synthetic ground-truth masks are used only for this reliability check, never as target labels. Appearance must be within twice the best eligible objective. A greedy farthest-point selector, penalized by appearance cost, keeps 16 configurations.

If the first pool has too few passing candidates, up to three 48-proposal rounds explore neighborhoods of trustworthy configurations. The thresholds are unchanged. Initial attempts and source versions in `attempts/` document a shortlist-count repair and the added proposal expansion. Completed archives were retained by hash; no held-out answers or reader performance informed these changes.

## Matched downstream experiment

Three pooled-family arms use City, CryptBB, DarkBay, Outlined and SimpleCaptcha:

1. **hybrid:** neutral controls on the hybrid's original renderer recipes.
2. **single:** the best eligible controller configuration per family from the v4 selection pool.
3. **diverse:** balanced sampling of all 16 selected configurations per family.

Each arm gets 12,800 training images: 2,560 identical generic clean anchors and 10,240 adapted images. Each family contributes 2,048 adapted images; the diverse arm allocates 128 to every configuration. All arms share identical synthetic transcripts and matched renderer seeds. If any arm cannot render the assigned transcript completely and readably, all three retry with the same next seed. No arm silently loses examples.

For recognition training, all arms use the same generic 62-character alphabet, unique strings of length 2–8 and line layout. Controller length overrides are disabled. Fonts marked caps-only or lacking the complete alphabet are replaced by a compatible training font in the same category. These common changes prevent missing-character coverage or target answer statistics from determining the comparison. Train and validation strings are disjoint.

All arms render at native height 64 with aspect-preserving width bounded to 128–512 before the common 192×48 reader resize. If ten matched seed attempts fail completeness/readability, every arm receives one additional pixel of stroke; after twenty, two pixels. Thirty failures stop generation. The first corpus attempt exposed a thin dotted hybrid font losing glyphs under resampling; its partial arrays, error log and source are preserved in `attempts/`. These repairs alter the raw starting recipes, so the hybrid arm is a shared-curriculum baseline, not an untouched hybrid replay. Per-example retry and stroke records disclose their frequency.

A common 1,600-image synthetic validation set mixes all three style arms and clean anchors. The reader is a 1.31-million-parameter image-only CNN/BiLSTM/CTC model, initialized from scratch. Two matched random seeds, identical starting weights within each seed, equal batches and 20 epochs are used. Selection is lowest common synthetic validation CER, then highest exact accuracy. No language model, pretrained OCR or real images enter reader training.

Post-generation audit: all families retain 16 distinct saved configurations, but disabling length controls and standardizing font support leaves 13 effective configurations for SimpleCaptcha and 16 for each other family (77 effective configurations total). Sampling remains balanced over the original 16 slots; SimpleCaptcha's three collapsed slots therefore upweight their matching effective profiles. The shared stroke repair was needed for 163 of its 2,048 adapted training examples and none from the other families. These limitations are recorded in `corpus_review/audit.json`.

## Evaluation boundary

`prepare4.py` reserves pixels by hash without opening annotations. The primary test is 1,000 Kaggle Simple Captcha images excluded from prior controller splits. City has 79 fresh images, DarkBay 39, Outlined 128. CryptBB has only 24 reused diagnostic images. These are fresh relative to known controller/ablation/localization splits, not a claim of no historical project exposure.

`train4.py` freezes all six selected models. `infer4.py` then records all target predictions without answers. Only afterward may evaluation annotations be read and matched to pixel hashes. `evaluate4.py` reports case-sensitive whole-CAPTCHA accuracy and character error rate, plus paired image-bootstrap intervals for diversity versus each baseline, averaged over the two fixed training seeds. The intervals do not establish robustness across arbitrary training seeds. Unverified annotations are excluded and disclosed, never guessed.

## Files

- `collections/`: candidate histories, 16-profile archives, single-profile choices and preview images.
- `corpus/`: matched training arrays, common validation data, labels and checksums.
- `solvers/`: initial weights, histories, selected checkpoints and frozen-model manifest.
- `evaluation_pixels/`: reserved image references with freshness status, without answers.
- `evaluation/`: frozen predictions, evaluation-only answer provenance and results.
- `report.html`: final experiment report, produced after evaluation.

`run_pipeline4.py` sequences corpus generation, training and answer-free inference after collection building completes. Its final state is intentionally `ready_for_answer_evaluation`; evaluation remains a separate step so annotations cannot enter earlier stages.

`sample4.py --family City --count 1024 --output <new-directory>` exports fresh 192×48 synthetic PNGs, assigned transcripts, profile slots and provenance from any retained family collection. It cycles through all saved profile slots using the shared recognition curriculum. Existing output directories are rejected to prevent accidental overwrites. Run with the project's Python environment; the generator and font-bank paths still point to the existing local project. `export_check/` contains a checked 16-image SimpleCaptcha export, independent of reader training and evaluation.

## Limits

The existing renderer still lacks some target structures, including explicit per-glyph rectangular frames. Diversity cannot guarantee coverage of an unsupported visual feature. These profiles inherit approximate v3 comparison metrics, and 12 fit / 24 selection renders provide noisy style estimates. Equal finite training budgets may still undertrain all readers; synthetic learning curves must accompany target results. A collection with better visual scores is not counted as a transfer success unless held-out reader accuracy improves.

In particular, all 16 SimpleCaptcha archive slots chose Great Vibes. Their variation covers geometry, effects and backgrounds, but does not establish font diversity or successful resemblance to the boxed-letter target. `collections/typography_audit.json` documents this before target answer evaluation. The other archives retain several font choices or a generic font pool. This is a test of the current selection mechanism, not a claim that its appearance descriptors reliably recognize the correct typography.
