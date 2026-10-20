# CAPRI official Phase3 — City results

Synthetic-only reader comparison, frozen real-fit scheme estimation, and targeted generation. Historical optional image refiners are excluded. This report does not claim real transcription accuracy or unseen-data generalization.

| Reader | State | Best selection CER | Selection exact | Selected epoch |
|---|---|---:|---:|---:|
| crnn_plain | complete | 7.09% | 73.58% | 9 |
| crnn_aux | training | 13.90% | 56.31% | 5 |
| svtr_plain | not_started | — | — | — |
| svtr_aux | not_started | — | — | — |
| parseq_plain | not_started | — | — | — |
| parseq_aux | not_started | — | — | — |

![Synthetic selection curves](learning_curves.png)

| Reader / test | CER | Exact text | Decoded count | Count head |
|---|---:|---:|---:|---:|
| crnn_plain / test | 7.19% | 74.29% | 99.98% | — |
| crnn_plain / font_test | 8.17% | 67.95% | 99.93% | — |
| crnn_plain / layout_test | 65.33% | 0.00% | 3.49% | — |

Per-length and digit/lowercase/uppercase precision, recall, F1 and multilabel accuracy are preserved in each synthetic_results.json. Layout test compresses the entire image horizontally and pads white; it is a stress test, not an independently rendered shortcut-free corpus. Heldout font records may mix heldout and seen fonts; no heldout-font records enter fitting or selection.

## Provenance and limits

The original 100k corpus and all historical datasets/vendor files remain intact. Full independent audit rehashed100k unique synthetic images and1599fit/400validation real images. Real split files and pixels matched; no cross-boundary duplicates. The real directory has2000PNG images, no annotation sidecar; the manifest selects1999. No filename-derived labels are used. Independent real transcription ground truth is unavailable, so no CER or accuracy is reported for real images. Historical split exposure prevents a final generalization claim; separate unexposed data would be required.

The initial joint span/gap metadata probe was algebraically redundant: span / mean gap +1 equals count. Its100% score alone is not independent evidence. Follow-up single-cue probes: normalized center gap96.11%, mean glyph width50.90%, mean height59.96%, versus17.23% majority baseline. These use known synthetic glyph geometry, not image detections. No shortcut-resistant counting claim. Frozen CRNN layout follow-up on1024 previously exposed synthetic test images: ordinary CER7.60%/count100%; width shrink plus white padding CER64.68%/count3.32%; replicated-edge padding CER64.64%/count3.32%; resolution reduction with original layout restored CER11.99%/count99.71%. This supports layout sensitivity; it does not prove a unique internal mechanism. These follow-up tests are development diagnostics.

Auxiliary improvement is a matched joint-head hypothesis test, not proof of counting mechanism. One seed and different parameter/FLOP counts limit model rankings. DarkBay changed fonts, length and training duration together and cannot isolate font diversity. Raw confidence definitions differ: CRNN includes all CTC frames/blanks, SVTR uses emitted nonblank tokens, and PARSeq includes tokens/EOS. Do not compare their raw confidence values as calibrated exact-string probabilities. Synthetic reliability bins are separate; no real calibration claimed.

Failures preserved: run01 first full batch failed on nonfinite float16 scaled gradients. Run02 uses bfloat16 without scaling; full-batch tests passed. Initial small test needed Reader.to(cuda) to propagate nested Lightning device metadata; local import precedence was also corrected. First layout diagnostic had an invalid replicate-padding tuple; partial01 and corrected02 retained. No failed historical optional refiner was reintroduced.
