# Length experiment 02

User-authorized test of a repaired direct count classifier against a class-agnostic character-instance transformer. Synthetic-only fitting; no OCR/character-type output, target generator/font assets or VLM in the algorithms. This experiment does not modify the active generator's length distribution.

## Frozen comparison

- Two independently trained model types, three seeds each (731–733).
- Shared corpus: 3,300 train, 440 selection, 440 calibration; three disjoint 440-image tests (same distribution, ten held-out fonts, nonuniform layout).
- Counts 0–10 balanced. Primary range 2–8; 0, 1, 9, 10 are diagnostics, not evidence for arbitrary lengths. More than ten is unsupported by the count head.
- 25 training fonts from the 35-font bank; ten held out from fitting/selection/calibration. Native canvas 256×80. Every character width/height drawn independently of count. Occupied span also sampled independently; visibility rejection can reintroduce residual dependence and is audited.
- Synthetic background rectangles, strokes and dots; no real style profile is used in classifier training. Per-glyph masks produce box targets but are not model inputs.
- 1,200 optimisation steps per model/seed, batch32, AdamW0.0005, weight decay0.02. Selected every100 steps on synthetic selection count-head MAE, then exact accuracy.
- CNN uses a convolutional encoder and11-way count head. Instance variant adds two transformer decoder layers, twelve object queries, character/no-object logits and boxes, bipartite matching plus the same global count objective. This is a small DETR-inspired test, not original DETR or CRAFT.
- Instance threshold selected on synthetic calibration only. The instance model's checkpoint is selected by its global count head, a disclosed limitation if the instance branch peaks at another time. Both outputs are reported separately; no post-hoc winner blending.
- Real inference: first up to64 unique cached Phase1 fit references per family, preserved order. No real text labels are read. Filename strings only locate files and are not fed to models. Evaluation directory guard enforced. Historical development exposure still applies.

The synthetic held-out-font and layout tests isolate two shifts, not every real-domain mismatch. Background and glyph widths remain narrower than real CAPTCHA possibilities. Independent recogniser jobs were checked before launch; GPU was idle. Do not infer OCR gains from these count scores.

## Evidence

`protocol.json`, `*_metadata.json`, `pixel_hashes.json`, `*_history.json`, checkpoints and per-test NPZs retain the run. `status.json` records actual progress. `real_inference.json` is produced after frozen synthetic evaluation. Accuracy on real images is not established without independent count-only reference annotations.

Research rationale and primary paper citations are in the preceding `CAPRI_typography_v3/LENGTH_RESEARCH.md`. DINO attention peaks were not retried because the previous experiment failed to demonstrate instance counting. This test uses explicit synthetic instance supervision.
