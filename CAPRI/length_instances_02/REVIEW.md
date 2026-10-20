# Completed pilot: reject automatic length integration

Both models trained from scratch on synthetic pixels only, three seeds each. Six runs completed in309.6seconds including corpus construction, synthetic evaluation and frozen real inference. Real labels were not loaded. Historical CAPRI and generated corpora were unchanged.

## Primary result: lengths2–8

Mean exact-count accuracy across three seeds,280primary-range images per test:

| Output | Ordinary synthetic | Withheld fonts | Nonuniform spacing |
|---|---:|---:|---:|
| Direct CNN count head |58.21%|59.17%|54.64%|
| Instance transformer's global count head |56.67%|56.43%|52.38%|
| Counting its detected instances |10.83%|11.79%|11.90%|

RESULTS.md additionally reports all440examples per test, including0/1/9/10diagnostics. Those easier/extreme-count cases inflate some overall scores relative to the primary range. Do not compare these harder splits directly against the old count model's88–91% on its original generator and claim a controlled regression.

The detector's box precision atIoU0.5 was14.94–15.08%, and recall13.32–15.55%, on the440-image ordinary test. Thus the detector had not learned reliable character localisation even in the synthetic domain. Its global head's much better count estimates do not validate its individual detections.

## Real inference

64cached style-fit references per family,320total; both models and all seeds inferred on each. No evaluation directory or real text labels were consumed. These are historically inspected development pools, not unseen data.

DarkBay: directCNN ensemble argmax was10for63images and9forone. Instance model's global ensemble head was10for61and9forthree. Median detected-instance count was12for all64. Three-seed count-head agreement was92.19%forCNN and79.69%forhybrid; agreement clearly does not imply accuracy.

Assistant reviewed the fixed first12DarkBay and12City panels. DarkBay boxes repeat across glyphs and background structure; City misses many visible glyphs. This is qualitative failure evidence, not a labelled real-accuracy estimate. Remaining family prediction galleries are available; no blanket manual-review claim is made for uninspected panels.

## What this test does and does not establish

It establishes that these implementations and this training budget are not suitable for setting generation lengths. It does not establish that all character detectors or transformers are unsuitable. This was an original small DETR-inspired model, not CRAFT, original DETR, or a pretrained text detector.

The direct classifier was stronger in this pilot. The instance model was checkpoint-selected using its global count-head score, as frozen in the protocol; it was not selected for its best possible detection score. That limits architecture comparisons. A future detector-specific selection protocol must be declared before new locked tests, not retrofitted to improve this report.

Widths and occupied spans were sampled independently of count, removing the previous explicit width/count clamp. The metadata-only shortcut baseline still reached17.5–20.7%against14.3%uniform chance. Average width variance, visibility rejection and other dependencies remain; “no possible shortcut” is not established. Font/appearance diversity was increased but not exhaustive. Synthetic glyph widths9–20pixels and heights24–48pixels may still differ materially from real images after resizing. No real-driven training modifications were made.

## Next justified development, not yet executed

First prove reliable instance localisation on controlled synthetic data, using detector-specific selection, box/centre precision-recall, hard negatives and wider glyph-scale coverage. Compare centre/region supervision with query detection before adding complexity. Retain the current direct-count model as a baseline. Use new locked tests if their failures guide revisions. Independent real count-only annotations are still required to claim accuracy or calibrated length distributions. Do not solve this by silently reading evaluation filenames or hardcoding six characters.

No model is promoted; default explicit2–8generation remains. Phase2's35-font changes are independent and remain available.
