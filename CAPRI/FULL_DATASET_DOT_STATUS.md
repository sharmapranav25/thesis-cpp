# Full-dataset dot inference and stroke status

User authorized full-scheme inference, including evaluation images, on 2026-09-28. Frozen5x3pop screening checkpoint copied before prior review; no real fitting. CPU inference preserves GPU availability for ongoing synthetic training. Results are distribution descriptions, not automatically verified accuracy. Threshold0.5 remains provisional. No runtime VLM.

Active output: full_real_dot_review_02, script full_real_dots.py. Full image directory inventory: CryptBB600, City2000, Courier2510, Darkbay999 (799pool +200eval images), outlined800. Total6909 image files. Actual readable/scored counts and decode failures recorded per family. Previously reserved images are now distribution-exposed; preserve that distinction for future evaluation.

An initial run01 was stopped during City after discovering Darkbay's sibling eval directory. Partial outputs preserved. Run02 includes pool and eval; this was a scope correction, not model/threshold tuning. No duplicate inference worker intentionally left active.

Two-stage strokes completed its controlled synthetic trial and failed its gate: locked positive mask IoU0.7813, detection recall0.9975, absent-image false positive rate0.155, oracle-mask shape accuracy0.8525, predicted-mask end-to-end shape accuracy0.765. It learns, but text false detections and mask-to-shape transfer remain weaknesses. No production promotion; this is not robust multi-stroke validation. Canonical evidence: feature_specialists_v2/strokes_two_stage_v1/run/results.json.

Assistant visual review of the earlier30-image development subset is retained in real_dot_review_01/REVIEW.md. Do not extrapolate it as a whole-dataset verdict. Full-run distribution montages sample score quantiles for subsequent visual inspection. Model scores alone cannot establish correctness.

