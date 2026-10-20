# Independent dot specialist, parallel to curves

Feature: non-text compact round scatter, radii1..3native pixels. Pixel-sized grain and arbitrary texture are not covered by this initial dot definition. Own CNN weights, training job, pop, data, thresholds and checkpoints. Same ninefont Tier1scope, sixtrainingfontfamilies; other font/evaluation boundaries as curve protocol. Real images never used for fitting.

20000images:6000pairedtraining groups(12000images),1000selection,1000calibration,500each locked same/font/clustered-layout/OpenCV-rasterizer groups. Seeds91m..97m, disjoint from curve81m..87m. Common backdrop includes glyphs, gradients, blocks, grids and long lines, with NO added dot scatter. Positive adds6..48disks;80%of negatives substitute short dashes,20%add no marks. Centers/colors/radii preserved per pair except absent-negative choice. Round marks can overlap or resemble text; visible contribution gate>=24pixels and>=.4support; ambiguous cases remain a limitation. Positive versus dash occupied area is NOT matched, so density/area shortcuts require separate future auditing; do not claim independent-source robustness.

Feature-pop is maximum of grayscale bright top-hat and dark bottom-hat responses with9x9elliptical kernel. Deterministic, input-only, concatenated toRGB; compare against rawRGB. It emphasizes small structures, not guaranteed dots, and can also highlight text/strokes. No target masks used for pop/inference.

Eight individual grids, raw/pop variants, then top-two-grid combination; select on selection AP only. Up to12screening epochs and24confirmation epochs for seeds431/432/433. Calibration separately chooses threshold with empirical FPR<=.05. Eachseed/lockedstratum acceptance F1>=.90,recall>=.90,FPR<=.05, with uncertainty/counts reported. No automatic production promotion. Same architecture/training recipe as curve worker is reused as CODE, not shared model weights. Learning quality remains unproven.

One extra GPUworker initially. GPUheadroom sampled before launch:RTXA4000,16376MiBtotal,1624MiBused,37%utilization. Sampled capacity is not guaranteed future peak; inspect both jobs after launch. Separate outputs/OSlocks prevent duplication. Monitor/specialists.html polls per-worker status so jobs do not overwrite each other's display. No recurring automation restart. Preserve current curve job and all histories.

Code built additively via build_dot_specialist.py; no change to frozen curve training. No external paper directly adopted. This is only the dot specialist, not a claim that all feature classifiers are implemented.
