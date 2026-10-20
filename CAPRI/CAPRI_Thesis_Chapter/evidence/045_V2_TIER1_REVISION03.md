# V2 Tier1 revision03: native-resolution feature recovery

Architectural diagnosis on the same synthetic images as02, now development benchmarks. No fresh acceptance claim. Five feature order: line, curve, dot, rectangle, grain. Added full-resolution detail branch, stride2 skip branch and full-resolution decoder; presence sees mask mean/top32 evidence as well as pooled views. 1800steps vs02 1500, so improvements cannot be attributed solely to architecture. All three view alternatives retrained, synthetic only. Real inference after checkpoint selection only. Density/geometry heads remain deferred.

| Model | Metric | Revision02 | Revision03 |
|---|---|---|---|
| whole_image | f1 | 0.559, 0.685, 0.601, 0.725, 0.959 | 0.609, 0.744, 0.845, 0.937, 0.997 |
| whole_image | false_positive_rate | 0.357, 0.183, 0.484, 0.212, 0.028 | 0.385, 0.343, 0.112, 0.038, 0.004 |
| whole_image | visible_mask_iou | 0.056, 0.216, 0.006, 0.320, 0.721 | 0.135, 0.274, 0.379, 0.601, 0.470 |
| single_grid | f1 | 0.611, 0.670, 0.562, 0.717, 0.988 | 0.591, 0.768, 0.868, 0.937, 0.983 |
| single_grid | false_positive_rate | 0.425, 0.307, 0.488, 0.343, 0.000 | 0.365, 0.279, 0.024, 0.008, 0.020 |
| single_grid | visible_mask_iou | 0.061, 0.216, 0.004, 0.446, 0.716 | 0.135, 0.266, 0.376, 0.574, 0.457 |
| multiview | f1 | 0.558, 0.672, 0.580, 0.691, 0.991 | 0.595, 0.760, 0.863, 0.903, 0.994 |
| multiview | false_positive_rate | 0.472, 0.414, 0.516, 0.225, 0.000 | 0.393, 0.335, 0.048, 0.008, 0.004 |
| multiview | visible_mask_iou | 0.063, 0.205, 0.020, 0.443, 0.724 | 0.135, 0.270, 0.405, 0.536, 0.455 |

No automatic promotion. Same procedural generator and contrast/semantic limitations remain. Reused test data are development data. Fresh independent recipe testing and false-positive controls remain necessary. Native-resolution IoU compared with02 resized-mask IoU is not strictly like-for-like; presence metrics use identical labels. No new paper reproduction claimed.

## Fresh-seed validation and common-resolution localization

420 calibration images and 420 each fresh same-recipe, font-only shift, and curve-recipe-only shift. Seeds disjoint from development corpus. Fonts remain within the same nine-family scope. Original procedural generator remains shared: this is not independent-source validation.

| Multi-view feature | 02 fresh F1 | 03 fresh F1 | 02 FPR | 03 FPR | 02 IoU | 03 IoU |
|---|---:|---:|---:|---:|---:|---:|
| line | 0.565 | 0.627 | 0.525 | 0.395 | 0.074 | 0.143 |
| curve | 0.706 | 0.737 | 0.457 | 0.360 | 0.220 | 0.398 |
| dot | 0.587 | 0.860 | 0.575 | 0.028 | 0.018 | 0.211 |
| rectangle | 0.585 | 0.828 | 0.252 | 0.016 | 0.422 | 0.416 |
| grain | 0.985 | 0.997 | 0.004 | 0.000 | 0.725 | 0.757 |

Clear synthetic improvement in dot/rectangle presence and sparse localization. Full-view aggregation still does not consistently outperform simpler alternatives. Fixed-threshold line/curve FPR remains high. The calibration trial chose thresholds on separate synthetic Anton-font samples, subject to calibration FPR<=0.10. It generalizes poorly: fresh multiview line recall falls to0.236 and rectangle recall0.404. REJECT this calibration as a deployment policy; it illustrates font/selection distribution sensitivity, not a fix. Report fixed and calibrated results separately; never hide lost recall.

Assistant visual review of20 previously exposed real fit examples finds uncertain rectangle responses, including missed backgrounds. No manual feature labels, quantitative real accuracy, rendering promotion or real gate pass. Next substantive work: varied synthetic background strips/low-contrast blocks; semantic separation of nearly-straight curves; feature-isolated examples and glyph-only hard negatives; perceptual visibility audit; multi-font calibration with reserved seeds. Do not declare native resolution alone sufficient.
