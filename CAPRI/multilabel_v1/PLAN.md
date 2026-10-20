# Multi-label feature model v1

2026-09-28. User redirects work from independent specialists to joint multi-label classification, requesting robust training without overfitting. Preserve all historical specialists and the current production release. This finite experiment does not restart the paused automation or train PARSeq. No real classifier fitting or runtime VLM.

## Feature coverage

Train all12 existing feature-bank definitions plus3 directly controllable image effects:

| Feature | Synthetic mapping | Scope |
|---|---|---|
| straight_interference | Pillow lines | Independent non-text strokes |
| curved_interference | Pillow sampled polylines | Sinusoids, held-out quadratics |
| dot_scatter | Pillow small disks | Radius1..3, not all texture |
| decorative_shapes | Pillow ellipse outlines | Circle/ring subset; not every possible decoration |
| periodic_grid | Repeated horizontal/vertical lines | Axis-aligned grid subset |
| background_texture | NumPy spatial noise | Noise texture; held-out smoothed noise, not every real texture |
| background_gradient | RGB ramp | Horizontal training, vertical recipe holdout |
| background_blocks | Pillow rectangles | Non-text patch regions |
| hollow_lettering | Eroded glyph subtraction | Hollow font effect |
| glyph_border | Dilation around filled glyph | Contrasting border |
| glyph_shadow | Offset glyph mask | Offset duplicate |
| nonlinear_text_deformation | OpenCV remap | Sinusoidal text displacement |
| blur | Pillow Gaussian filter | Known image blur |
| glyph_rotation | Pillow rotate | Explicit per-glyph rotation, not inferred intrinsic slant |
| multicolor_text | Independent glyph RGB | Between-glyph colour variation |

Font identity remains the existing separate9-family task. Density, tilt magnitude, colour values, position and size are continuous parameters, not binary presence labels; they are not claimed solved by this model. Boldness/perspective/elastic deformation and arbitrary decorative/texture styles need dedicated definitions and further generators before inclusion. Existing generic capabilities are reused via Pillow/OpenCV, not target-family generators or fonts. No production renderer modifications.

## Data integrity and scope

Raw RGB192x96, no feature-pop (retaining latest preference). Random full-range foreground/background palettes. Text2..8alphanumeric is nuisance, not OCR labels. Single operation can coexist with others; hollow and filled-border effects are explicitly exclusive. Requested positives are checked against exact leave-one-operation-out renders using independently seeded operations. Fewer than20 visibly changed pixels marks that target uncertain and excludes it from loss/scoring, not a false negative. These are procedural-effect labels; counterfactual difference does not guarantee human semantic identifiability. Intrinsic font effects and low-contrast controls remain potential ambiguity.

8000train,1200selection,1000calibration,800each locked same-distribution/font/combination/recipe =13400 images. Seeds201m..207m are disjoint. First6fonts train; last2reserved for font holdout; all9 font files hash-checked. Seventh font not used in this run. Combinations allocated by stable bitmask hash: one bucket excluded from training/selection/calibration and used only for combination holdout. Duplicate pixel hashes forbidden. Each class requires35known positives and negatives per split. Recipe shift applies specifically to curves, gradients and texture; it is not independent-source generalization for every feature. No real images enter training or calibration. Consumed locked data becomes development evidence for revisions.

## Architecture and overfitting controls

Shared convolutional RGB encoder; multiscale feature-map regions1x1,2x2,5x3,10x3 pooled into coordinate-tagged tokens. This is multiscale feature-map pooling, not independently resized RGB crop encoders. Each of15 heads learns separate attention over50tokens and separate sigmoid output; multiple features can be positive. Full-image receptive context is preserved. No localization mask claim: binary feature classification is a distinct task.

Three independently initialized seeds731/732/733; max40epochs each, minimum8, stop after6epochs without meaningful selection macroAP improvement. Save best selection checkpoint, not last. Dropout0.2, AdamW weight decay0.02, clipped gradients, random horizontal flips, bounded model size. Train-subset vs selection AP gap recorded. These reduce overfitting risk; they do not guarantee its absence. No early stopping or tuning on locked tests or real distributions. All checkpoints freeze before locked testing.

Per-head thresholds fit only on separate synthetic calibration at exact score breakpoints, maximizing F1 subject to FPR<=0.10. All-negative operating points retained as failures through recall. For a feature to pass, EACH seed and EACH locked stratum must meetF1>=0.85,recall>=0.85,FPR<=0.10. Report per-feature metrics/confusion counts, not just macro average. No automatic promotion even if synthetic gates pass. Real-family tests come after freeze and cannot establish accuracy merely through prevalence.

## Execution and records

Run check.py then train.py. Hidden finite worker, exclusive run lock, per-job JSON status and append-only events. Existing completed run cannot be overwritten. Partial corpus reused only under unchanged protocol hash; an interrupted training seed restarts rather than silently resuming optimizer state. All historical runs/vendor/default release untouched.

No external research paper newly consulted or reproduced here. This is an original engineering baseline using standard BCE multi-label supervision, shared CNN features and attention pooling. Do not attribute it to an unconsulted paper. Robustness remains pending until measured.

