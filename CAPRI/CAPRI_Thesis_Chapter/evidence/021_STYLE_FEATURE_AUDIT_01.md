# Frozen feature inference audit01

No training or threshold changes.48 stored synthetic test crops replayed with maximum descriptor difference0.0.9 blank controls score below.008 for all threefeatures. Historical decision_function followed by saved sigmoid matches live inference.

All480 exposed sloth reference blocks score>.99 for straight and curved interference. Straight raw scores11.996–28.223 exceed historical synthetic-test max11.787. Largest straight contributions include gray standard deviation and Canny edge occupancy; curved uses CLAHE edge occupancy and other texture statistics. This supports distribution mismatch/texture confusion, not an inference implementation repair.

## New synthetic dot-only controls
32seeds120000–120031,24x24 whiteRGB blocks, grayscale points intensities100–219, counts0/10/40/100, coordinates independently sampled with replacement. No line/curve operations. At40points all32 curve scores>.99. At100points all32 curve scores>.99 and28/32 straight scores>.99; median dot score falls to.111. Empty controls scorelow. Chance dot alignments are possible; these operation-labeled negatives expose nuisance sensitivity rather than verified perceptual labels.

## Outcome
Do not calibrate thresholds on real images. Next justified revision is synthetic-only hard-negative coverage of dense dots/textures, keeping disjoint base-seed train/selection/calibration/test partitions, fitting a new version while preserving original LR bundles. Compare ordinary synthetic performance and fresh dense-dot controls before frozen real inference. Existing real samples are development evidence, not unseen evaluation. Phase gates remain unpassed.

Evidence: runs/style_feature_audit_01/audit.json and dot_only_controls.json; replay script scripts/audit_feature_inference.py. Initial import failure retained in console log and corrected to the original staging module path. No new paper adopted; original diagnostic.
