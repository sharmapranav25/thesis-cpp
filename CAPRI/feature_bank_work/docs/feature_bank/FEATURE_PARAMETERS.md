# Feature density and tilt after classification

**PROPOSED following user feedback, 2026-09-28:** Extend the recognition path to local feature classification -> image-derived density/orientation estimates -> spatial aggregation -> compatible rendering functions and parameter ranges. The user suggested these two quantities; the definitions below are a proposed implementation design. No estimator has been implemented or validated yet.

## Meaning depends on the feature

| Feature | Density measurement | Orientation measurement |
|---|---|---|
| Dot scatter | Distinct mark centers per 10,000 native pixels; separately occupied pixel fraction | Not applicable unless an independently supported directional arrangement exists |
| Straight interference | Stroke length per 10,000 native pixels; separately occupied pixel fraction | Undirected angle modulo 180 degrees, possibly multiple dominant directions |
| Curved interference | Curve length per 10,000 native pixels; separately occupied pixel fraction | Local tangent directions and their spread; one global tilt can be misleading |
| Periodic grid (unsupported currently) | Spacing/line frequency for each direction | Two or more grid directions |
| Lettering | Separate text occupancy if needed | Separate glyph rotation/slant from baseline tilt; do not conflate with interference |

Use native coordinates with x right and y down. Zero degrees means horizontal; positive angles turn toward increasing y. An angle is an axis, so 0 and 180 degrees are equivalent. Density is normalized for area, while original image dimensions are retained for later rendering. Pixel coverage and object count are different quantities: thickening a line changes coverage without adding another line.

## No-cheating boundary

**PROPOSED:** Classify first, then estimate from pixels in regions with credible evidence. Feature confidence does not guarantee parameter confidence; allow an uncertain parameter or multiple orientations even when the feature is present. Do not force a value for absent, unknown or unsupported features.

Synthetic operation records and contribution masks provide supervision and scoring references only. Runtime estimators must never read those masks, recipes, font identities or generator settings. Record requested geometry separately from visible density/orientation after occlusion; invisible marks cannot be counted as detected marks. Neither local class positives nor raw edge counts alone isolate interference from glyph strokes.

Aggregate a spatial evidence map over native pixels. Use non-overlapping core ownership or normalized overlap weights, and reconcile connected components across boundaries. Never sum overlapping crop counts directly. Circular/axial angle aggregation uses doubled angles and reports spread or multiple modes; ordinary arithmetic averaging fails near the 0/180 boundary. Mixed directions must not cancel into a confident single tilt.

## Additional gate before matching parameters

1. Implement verified generator controls with independent density and angle sweeps. Current straight endpoints provide geometry; a scalar tilt control and dot-count/line-count ranges are not yet registered capabilities.
2. Validate paired synthetic examples that change one parameter while preserving text, colors and other operations. Freeze ranges before any target evaluation.
3. Split complete base groups before rendering/cropping. Keep parameter sweeps and paired counterfactuals in the same group; do not fragment them across splits.
4. Test both polarities, low contrast, glyph-like negatives, thick strokes, clipping, occlusion, several directions and different image sizes. Verify density does not increase merely because overlap is enabled.
5. Score normalized-density absolute error, modulo-180 angular error, uncertainty/coverage and errors conditional on correct classification. Also report end-to-end errors so classification mistakes are not concealed. Use base-group uncertainty intervals.
6. Set numerical acceptance thresholds from a declared synthetic protocol, not real test labels. Real parameter evaluation would need additional audited parameter references; existing class-only drafts cannot validate density or tilt accuracy.
7. Only return a rendering function's parameter values when its capability/range is verified. Do not interpret matching as identifying the original generator or successfully recreating the CAPTCHA.

**Current status:** Design only. The three-feature [candidate corpus](CORPUS_CANDIDATE_01.md) records geometry useful for later supervision, but density/tilt sweeps, runtime estimation and parameter-aware matching have not passed any gate. This addition does not authorize bypassing the existing corpus, five-dataset CV, Logistic Regression and Extra Trees stage order.
