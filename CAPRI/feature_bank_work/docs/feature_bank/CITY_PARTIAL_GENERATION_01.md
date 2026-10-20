# City partial-reference generation

**OBSERVED:** At the user's request, stopped further City processing and generated 32 examples from 1,448 complete saved predictions out of 1,999 registered images (43,440 overlapping bricks). The processing-order subset is not a random or established representative sample. Earlier artifacts remain intact; the interrupted run is marked separately. Other datasets were not restarted.

[Generated gallery](../../runs/feature_bank_city_partial_generation_01/city_generated.png) · [Reference/prediction gallery](../../runs/feature_bank_city_partial_generation_01/predictions.png) · [Recipes](../../runs/feature_bank_city_partial_generation_01/generation_recipes.json) · [Counts](../../runs/feature_bank_city_partial_generation_01/summaries.json).

Predicted image-level presence: straight interference 1,448; curved interference 1,424; dots 236. Curves: 22 absent and 2 uncertain; dots: 1,183 absent and 29 uncertain. An image is positive if any brick is positive. Positive brick fractions are 51.18%, 19.43%, and 0.67%, respectively; these overlapping counts are not occupied-area/density measurements. No real accuracy or audited prevalence is claimed.

**OBSERVED:** Reused the frozen synthetic-trained models and synthetic-only revised thresholds. No threshold or weight was fitted to City labels. Generation samples predicted feature combinations and reference dimensions/median background colors, using the existing Python renderer, audited library fonts and instrumented Pillow effects. Generic five-character alphanumeric text, effect geometry/density and font choices are retained; PHP renderers are not used.

All 32 attempts passed the per-glyph contribution check. **PASS means text visibility, not successful style matching.** Assistant inspection of eight deterministic reference panels and all 32 generated examples shows a clear mismatch: reviewed City references contain small grayscale digits and layered nearly horizontal gray strokes, whereas outputs contain larger alphanumeric lettering and generic colored waves. The classifier often flags curve regions around glyphs. These are feature-guided demonstrations, not City replicas. Density, direction, glyph style and richer background matching remain unresolved. No target text was transcribed or used as a label.

These references now guide generation and cannot be called untouched evaluation inputs. Artifacts and this Markdown record are saved in the staging workspace; publication to the canonical E: project is not claimed.
