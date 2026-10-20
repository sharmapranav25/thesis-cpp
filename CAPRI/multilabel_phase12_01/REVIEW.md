# Multi-label Phase1/2 full-family experiment

Frozen seed731 chosen by synthetic selection AP only. Completed inference on6909 real images across five Tier1 families; no real fitting or tuning. All15head scores retained. Seven synthetically accepted heads gate compatible renderer effects; failed heads remain diagnostic. Existing cached Phase1 CV measurements and Phase2 font choices are reused, not falsely described as fresh font training/inference. Existing straight-line settings retain legacy CV/profile decisions because the new straight head failed validation. This is a hybrid experimental adapter, not complete replacement of all old classifiers.

105new images,21per family, cover lengths2..8 across numeric/alphabetic/alphanumeric buckets. Per-reference feature co-occurrence is preserved by looking up the selected reference's predictions alongside its measured style. Full-family prevalence is reported separately; features are not independently resampled from marginal rates. Text labels are generated, never recognized from real images. No family generator/fonts or runtimeVLM. User-authorized assistant visual review only.

Mapping: curves/dots gate existing generic functions; gradient/block/texture evidence can be suppressed by corresponding predictions. Grid adds an explicitly generic axis-aligned function (spacing based on canvas height, not claimed inferred). Blur uses explicitly generic radius0.6, not measured strength. Font adjustment and colours retain existing profile mechanisms. No phase3 training/refinement. Original release integrity verified before/after. Masks/readability checks come from the original renderer before optional blur; no OCR-preservation claim after blur.

| Family | Curve% | Dot% | Grid% | Blocks% | Gradient% |
|---|---:|---:|---:|---:|---:|
| CryptBB |7.0|12.7|84.3|58.2|16.3|
| City |10.5|0.0|1.1|0.0|29.6|
| Courier |55.5|21.4|11.2|20.8|14.8|
| Darkbay |0.0|5.2|68.7|0.2|76.5|
| Outlined text |0.1|9.9|3.7|0.1|100.0|

These are predictions, not ground truth or accuracy. Comparisons show real reference / old Phase1+2 / new experimental Phase1+2. Matched text/font/seed maintained within old/new rows, but changing operations can shift renderer RNG and thus other drawing details; not a fully isolated pixel-level ablation.

Assistant visual findings: City dot predictions are more plausible than earlier dot-only rates, but broad background bands still render as thin lines. CryptBB often loses too much texture/curvature and receives generic grids. Courier receives some incorrect grids and misses rough glyph styling. Darkbay grids are a regression: classifier confuses block patterns with grids and usually misses blocks. Outlined text gradients persist, but curve detection nearly always fails and cached font sizing/style remain visibly poor. Synthetic gates did not establish real transfer. No promotion; preserve comparative evidence.

Generation initially failed serializing NumPy grid-colour integers after inference and the first family's images. Fixed scalar JSON serialization and reused already-completed predictions; no classifier retraining, threshold changes or failed-image cherry-picking. Saved deterministic same-seed renders and final complete status. Full predictions/recipes/renderer snapshot/checkpoint hash retained.

No new research paper used. Production release and vendor files unchanged. Data is historically exposed development plus explicitly authorized full-family distribution exposure; never claim untouched real evaluation.
