# Phase 2 pilot protocol 01

User explicitly approved Phase 2 exploration after reviewing the Phase 1 limitations. This is an authorization to experiment, not a retrospective Phase 1 gate pass.

See D033-D034 in STYLE_DECISIONS.md for the frozen initial split, budgets and comparisons. Five original families; 96 fit and 32 development profiles/family. Final historical validation hash window 80:120, 40/family, must not train or select parameters/models. Prior global exposure is disclosed. Fit-only scales and synthetic-trained feature-classifier outputs are frozen. No real transcription, source family generator/fonts, pretrained models or runtime VLM.

Recipe models predict bounded adjustments for a non-differentiable general-purpose renderer. CNN and transformer are trained to imitate black-box recipe-search targets, not to label real features. A local discriminator supplies an additional candidate-ranking comparison. Direct CNN/transformer/GAN refiners train on unpaired synthetic and unlabeled real fitting images, preserving all synthetic glyph pixels and a surrounding band. This does not solve missing fonts or character-count estimates.

Pilot evidence requirements: unchanged feature-classifier hashes; successful render integrity and unchanged labels; report style-distance changes on the reserved window only after selection; report variability, all failures, wall time and example grids. An exploratory improvement requires >=5% lower average independent style distance and improvement in >=3/5 families, no glyph-integrity failures, and no obvious visual collapse. Scores alone do not establish visual success. Training losses are diagnostic only. Preserve failed methods and the no-change baseline.
