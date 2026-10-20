# Current pipeline sequence - user revision

This section supersedes the historical phase numbering below. Historical run names and artifacts remain unchanged.

1. Phase 1 stays unchanged: coherent 10x3 blocks, CV measurements plus synthetic-trained frozen feature inference, general rendering-function mapping.
2. Phase 2 is font adjustment (formerly Phase 3). Feed Phase 1 output directly into font selection/typography adjustment. Do not apply the former Phase 2 recipe or image refiners, including their background offsets or typography gains. Keep background rendering intact while assessing font changes.
3. Phase 3 is optional distribution/refinement work (formerly Phase 2). It is deferred. First show and evaluate Phase 1 plus font adjustment across the five families and new-source tests; then discuss whether refinement is needed. Do not automatically resume CNN/transformer/GAN refinement or recipe optimization.

New runner: scripts/run_style_fonts_direct.py. Historical scripts and checkpoints remain for reproducibility. Fonts and feature classifiers train on synthetic only; real images are inference/development references. No runtime VLM, no target-family generator libraries/fonts, preserve split boundaries and paper/decision records. Do not claim that older font results were produced in this new order: they inherited former Phase 2 recipes.

