# Joint recognition and renderer-control learning: two architectures

User clarification,2026-09-28: learn available library effects, recognize them in image grids, and infer a feature-to-function recipe. Test BOTH gridded pseudo-token processing and bidirectional CNN/transformer coupling. No real fitting, target-family code/fonts, runtimeVLM or default-release changes. Explicit experimental mapping tests are allowed even when quality gates fail; never promote automatically.

## What is actually learned

15 feature-presence labels from existing synthetic operation bank, plus9regression targets: straight count/meanwidth, curve count/meanwidth/meanamplitude, dot count/meanradius, gridspacing, blurstrength. These parameters are traced exactly from the same independently seeded Pillow/OpenCV operations that produced the training image. Ground-truth settings supervise training only. Runtime consumes pixels, not seeds/recipes. A fixed adapter translates named outputs into compatible API arguments; models do not discover arbitrary Python functions or recover original proprietary generators. Counts are requested operation counts; after occlusion they may not be identifiable. Loss excludes feature effects with insufficient counterfactual visibility. Parameters are aggregate settings, not exact object positions or per-instance geometry.

Colours, spatial layouts, fonts and unsupported settings remain existing CV/Phase2 or generic controls. All15features have classification outputs; only9numeric controls have regression supervision in this bounded test. Renderer supports direct learned controls for lines/curves/dots/grid/blur; other mapped labels gate existing CV recipes. No claim of learning every possible library parameter. No feature-pop. RawRGB192x96.

## Architectures and attribution

Gridded variant: CNN local embeddings,30spatial pseudo-tokens on10x3 grid, local cross-attention within grid cells,3transformer blocks, per-feature pooling. Borrowed grid-encoding/processing concept from Ashman et al., Gridded Transformer Neural Processes for Spatio-Temporal Data, ICML2025, https://proceedings.mlr.press/v267/ashman25a.html ; preprint sections4.1–4.3 https://arxiv.org/html/2410.06731v2 . This is an image classifier/regressor inspired by the paper, NOT the full conditional neural-process model: no arbitrary target-coordinate decoder, probabilistic field likelihood, temporal inputs or claimed translation equivariance. Official reference repository MIT; no code copied.

Bidirectional variant: spatial CNN stream and transformer tokens exchange local-to-global and global-to-local features in3successive blocks. Borrowed interactive feature-coupling idea from Peng et al., visual Conformer, ICCV2021, https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Conformer_Local_Features_Coupling_Global_Representations_for_Visual_Recognition_ICCV_2021_paper.html ; official Apache2.0 reference https://github.com/pengzhiliang/Conformer . Original small adaptation with gated interpolated feedback, not a paper reproduction. Existing framework nonlinearities remain SiLU/GELU; no claim that activations alone learn stroke continuity.

Implementation links: models.py, controls.py, train.py, generate.py. Exact target reconstruction tested against instrumented generator output; all15classification and9regression outputs receive gradients. Architecture parameter counts recorded by check.py. Not compute/capacity matched; comparison tests these concrete candidates, not a causal architecture proof.

## Training and validation

Same8000synthetic train/1200selection/1000calibration data as earlier comparison, copied with hashes. Fresh404m..407m seeds for four800-image test strata; same font/combination boundaries. Three seeds per architecture, max40epochs, minimum8/patience6, dropout0.2, AdamW/weightdecay/gradientclipping. MaskedBCE plus0.5maskedSmoothL1 for normalized active-feature parameters. Select checkpoint on selection macroAP minus0.25parameterMAE; do not use real outputs or test scores for selection. Record train-validation gap and every per-feature score.

Classification gates unchanged F1/recall>=.85,FPR<=.10 each seed/stratum. Parameter validation reports per-control normalized/physicalMAE against training-positive median baseline. No parameter-success claim unless it improves on that baseline consistently; exact count recovery is not guaranteed. Visibility uncertainty applies to both heads. Gates report classification separately from parameter quality. Save failed outcomes.

After freezing each architecture, run all6909 previously exposed real source images through both heads. Generate matched cases using learned controls versus the SAME classifier's training-median controls, preserving text/reference/font/seed. This isolates control prediction better than changing labels and mapping together. Rendering RNG can still shift with changed counts; report that caveat.21cases×5families×2controlmodes per architecture; synthetic labels not recognized real text. Final comparative galleries require modest assistant visual review; subprocess generation itself performs noVLM. Previous production integrity verified. No real accuracy claims from prevalence or cycle self-scores. Controls outside training support remain clipped/approximate; no target-specific rules.

## Execution

Separate run_gridded and run_conformer, exclusive locks, separate monitor files. Run train.py --kind gridded/conformer. Workers finite; automation remains paused. Shared read-only source, separate checkpoints/history. Preserve all prior experiments. On completion generate.py creates inference and matched galleries and then status becomes complete; failed generation is explicitly failed rather than silently declared finished.
