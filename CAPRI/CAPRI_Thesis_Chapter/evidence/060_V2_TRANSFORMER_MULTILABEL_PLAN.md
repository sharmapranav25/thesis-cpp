# Transformer multi-label comparison

User explicitly requests transformer training instead of CNN. Finite synthetic-only experiment; preserve CNN, vendor files, previous experiments and release. Renderer-parameter experiment has not been implemented or trained; this run concerns feature classification only.

## Model

Pure image-patch transformer built with installed open-source PyTorch. RGB192x96 split into288 patches8x8, projected linearly to96dimensions. Learned positional embeddings;15feature query tokens;3pre-norm transformer encoder blocks with4attention heads,384-wide feedforward, dropout0.2. Each feature token has its own binary head. All patches attend together: this replaces the CNN's multi-scale feature-map pooling rather than calling that pooling a transformer. No CNN backbone, pretrained weights, feature-pop or runtimeVLM. No new packages/downloads needed. Raw RGB only.

## Controlled training and fresh comparison

Reuse exact hash-recorded8000training,1200selection and1000calibration images from CNN run. Same15feature definitions, label-validity rules, augmentation, AdamW learning rate0.0005/decay0.02, batch32, gradient clip3, three seeds731/732/733, max40epochs, minimum8, patience6 on selection macroAP. Same calibration/gates. Models differ in architecture/parameter count and compute; this is not FLOP-matched. Identical generic optimizer settings are a controlled baseline, not a claim of optimal transformer tuning. Train/selection AP gap retained.

Do NOT reuse the CNN's already-consumed locked test images as fresh evidence. Generate800each same-distribution/font/combination/recipe cases using304m..307m seeds, with the same definitions and font/combination boundaries. Freeze all three transformer checkpoints before evaluating. Replay all three frozen CNN checkpoints on these exact fresh cases with their original synthetic-calibration thresholds. Write cnn_fresh_comparison.json alongside transformer results. No tuning on these tests or real images. Tests become development evidence after this run. Reused corpus support and duplicate hashes re-audited. Same recipe limits as CNN: only curves/gradient/texture explicitly change recipe; no generalization claim for every possible effect/library.

Each feature must meetF1>=0.85,recall>=0.85,FPR<=0.10 across every seed and fresh stratum. No average-score promotion. Even passing synthetic gates does not establish real transfer. Preserve all failures. No production promotion or automatic ongoing loop.

## Research attribution

Dosovitskiy et al., An Image is Worth16x16Words: Transformers for Image Recognition at Scale (ICLR2021), https://arxiv.org/abs/2010.11929. Primary abstract consulted: pure transformer over image-patch sequences. Borrowed this high-level architecture idea. Our8x8patches, compact model,15learned feature tokens and synthetic-only scratch training differ from the paper; no paper reproduction or equivalent performance claimed. Update THESIS_PAPERS.md with this distinction and results.

## Operations

Separate run/status/monitor; exclusive run lock; source hashes frozen. No active training jobs observed before launch. Corpus/outputs reside in feature_transformer_multilabel_v1/run. Monitor endpoint monitor/transformer_multilabel_status.json. Finite worker completes three-seed training, calibration, fresh evaluations and baseline replay, then records actual gates. Existing partial corpus only reused with same code hash; interrupted seed restarts rather than claiming optimizer resume.
