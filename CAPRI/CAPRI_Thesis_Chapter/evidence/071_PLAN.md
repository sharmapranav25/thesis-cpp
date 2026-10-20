# CAPRI Meta-Sim-inspired distribution refinement 01

## Scope and attribution

User asks whether Meta-Sim can bring existing CAPRI generations closer to real images. Trial uses frozen CAPRI seed731 hybrid CV-preserving profiles, with independently initialized per-family residual neural adapters. This is a small adaptation of the distribution-matching part of Meta-Sim, not a full reproduction or an official pretrained Meta-Sim model.

Primary source: Kar et al., *Meta-Sim: Learning to Generate Synthetic Datasets*, ICCV2019, https://arxiv.org/abs/1904.11621 and https://arxiv.org/pdf/1904.11621 . Read sections3.1–3.2.3. Directly borrowed: immutable versus mutable renderer attributes; identity starting point; feature-space Gaussian MMD; finite-difference gradients through a nondifferentiable renderer. Differences: flat8-input/24-hidden/6-output MLP instead of scene-graph GCN; cached frozen DINO ViT-S/8 CLS instead of Inception features; zero residual initialization instead of autoencoder pretraining; no labeled downstream task objective, REINFORCE, or PARSeq training. The paper's "Distribution Transformer" is not a modern attention-transformer architecture.

DINO: Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, ICCV2021, https://arxiv.org/abs/2104.14294 . Reuses previously cached official model/weights with weights SHA256 recorded. DINO is frozen and used only for a representation-space distance, not as a language/VLM system and not proof of perceptual or OCR quality. The code does not use attention heatmaps in this trial. No downloads or copied target generators/fonts.

## Fixed protocol

- Five existing families,96distinct cached-profile images each:48fit,16selection,32report. Deterministic salted decoded-pixel-hash ordering. All have historical development exposure; cached font/profile decisions used broader data earlier. This is within-trial separation, NOT pristine unseen-family validation.
- Fit recipes only supply generation. Report images do not supply generation recipes and are opened for metric evaluation only after adapter selection is frozen. Real filenames are paths only; labels are never parsed. Splitting requires hashing all decoded pixels, but report pixels are not used by fitting/selection objectives.
- Independent adapter per family,12steps,batch4,Adam.035,gradient clipping1,selection every4steps,identity baseline retained as eligible checkpoint. No report-based checkpoint selection or restart.
- Mutable gains: linecount,line width,dotcount,curvecount,curve amplitude,measured blockcount. Predicted gain range.25–4. Existing primitive caps still apply. If no supported base primitive exists, a gain cannot invent it. Six mutable dimensions use symmetric finite differences epsilon.2 in log-gain space.
- Preserve generated characters/count, glyph geometry, fonts, exact glyph RGB, source palettes and gradients. Renderers retain visibility tests; invalid candidate returns matched baseline and is counted. Changes in background coverage can change RGB histograms without recoloring source palettes. Generic existing drawing colors remain; no RGB-offset learning.
- Training loss: biased Gaussian MMD squared on normalized DINO CLS. Kernel bandwidth from fit features only. Weak output regularization. Model weights and CAPRI classifiers stay separate.
- Evaluation:64fresh paired generated images per family, from fit recipes only;32report real images. Compare DINO MMD, paired bootstrap delta intervals300resamples, and independent12-dimensional CV diagnostic standardized using fit data. CV diagnostic is not trained/selected on and is not a validated perceptual metric.
- Preview first6 deterministic pairs; real references are unpaired report examples. Known texts2–8/numeric/alphabetic/alphanumeric remain generic. No claim to infer real character lengths or character set.

## Artifacts and decisions

Implementation: capri_metasim_trial.py in thesis workspace. Output capri_metasim_01: protocol, per-family split manifests/checkpoints/freeze/histories/actual operation recipes, before/after PNGs, comparison sheets and aggregate results. status.json reports actual PID/stage. Base CAPRI manifest checked before and after; no base package files changed.

Decision: start with bounded12step pilot to establish whether neural recipe refinement shows any evidence of benefit before larger training. Do not call lower optimized DINO distance alone "closer"; report independent metric disagreement and visual failures. No automatic promotion.

Decision: no literal editing of old generated PNGs. Meta-Sim changes recipes and re-renders matched text/seed examples. This preserves label provenance and follows the previously discussed optional recipe-refinement stage.

Decision: use the immutable packaged seed731 hybrid, not later completed training seeds, to isolate the effect on the user's packaged experiment. Both original multi-seed training jobs completed before this trial; no duplicates launched.

Results pending; append outcome after evaluation and modest assistant visual review.
