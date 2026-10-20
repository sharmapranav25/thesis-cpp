# Active pipeline: Phase3 character count (user override)

Phase1 remains visual extraction/mapping with CV and frozen learned features. Phase2 is font adjustment directly. Phase3 now predicts CHARACTER COUNT ONLY: no text recognition or character-type/alphabet prediction. Train on synthetic images only; real images inference only. Prior color-locked refinement remains optional and OUTSIDE the active pipeline. Continuous automation remains paused at user request; this count-training request does not resume it.

Count pilot supports2-8 characters. Existing character strings remain configurable. Do not automatically change generation lengths from unverified real predictions. See STYLE_CHARACTER_COUNT_01.md.

# Current pipeline sequence - user revision

This section supersedes the historical phase numbering below. Historical run names and artifacts remain unchanged.

1. Phase 1 stays unchanged: coherent 10x3 blocks, CV measurements plus synthetic-trained frozen feature inference, general rendering-function mapping.
2. Phase 2 is font adjustment (formerly Phase 3). Feed Phase 1 output directly into font selection/typography adjustment. Do not apply the former Phase 2 recipe or image refiners, including their background offsets or typography gains. Keep background rendering intact while assessing font changes.
3. Phase 3 is optional distribution/refinement work (formerly Phase 2). It is deferred. First show and evaluate Phase 1 plus font adjustment across the five families and new-source tests; then discuss whether refinement is needed. Do not automatically resume CNN/transformer/GAN refinement or recipe optimization.

New runner: scripts/run_style_fonts_direct.py. Historical scripts and checkpoints remain for reproducibility. Fonts and feature classifiers train on synthetic only; real images are inference/development references. No runtime VLM, no target-family generator libraries/fonts, preserve split boundaries and paper/decision records. Do not claim that older font results were produced in this new order: they inherited former Phase 2 recipes.

# Two-phase style pipeline — protocol 01

User-authorized scope expansion, 2026-09-28. This supersedes the earlier restriction to classifiers/matching only, without rewriting historical records. Repeated retraining is authorized. No runtime VLM or target-family generator code/fonts. Assistant visual review may supplement objective diagnostics; it is disclosed and modest, not ground truth. All executable dependencies, models and fonts must have open-source licenses recorded. Reference-image usage permission is inherited from existing research pools; do not claim that every private reference dataset is openly licensed.

## Phase 1

Extract native-pixel overlapping 10x3 blocks plus global context. Estimate background/foreground colors, local occupancy, directional structure, approximate glyph components/layout/tilt with uncertainty. Reconcile blocks using spatial adjacency and global lines; don't sum overlapping pixels or mistake PCA glyph axes for verified rotation. Parameter mapping returns traceable general-purpose drawing functions. Do not use source-family libraries, transcribed real text or real-image crops as generation assets.

Initial development experiment: hash-order sample up to 128 historical fit images/family; 40 historical validation images/family for diagnostic comparisons. Historical exposure of these pools has already occurred; call validation reused, not pristine. Preserve external excluded pools. Parameter fitting/search uses fit only. Generic procedural baseline and conditional candidates use the same fonts, text and seed budgets. Metrics use training-only scaling.

Predeclared exploratory Phase 1 gate: all renderer integrity checks pass; each family has its own generated descriptor centroid nearest its own real validation centroid for at least 4/5 families; conditional descriptor distance improves over family-agnostic procedural baseline by >=20% for at least 4/5 families; modest disclosed visual review rates broad layout/color/background resemblance at least partial for 4/5 families, with no catastrophic unreadability. These are project engineering criteria, not established research benchmarks. Distinctness alone cannot pass. Record failures and fix them; do not weaken a failing gate after viewing results. Avoid relying solely on extraction descriptors: visual review and independent held-out measurements supplement the metric.

## Phase 2 (gated)

Only after an explicit Phase 1 gate record: conventional CNN refiner, transformer refiner, GAN refiner. Same data splits, image resolution, initial training-step budget and text-preservation probes. Model licenses checked before use; train small local models from scratch if appropriate. Compare distribution/style distance, family distinctness, unchanged synthetic character labels, stability, throughput and memory. No success claim from training loss alone. Retain failures/checkpoints, repeat training on development evidence, record incremental held-out exposure.

## Operations

Canonical new implementation goes in `style_pipeline/`, new immutable runs in `runs/style_phase1_*` and `runs/style_phase2_*`. Log substantive choices before execution in STYLE_DECISIONS.md and append outcomes. Research log must distinguish papers read from ideas merely suggested. No external messaging or paid compute. Verified local PyTorch 2.5.1+cu121 with an NVIDIA RTX A4000 is available. Use small pilot budgets first; no unbounded GPU jobs. Overnight same-chat continuation checks existing processes before starting another run.

## Current three-phase authorization (historical gates preserved)

The user subsequently authorized exploratory Phase2 despite unresolved Phase1 gates, added Phase3 synthetic-trained font adjustment, and requested repeated improvement loops. Earlier gate text above is historical protocol, not a claim of passage. Current loop: diagnose fit/development failures, implement a justified revision, train classifiers on synthetic only, freeze before inference/evaluation, perform both CV and learned inference, compare actual renders, and document failures. Phase2 may fit unlabeled real fit distributions. Each completed loop needs an independently sourced new-family test; preview-only checks are smoke tests. Reused images cannot become unseen again. Maintain THESIS_PAPERS.md, decisions, handoff and monitor. Completing one run does not complete the ongoing loop.


LATEST USER ORDER: Phase1 now includes synthetic-trained frozen CHARACTER COUNT inference with uncertainty, alongside CV and frozen style features. Phase2 font adjustment. OptionalPhase3 refinement remains outside default pipeline; DINO comparison only proposed. No character-type inference. Count uncertainty is not a verified distribution or +/-1 bound. AutomationPAUSED.


LATEST USER OVERRIDE: remove character-count inference from active pipeline. Phase1 CV+frozen style features; Phase2 font adjustment; optionalPhase3 now testing frozen DINO feature/attention-guided noncolor parameter search. Fixed5 characters is a generic preview setting, not a family count estimate. Automation remains paused.


LATEST SUPPORTED IMPLEMENTATION: releases/phase12_v1, docs/PHASE12_START_HERE.md. Phase1+Phase2 only; 2–8lengths ×numeric/alphabetic/alphanumeric generation priors. No real type/count inference. This session did packaging/validation only. Next session owns PARSeq implementation/training/evaluation. Historical variants retained outside release.
