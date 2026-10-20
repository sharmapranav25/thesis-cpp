# Thesis paper and implementation register

Maintained alongside `STYLE_RESEARCH.md` (chronological reading notes) and `STYLE_DECISIONS.md` (experiment decisions). Last updated: 2026-09-28. **A cited idea is not a reproduction of a paper, and published results are not our results.** No paper-supplied pretrained model is used. No runtime VLM is used.

## Directly used ideas

### DeepFont: Identify Your Font from An Image

- Authors: Zhangyang Wang, Jianchao Yang, Hailin Jin, Eli Shechtman, Aseem Agarwala, Jonathan Brandt, Thomas S. Huang.
- Venue/year: ACM Multimedia, 2015. [Primary paper](https://arxiv.org/abs/1507.03196), [full text](https://arxiv.org/html/1507.03196).
- Read: sections 2.1–2.3 and 3.1, including synthetic/real mismatch, augmentation and multi-view prediction.
- Used: synthetic font-labelled rendering; variation in spacing, proportions, noise, blur and background; averaging multiple image views without transcribing the text.
- Our implementation: `style_pipeline/font_match.py`, `scripts/run_style_phase3.py`. Small original CNN with font and hollow/filled heads; synthetic-only training. The bank has 12 independently sourced OFL font files.
- **Not reproduced:** AdobeVFR dataset, DeepFont architecture/compression, real-image autoencoder adaptation, published accuracy. Our four views differ from their fifteen-view procedure. We do not claim exact source-font identification.
- Evidence: first training/testing run is recorded under `runs/style_phase3_font_01`; outcomes belong in `STYLE_PHASE3_FONT_01.md`, not inferred from the paper.

### Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN)

- Authors: Ashish Shrivastava, Tomas Pfister, Oncel Tuzel, Joshua Susskind, Wenda Wang, Russell Webb.
- Venue/year: CVPR, 2017. [Primary paper](https://arxiv.org/abs/1612.07828).
- Read: primary abstract; this is limited reading, not a full-method reproduction.
- Used as design inspiration: synthetic-to-real refinement, an adversarial local discriminator and a self-regularization term to limit changes.
- Our implementation: `style_pipeline/refiners.py`, direct GAN branch in `scripts/run_style_phase2.py`.
- **Not reproduced:** original architecture, training schedule, history buffer and paper benchmarks. Our exact glyph-mask protection is a project-specific constraint.
- Result: Phase 2 pilot ran; direct refiners were rejected visually for character halos despite lower descriptor distance. See `STYLE_PHASE2_PILOT_01.md`.

### Deep CORAL: Correlation Alignment for Deep Domain Adaptation

- Authors: Baochen Sun and Kate Saenko. Year: 2016. [Primary paper](https://arxiv.org/abs/1607.01719).
- Read: primary abstract.
- Used as a broad motivation for aligning feature-distribution statistics across synthetic and real domains.
- Our implementation: `moment_loss` in `style_pipeline/refiners.py` matches means and diagonal standard deviations of fixed low-level features.
- **Important distinction:** we do not implement CORAL's full covariance loss or its learned-feature adaptation method. Cite as motivation, not “we implemented Deep CORAL.”
- Result: Phase 2 pilot; metrics and visual limitations recorded separately.

### An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

- Authors: Alexey Dosovitskiy et al. Year: 2020 preprint / ICLR 2021. [Primary paper](https://arxiv.org/abs/2010.11929).
- Read: primary abstract/landing page.
- Used: patch-token transformer architecture as background for small recipe and image-refinement models.
- Our implementation: `RecipeTransformer` in `style_pipeline/recipes.py`, `TransformerRefiner` in `style_pipeline/refiners.py`.
- **Not reproduced:** ViT model, pretraining, patch size, scale or classification experiments. No ViT weights loaded.
- Result: both local transformer pilots ran; recipe prediction is exploratory, direct refinement was rejected visually.

## Consulted, but method not implemented

### Detecting Text in Natural Scenes with Stroke Width Transform

- Authors: Boris Epshtein, Eyal Ofek, Yonatan Wexler. Venue/year: CVPR, 2010.
- [Primary paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/201020CVPR20TextDetection.pdf).
- Read: abstract and introduction. Motivates stroke-width consistency as a text cue.
- Our region code uses OpenCV MSER, a distance-transform stroke proxy and contour-hole statistics. **It does not implement ray-based SWT.** Do not list SWT as an implemented algorithm.

### Font Recognition in Natural Images via Transfer Learning

- [Primary paper hosted by the authors' institution](https://www.icst.pku.edu.cn/zlian/docs/20181024110641005904.pdf).
- PDF opened for context; bibliographic metadata and full method have not yet been verified. No specific method was adopted. Complete the citation before using it in the thesis bibliography.

## Implementation documentation and data/font sources (not papers)

- [OpenCV MSER documentation](https://docs.opencv.org/4.12.0/d3/d28/classcv_1_1MSER.html): directly used API in `style_pipeline/regions_text.py`; proposals are not verified characters.
- [Google Fonts](https://github.com/google/fonts): nine extra font files and OFL notices, pinned at commit `23e54b51ddffbc7713c583748e3bd86f62b1fa4a`; hashes in `resources/style_phase3_fonts_01/manifest.json`. Three previous OFL fonts are included from the existing audited bank.
- [Additional CAPTCHA collection](https://github.com/roncuevas/Captchas): dataset source, not a method paper; pinned image sample provenance is retained. Repository license declaration is not independent verification of every source site's rights.

## Maintenance rule

For each new paper, record the primary link, verified citation, sections actually read, exact borrowed idea, code/run locations, deviations, and observed result. Promote “consulted” to “directly used” only when code exists. Keep failed applications in this register. Never turn a paper's reported accuracy into a claim about this thesis pipeline.

## Experiment evidence update — 2026-09-28

DeepFont-inspired local run `style_phase3_font_01` is now trained and tested: 50.52% synthetic top-1, 81.64% synthetic top-3, 92.19% synthetic hollow/filled accuracy. This is **our small model's synthetic benchmark**, not DeepFont's benchmark and not real CAPTCHA accuracy. Real development matching failed to reproduce hollow text for the outlined family. See [experiment report](STYLE_PHASE3_FONT_01.md). A second synthetic-only training iteration is underway based on the still-improving synthetic selection curve; preserve both outcomes.

## Further evidence update — 2026-09-28

Run `style_phase3_font_02` continued to step 2,000 using synthetic training/selection only. On a shared fresh 768-image synthetic test, run 01 -> run 02 changed top-1 45.70% -> 56.25%, top-3 77.47% -> 84.77%, hollow/filled 89.06% -> 90.23%. No published-paper accuracy is being substituted for these results. Real matching remains unreliable; outlined is still wrong. A frozen 12-image preview test on [pymlex/4-captcha](https://huggingface.co/datasets/pymlex/4-captcha) chose an inappropriate pixel font and failed visual resemblance. See [run 02 report](STYLE_PHASE3_FONT_02.md). This source is a dataset card, not a directly implemented research paper; images/card only, no generator code or model weights used.

## Direct implementation update — robustness iteration

DeepFont-inspired augmentation/multi-view alignment is now being tested in `style_phase3_font_03`: standard plus stronger synthetic line/noise clutter, training crops drawn from the same four view types used at inference. This addresses a local implementation mismatch; it does not adopt the paper's real-data adaptation. No claim yet that the new run improves real fonts. Training/selection remain synthetic-only.

## Completed robustness evidence — 2026-09-28

DeepFont-inspired run 03 is complete: matched harder synthetic top-1 45.96% -> 54.56%, top-3 74.09% -> 84.24%, hollow/filled 85.55% -> 89.97% compared with run 02. This supports our augmentation/view-alignment implementation on synthetic data only. All 160 real-development renders completed, but outlined lettering remains wrong. No real font accuracy or Phase 3 pass claimed. See [run 03 report](STYLE_PHASE3_FONT_03.md).

## Domain-gap diagnostic follow-up

Run style_phase3_outline_diagnostic_01 tests the local DeepFont-inspired matcher without new training: synthetic hollow controls mean probability .836, real outlined fit images .250; CV MSER topology also misses outlines. This is evidence of remaining domain mismatch, not evidence that the published DeepFont method fails. No new paper adopted in this diagnostic; original local analysis code is scripts/diagnose_font_outline.py.

## Contrast robustness extension

Original paired synthetic diagnostic (style_phase3_contrast_diagnostic_01) found severe low-contrast confusion: font55.99% ->18.23%, hollow87.50% ->41.67% at gain.15. This supports testing wider synthetic contrast augmentation, not a claim that contrast explains all real failures. Run04 uses four gain levels with independent view cycling, extending the existing DeepFont-inspired augmentation; it is not a paper reproduction. Outcome pending.

## Completed contrast extension — mixed outcome

DeepFont-inspired local augmentation extension run04 improves very-low-contrast synthetic hollow accuracy39.71%->68.75%, but normal-contrast accuracy95.31%->91.93%. Real outlined rendering still fails. This negative/partial outcome is retained in [run04 report](STYLE_PHASE3_FONT_04.md); no new paper or reproduction claim. Evaluation code: scripts/compare_font_contrast.py.

## Geometry ablation and augmentation extension

Local original diagnostic scripts/diagnose_outline_geometry.py tested1152 controlled images with fixed03/04 checkpoints. Large outer-boundary outlines were often missed (52px recall17.7%/12.5%), unlike thin inner boundaries (89.6%/78.1%). This motivates run05 geometry support expansion in synthetic training, within existing DeepFont-inspired augmentation rather than any newly adopted paper. No real accuracy claim. Run05 outcome pending.

## Geometry run05 and independent-source test

Run05 completed synthetic geometry augmentation training: synthetic test font47.92%, top3 76.95%, hollow90.10%, selectedstep3300. Test distribution differs from older runs; no direct headline comparison. All160 real-development renders completed; outlined stillbold. New-source frozen test now uses [project-sloth/captcha-images](https://huggingface.co/datasets/project-sloth/captcha-images), pinned revisioneeaf2b6ec9086645f270f7e2aaa9ea90730683de, WTFPL declaration/license saved,32 seeded images. Dataset source, not a research paper or adopted generator. Archive label-bearing filenames are ignored; no text labels enter inference. Results pending.

New-source test02 completed:32images,960blocks,48outputs, unchangedmodels. Assistant visual review failed style resemblance: spuriouscoloredlines, wrong generictextlength/alphabet, falseholloweffect. Phase2 worsened clutter. No new paper adopted; this is negative evidence for the local pipeline, not a claim about DeepFont.

## Achromatic consistency engineering

Original optional measured-color constraint in style_pipeline/constraints.py and render.py; no newpaper adopted. Paired developmentablation removes inventedRGBcolor but falseinterference persists. See STYLE_COLOR_CONSTRAINT_01.md. This is engineering evidence, not a publishedmethod reproduction or generalization result.

## Feature-classifier audit evidence

Original local audit reproduced dense-dot confusion in synthetic controls with no line/curve operations, despite exact descriptor replay. See STYLE_FEATURE_AUDIT_01.md. No newpaper used and no claim that this implements a named hard-negative-mining method. A future retraining revision must document its actual method and outcomes.

## Synthetic negative-coverage revision started

Implementation scripts/train_feature_negatives.py trains separate LR models on originalsynthetictraining plus new operation-labeled dot-only negatives, with disjoint seed partitions and ordinaryselection regression gate. Original local data-augmentation experiment; no additionalpaper adopted and not a claim of reproducing a named mining algorithm. Originalbundles preserved; results pending in runs/style_feature_negatives_01.

## Negative-coverage revision outcome

Originallocal syntheticaugmentation run01: straight modelpasses selectiongate, freshnegativeFP71.09%->.78%, ordinarytestAP.8832->.8709. Curvefails gate and isnotaccepted (ordinaryAP.8800->.8063). No newpaper adopted. See STYLE_FEATURE_NEGATIVES_01.md for protocol and limitations.

## Straight revision transfer outcome

Synthetic-only thresholdcalibration followed by frozenrealdevelopment CV+inference failed:480/480blocks stillstraightpositive, renderingunchanged. See STYLE_STRAIGHT_REVISION_EVAL_01.md. Retain negativeoutcome: localdotaugmentation improves syntheticcontrols butdoesnotdemonstrate realtransfer. Noadditionalpaper used.

## Geometric line-support negative result

OriginalCVcontinuity heuristic (style_pipeline/line_support.py), not a paperimplementation. Two syntheticdevelopment runs reject dot-onlynoise but recall68.75%/59.375% fails required90%; notenabled, no realablation. No newpaperadopted. Keep failedresults in runs/style_line_support_01 and02.
\nBackground audit: original local matched-render analysis, no new paper. 160 exact replays isolate recipe controls from font changes; mixed rollback outcome. See STYLE_BACKGROUND_REGRESSION_01.md.


Pipeline order revised by user: DeepFont-inspired adjustment is now Phase2. Prior recipe/CNN/transformer/GAN work is historical and optional Phase3, deferred pending assessment. Existing paper attribution and failed results remain unchanged.


Direct Phase1+fonts bank preview01 uses existing documented methods; no newpaper. Eight sources/32renders; backgroundoperations preserved but severalfont/layout errors persist. See STYLE_DIRECT_BANK_01.md.


Colorlocked Phase3 trial uses existingrecipe methods with original engineeringconstraints; no newpaper.32renders, colorassertionspass but qualitymixed; see STYLE_PHASE3_COLOR_LOCKED_01.md.


Phase3 clutter ablation is original local one-control-at-a-time diagnostic, no newpaper.20renders confirm linecount/width amplification; not generalization evidence. See STYLE_PHASE3_CLUTTER_ABLATION_01.md.


Per-parameter audit01: original local320renderablation, no newpaper. Reused grayscale fixed_features metric is related to previousPhase2optimization and notindependentvalidation; qualitativecandidatefindings separatedfromscores. See STYLE_PHASE3_PARAMETER_AUDIT_01.md.


Character-count pilot01 is original smallCNN syntheticclassifier, no newpaperadopted. Existingaugmentationutilities reused; not DeepFont countmethod or OCRreproduction.89.29%synthetic/93.53%withheldfont, realtransfer mixed. See STYLE_CHARACTER_COUNT_01.md.


## Count heatmaps discussion: consulted, NOT implemented

- Caron et al., Emerging Properties in Self-Supervised Vision Transformers, ICCV2021. Primary abstract/search-indexed paper summary consulted: https://arxiv.org/abs/2104.14294 . DINO attention can expose object regions; no evidence from this paper that one CAPTCHA character equals one attention peak. No weights downloaded or DINO code used.
- Darcet et al., Vision Transformers Need Registers, arXiv2309.16588. Primary abstract opened: https://arxiv.org/abs/2309.16588 . Documents high-norm/background artifacts in attention/features and register remedy. Relevant caution for heatmap interpretation, not implemented.
- Lempitsky and Zisserman, Learning To Count Objects in Images, NeurIPS2010. Primary Oxford abstract consulted: https://robots.ox.ac.uk/~vgg/publications/2010/Lempitsky10b/ . Density integrates to count; point annotations supervise localization/counting. Proposed future adaptation: synthetic glyph centers and unit-mass density targets, no real labels or character recognition. No density model implemented yet; would be an original neural adaptation, not reproduction of their convex optimization method.

User suggests DINO heatmap counting. Assessment: worth an experimental baseline, not assumed reliable or simpler computationally. Compare with simple projection/component baseline and synthetic-supervised center/density map; do not promote on appearance alone. Existing direct count model remains experimental. Automation remains paused.


## Implemented count-heatmap experiment 01 (supersedes earlier consulted-only status for DINO/density)

- Caron et al., Emerging Properties in Self-Supervised Vision Transformers, ICCV2021: https://arxiv.org/abs/2104.14294 ; official implementation https://github.com/facebookresearch/dino . Used cached Apache-2.0 ViT-S/8 natural-image pretrained backbone and `get_last_selfattention` CLS-to-patch extraction as in official `visualize_attention.py`. Reviewed official code and attention extraction. Our horizontal projection, peak counting and synthetic-only head/threshold selection are original heuristics, not a DINO counting method or paper reproduction. Exact cached commit unknown; source snapshots and SHA256 in run protocol. No CAPTCHA training or OCR. Outcome:53.87% synthetic exact, unreliable real localization/counts; not promoted.
- Lempitsky & Zisserman, Learning To Count Objects in Images, NeurIPS2010: https://www.robots.ox.ac.uk/~vgg/research/counting/NIPS2010.pdf . Read introduction and section2.1/equation1. Directly borrowed density-as-count-integral and Gaussian targets centered at point annotations. Here points come from synthetic glyph masks. Deviations: original positive CNN with map/count MSE, no linear feature regression, MESA loss, cutting-plane solver or convex QP. We renormalize boundary Gaussian mass to1 because synthetic glyphs are fully visible; paper permits boundary mass loss. Outcome38.99% exact synthetic, poor real transfer; not promoted.
- Darcet et al. remains consulted-only; no register-based model implemented.

Implementation and outcomes: runs/style_count_heatmaps_01/compare_count_heatmaps.py and docs/STYLE_COUNT_HEATMAPS_01.md. This records an unsuccessful counting baseline, not evidence that heatmap peaks reliably equal characters.


All-family count+font integration01 reuses previously attributed feature/font methods and original countCNN. No newpaper or DINO integration. See STYLE_COUNT_FONT_BANK_01.md for mixed results; renderer integrity is not real length accuracy.


## DINO renderer-parameter experiment01: implemented use

Caron et al., Emerging Properties in Self-Supervised Vision Transformers, ICCV2021, https://arxiv.org/abs/2104.14294 ; official source https://github.com/facebookresearch/dino/blob/main/vision_transformer.py . Used cached Apache-2.0 ViT-S/8 `get_intermediate_layers` and `get_last_selfattention`; code reviewed, exact cached source/weights hashed, upstream commitunknown. Borrowed pretrained patch/CLS representations and attention extraction. Original additions: mean-head attention pooling,3x10 occupancy, weighted family-mean discrepancy, finite noncolor renderer search and CLS-only ablation. These are NOT paper-reproduced counting/distribution adaptation algorithms; no DINO fine-tuning, GAN, MESA, or register method.30development outputs: small same-objective gains, CVmixed4/8, attention changes only1/8 selections; visual mismatch persists. Not promoted. Exact implementation runs/style_dino_recipe_01/run_dino_recipe.py; limits/outcomes docs/STYLE_DINO_RECIPE_01.md.


Phase1+Phase2 release01 is packaging and data-integrity engineering, not a new research method. Existing feature/font methods copied unchanged with source hashes; no new paper adopted. Generic21cell balancing and stable text-hash split are original experiment controls. No PARSeqpaper reproduction/training claimed; next-session guide explicitly separates proposal from executed work. See releases/phase12_v1/VALIDATION.md.


Background-block correction01 is original CV engineering: connected quantized-color rectangular components, fill/text-overlap/boundary checks. No newpaper adopted and no trained rectangleclassifier claimed. See STYLE_BACKGROUND_BLOCKS_FIX_01.md for validation and false-negative limitations.


V2Tier1initial pilots01/02 use original procedural mask supervision, sharedCNN cropencoder,position-aware tanhattentionpooling and BCE+dice multitasktraining. No newpaper directly adopted and no transformer/MILpaper reproduction claimed. Full/visible support masks and failureoutcomes documented in V2_TIER1_INITIAL_RESULTS.md. These are pilotmodels, not a robust upgrade or source of thesis success claims.


V2Tier1 revision04: original engineering experiment with shared CNN, per-grid MLP classifiers, synthetic crop presence supervision and hierarchical learned pooling. No new external paper directly used or reproduction claimed. Inherits prior03 native-resolution decoder. Exact design, controls and results in V2_TIER1_GRID_SPECIALISTS_04.md and runs/style_v2_tier1_04; findings limited to one procedural generator/training seed.


Curve debugging05: no new external paper used. Original engineering diagnostics: single-curve centerline residual proxy, frozen head-score/pooling sensitivity probes, matched auxiliary-loss ablations. Document evidence and limits in V2_CURVE_DEBUG_05.md; do not attribute these original checks to a paper or claim causal proof from attention.


V2Tier1 06: original engineering implementation of local PCA-based axial geometry supervision and neighboring-grid message aggregation from predicted geometry. No external paper directly adopted or reproduction claimed. Exact inputs, connectivity-proxy limitations and matched-control outcomes in V2_STROKE_RELATIONS_06.md and runs/style_v2_tier1_06. Do not cite generic graph/transformer papers as directly used.


Independent curve-specialist reset: original paired synthetic-corpus design, deterministic local-contrast pop candidate, spatialCNN, per-grid/raw-pop search and grouped validation. No new external paper directly used or reproduction claimed. Uses standard PyTorch components, including GroupNorm; exact code/source hashes and limitations in feature_specialists_v2 and V2_FEATURE_SPECIALIST_PLAN.md. Results pending.


Independent dot specialist: original synthetic paired data and standard OpenCV top-hat/black-hat morphological operations as a candidate pop. No new external paper directly consulted or reproduction claimed. Source and limitations V2_DOT_SPECIALIST_PLAN.md; training/evaluation pending.


Multi-label v1: no new primary paper directly consulted or reproduced. Standard shared CNN/BCE supervision/attention pooling used as original baseline; exact architecture and limits in docs/V2_MULTILABEL_PLAN.md and feature_multilabel_v1/train.py. Results pending; no fabricated paper attribution.


## Vision Transformer architectural reference (2026-09-28)
Dosovitskiy et al., An Image is Worth16x16Words: Transformers for Image Recognition at Scale (ICLR2021), https://arxiv.org/abs/2010.11929. Primary abstract consulted. Borrowed high-level idea: image-patch sequences processed by a pure transformer. Implementation: feature_transformer_multilabel_v1/train.py Model. Original small adaptation:8x8patches,96dimensions,3encoder layers,4heads,15feature tokens and binary heads instead of a single classification token. No pretrained weights or large-scale pretraining; no claimed paper reproduction or equivalent performance. Training/evaluation pending; results recorded in V2_TRANSFORMER_MULTILABEL_RESULTS.md when complete.


# Grid and bidirectional hybrid research â€” 2026-09-28

Status: researched, not implemented or trained. Current models and production remain unchanged.

## Gridded Transformer Neural Processes

Ashman et al., ICML2025, https://proceedings.mlr.press/v267/ashman25a.html . Read final publication abstract and preprint sections4.1â€“4.3 at https://arxiv.org/html/2410.06731v2 . Distinguish preprint from final publication: final PDF retrieval failed. Official MIT repository: https://github.com/cambridge-mlg/gridded-tnp .

The preprint encodes observations into spatial pseudo-tokens using neighbourhood cross-attention, processes them with ViT/Swin, and decodes predictions at queried locations. Its task is conditional regression, not CAPTCHA multi-label classification. Potential borrowed idea: position-aware aggregation into a compact latent grid. No temporal or weather machinery is needed here. Our dense images already have a grid, so handling irregular observations offers limited immediate benefit. Equivariance/uncertainty claims do not automatically transfer to a modified classifier. No implementation copied or experiment run.

## Visual Conformer

Peng et al., ICCV2021, https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Conformer_Local_Features_Coupling_Global_Representations_for_Visual_Recognition_ICCV_2021_paper.html . Read abstract, indexed paper discussion and official README/source: https://github.com/pengzhiliang/Conformer ; Apache2.0. This is visual Conformer, not the speech architecture with the same name.

Concurrent CNN and transformer branches exchange representations through feature-coupling units. This is the closer match to the user's proposed local-to-global and global-to-local feedback. Candidate adaptation: small CNN preserving fine detail, a coordinate-aware token grid, and repeated feature exchange before15classification heads. This proposed compact variant would not be a faithful paper reproduction. Implementation and performance are pending.

## Proposed discriminating experiment

Keep data and feature definitions fixed initially. Compare CNN alone, one-way CNN-to-transformer, and bidirectional coupling; optionally add learned grid pooling as a separate ablation. Match capacity/compute approximately and disclose differences. Use fresh synthetic tests, repeated seeds, per-feature gates and real-development comparisons after freeze. Specifically inspect block/grid confusions, fine strokes and colour/texture shifts. Label quality and generator coverage remain alternative failure causes. Renderer-parameter regression remains a separate unimplemented hypothesis. No new training launched by this research review.


Joint mapping01 now IMPLEMENTS original adaptations of the researched ideas (previously research-only): Ashman et al.ICML2025 gridded pseudo-token local encoder/transformer processor, https://proceedings.mlr.press/v267/ashman25a.html ; Peng et al.ICCV2021 visualConformer bidirectional feature exchange, https://github.com/pengzhiliang/Conformer . Exact code: feature_joint_mapping_01/models.py; joint supervision controls.py/train.py; renderer adapter generate.py. No source copied, no temporal/probabilistic NP decoder, no paper-scale reproduction.9parameter heads and15binary heads are our adaptation. Tests verified targets and both gradient paths; training outcomes pending and will be saved separately for each model.


## Meta-Sim — directly used in CAPRI optional pilot

Kar et al., ICCV2019, https://arxiv.org/abs/1904.11621 . Read primary preprint sections3.1–3.2.3. Borrowed mutable attribute transformation, identity start, Gaussian featureMMD and finite-difference renderer gradients. Implementation runs/capri_metasim_01/trial_snapshot.py; full protocol docs/CAPRI_METASIM_PLAN.md. Deviations: residualMLP rather than GCN, frozenDINO rather than Inception, zero-init rather than autoencoder pretraining, no downstream task loss or REINFORCE. Not full reproduction. Separate adapter per family, real unlabeled fit allowed only for this optional stage. Results including negative independentCV outcomes: docs/CAPRI_METASIM_RESULTS.md. No novelty attribution for these borrowed ideas.

DINO (Caron et al., ICCV2021, https://arxiv.org/abs/2104.14294 ): reused cached official ViT-S/8 frozenCLS as representation only, no attention or VLM. Weight hash in protocol; source/LICENSE snapshot in run. It is not a validated CAPTCHA similarity measure.
