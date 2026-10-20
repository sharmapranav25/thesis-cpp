# Character length without reading the text

Research decision, 28 September 2026. No new count model has been trained or declared accurate. The existing count model is not promoted. This note uses primary papers, historical experimental evidence and explicitly labelled proposals. Real character strings and the 200 DarkBay recognition-evaluation labels must not select the new length policy.

## What the current results imply

The user reports epoch-8 synthetic CER3.47% versus real DarkBay CER82.75%; on118 correct-length real predictions CER remains75.99%. These are user-reported recogniser measurements, not rerun here. Length is useful because it changes layout and training allocation, but this evidence does not establish length as the dominant failure. The old100k corpus used Anton61.990%, Pixelify24.667%, Oswald13.343%; that narrow, skewed typography policy is a separate plausible bottleneck.

Historical count trials already tested direct classification, foreground projections, DINO attention peaks and density regression. Fresh synthetic/withheld-font exact scores were88.10/91.37%,50.89/48.81%,53.87/53.27%,38.99/48.81% respectively. None established accurate real counting. The synthetic generator's count-dependent width clamp was a shortcut risk. Repeating the same training at greater scale is insufficient evidence of progress.

## Primary research and exact relevance

### 1. CRAFT — character regions and affinity

Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee. *Character Region Awareness for Text Detection*, CVPR2019. [Paper](https://arxiv.org/abs/1904.01941), [full text](https://arxiv.org/html/1904.01941v1).

Read the method description of character-region scores and adjacent-character affinity. Useful idea: supervise character locations rather than expect generic attention to discover one instance per glyph. Our renderer can supply individual synthetic glyph geometry without real transcription. Caveat: CRAFT detects text regions; counting connected components in its final grouped word boxes is not character counting. Its training also uses real-image supervision/pseudo-label procedures that we are not adopting. CAPTCHA clutter is outside the paper's guarantee. Proposed local adaptation only; no CRAFT weights or code downloaded in this task.

### 2. DETR — predict a set of character instances

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko. *End-to-End Object Detection with Transformers*, ECCV2020. [Paper](https://arxiv.org/abs/2005.12872), [full text](https://arxiv.org/html/2005.12872v3).

Read set prediction, object queries, matching and no-object classification. Proposed use: a small CNN plus transformer with12queries predicting character/not-character and a box or centre, trained with bipartite matching against synthetic instances. Count accepted instances; no alphabet head. Matching discourages duplicate assignments and explicit locations expose counting mistakes. Caveats: small touching glyphs, no-object calibration and missed low-contrast instances remain difficult. Original DETR's natural-image results do not demonstrate CAPTCHA counting accuracy. This is a candidate experiment, not an implemented or reproduced model.

### 3. CTC — possible count-only sequence baseline

Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber. *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*, ICML2006. [Primary PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

Read path collapse, blank symbols and forward-backward alignment. A count-only adaptation could label a string of n identical generic character tokens. Repeated tokens require blank-separated paths to remain distinct; summing nonblank frame probabilities or counting arbitrary peaks is not a valid substitute. This avoids real transcription but may still struggle with touching glyphs and sufficient temporal resolution. Secondary baseline, not the first implementation; no claim that the original paper tested count-only CAPTCHA targets.

### 4. Density counting and DINO are retained as negative baselines

Lempitsky and Zisserman, *Learning to Count Objects in Images*, NeurIPS2010: [paper](https://www.robots.ox.ac.uk/~vgg/research/counting/NIPS2010.pdf). Caron etal., *Emerging Properties in Self-Supervised Vision Transformers*, ICCV2021: [paper](https://arxiv.org/abs/2104.14294). Their broader counting/representation ideas do not override our failed local trials. We will not describe CLS attention peaks as verified character instances. Density regression may be retried with better supervision, but is lower priority than explicit instances plus a count head.

## Recommended experiment

Compare two models under the same synthetic splits, budgets and three seeds:

1. **Repaired direct count baseline:** CNN with whole-image context, outputs a count distribution; optional character-region auxiliary target. This tests whether fixing the training distribution is enough.
2. **Character-instance model:** CNN plus small set-prediction transformer,12queries, class-agnostic character boxes/centres and a global count head. Loss combines matched objectness/location losses and count cross-entropy. Disagreement between object count and global count triggers uncertainty, not forced consensus.

Exact architecture, loss weights and schedules must be frozen in a training protocol before execution; none has been selected by target evaluation accuracy. An instance model is recommended because its errors are inspectable, not because its accuracy is already established. Avoid adding CTC, density and every architecture simultaneously before these two baselines discriminate the hypothesis.

## The training-data repair is essential

- Use a separate count corpus with broad fonts and mixed-font examples, varied backgrounds, colours, clutter, contrast and character geometry. Keep training synthetic-only.
- Balance lengths2–8, especially2–4; include separate out-of-range/blank diagnostics. A closed2–8 classifier otherwise cannot admit a nine-character image. Define an overflow/unsupported policy before inference.
- Randomise canvas width, character width, occupied band width and spacing across lengths. Do not reuse the current generator's count-dependent width clamp as the only layout generator. Construct overlapping feasible layout ranges and audit the joint distributions after rejection sampling.
- Include paired nuisances at fixed count and different counts at comparable occupied width. Otherwise “band width” can replace counting.
- Save one box/centre/mask per synthetic glyph, including visibility, even for hollow, rotated and disconnected glyphs. A glyph with several connected components remains one instance.
- Split font families, parent recipes, rendering implementations and layout mechanisms before generation. Keep entire paired groups together. Hold out crowded/short-band and mixed-font regimes.
- Measure performance separately by count, font, contrast, interference, overlap and band occupancy. Report comparison against constant-count and occupied-width-only baselines.

The new Phase2 generator supplies per-glyph boxes and provenance, but is **not yet the repaired count-training generator**: it still fits text into count-dependent cells. This limitation is explicit so the next session cannot accidentally repeat the old shortcut experiment.

## Real inference and what “accurate” would require

Infer only on explicitly declared style-fit images, with filenames/text labels excluded from model input. Preserve reference hashes and pool membership. Use a calibrated posterior p(length|image), consistency under count-preserving transforms, and an uncertainty flag. Do not convert synthetic confidence into a claim of real-domain calibration.

Aggregate soft posteriors across a family to estimate its length distribution. Resampling images gives sampling uncertainty, not protection against systematic model bias. All uncertain cases must remain represented through a broad fallback distribution; dropping them can bias the family histogram. Mixed-length families must not be forced into their modal length.

**Proposed gates, not achieved results:** compare against the old classifier and simple baselines on locked synthetic shifts; require strong short-length performance rather than only an overall score. For a claim of real accuracy, use an independently audited count-only evaluation subset disjoint from style fitting and the repeatedly inspected200-image recognition test. Count annotations are evaluation-only, never classifier fitting or threshold tuning. Report exact-count accuracy, MAE, per-length recall, accepted accuracy versus coverage, and family-histogram total-variation error. A useful target is at least95% accepted count accuracy with at least80% coverage, with sample-size-aware confidence intervals; these are proposed engineering goals, not paper-derived guarantees.

Without count-only reference annotations, report predictions and consistency—not accuracy. Existing DarkBay predictions or the near-success string do not justify setting every generated image to six characters.

## Integration boundary

Length belongs in Phase1 as a distribution plus uncertainty once validated. Phase2 consumes that distribution when placing and varying characters. Until then, generation retains explicit lengths2–8 or an explicitly supplied configuration; it does not silently infer six from evaluation labels. No runtimeVLM, real-label fitting or count retraining is introduced by this research note.
