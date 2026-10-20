# CAPRI: From Local Feature Classification to Distribution-Guided Procedural CAPTCHA Generation

**Detailed thesis chapter draft — evidence cut-off: 28 September 2026.**

This chapter reconstructs the progression from the original three-feature, 10 × 3 block classifier to CAPRI and its Meta-Sim-inspired refinement. It includes unsuccessful experiments rather than presenting the final system as the inevitable outcome of a clean sequence. Numerical results are project measurements, not results borrowed from the cited papers. Historical documents sometimes retain superseded status statements; later run records take precedence. The accompanying evidence index identifies the local sources. This is an experimental-methodology chapter, not a claim of demonstrated downstream recognition success.

## 1. Research problem and scope

### 1.1 Why a generic generator was insufficient

The starting point was a practical mismatch between generating plausible CAPTCHA images and generating training images representative of a particular CAPTCHA family. An imported generator could draw characters, backgrounds and interference using predefined choices. Configuring its alphabet or text length did not make it infer the fonts, spatial organisation, colour distribution or clutter of an unfamiliar reference collection. A working generator therefore did not by itself solve the distribution-matching problem.

Earlier work using a generator closely associated with a target family offered a potentially misleading baseline. High similarity between that generator and the real family could make recognition transfer easier without establishing a generic adaptation method. This motivation should not be confused with a measured comparison in the present chapter: no newly verified PARSeq result is available here demonstrating that the proposed method beats that earlier setup.

The revised objective was to inspect untranscribed reference images, infer which available rendering operations could approximate their appearance, and generate new images with known synthetic text. The method must work within a bank of general-purpose operations and independently sourced fonts. It must not obtain the target family's generator, copy its font assets, or use its text labels to train the style classifiers.

The resulting system is called **CAPRI: CAPTCHA Analysis and Procedural Rendering Inference**. It is best described as constrained inverse graphics followed by procedural generation. It is not an unrestricted image generator, a transcription system, or a mechanism that discovers arbitrary source code. Its representational ceiling is the set of operations already implemented in its renderer.

### 1.2 Formalising the task

Let X be a collection of untranscribed real reference images. Let R(t, f, θ, s) be a procedural renderer with chosen text t, font f, rendering controls θ and random seed s. The initial objective is to estimate a distribution of plausible recipes from X, rather than reconstruct each observed image pixel for pixel:

```
X → {CV measurements, frozen learned feature estimates}
  → family recipe profile → font selection/adjustment
  → R(new text, font, recipe, seed) → labelled synthetic corpus
```

An optional adapter subsequently modifies a restricted subset of recipe parameters using an unlabelled real fitting subset. In the final tested refinement, it does not directly paint over the output image.

This formulation has an identifiability problem. A dark stroke could be a character edge, an interference line, a border or overlapping primitives. Different recipes can produce similar pixels. Requested primitive count also differs from visible count after overlap and occlusion. Consequently, even a correct pixel interpretation need not uniquely recover the original generator settings. Learning to reproduce a useful distribution is a more defensible objective than claiming to identify the original source recipe.

### 1.3 Questions investigated

The experiments address six connected questions:

1. Can classifiers trained only on procedural examples recognise useful features in real CAPTCHA images?
2. Does dividing images into blocks, or supplying multiple grids, improve feature recognition and coherent global interpretation?
3. Can local CNN representations and global transformer reasoning jointly estimate both feature presence and usable rendering controls?
4. Can independently sourced fonts approximate target typography without obtaining the target font?
5. Can distribution-based refinement improve generated appearance without changing text identity or creating artefacts?
6. Can the resulting corpus improve real-image character recognition?

The first five received experimental investigation, with mixed outcomes. The sixth remains a downstream hypothesis. Generating and auditing labelled files is not a substitute for training and evaluating PARSeq.

### 1.4 What “no VLM” means in this work

No vision-language model is used by the implemented feature inference, font inference, generation or refinement algorithms. Assistant visual inspection was used during development to review galleries and identify obvious mismatches. The research process therefore includes VLM-assisted qualitative feedback, even though the deployed computational pipeline does not. This distinction must remain explicit.

The later DINO experiments use a pretrained visual transformer, not a VLM. Their presence also means that a blanket claim of “no pretrained model anywhere” would be false. Feature and font classifiers were trained on synthetic data; DINO representations were imported for selected experiments. Optional distribution adapters were fitted to unlabelled real images. Thus, “the classifiers never train on real data” is accurate within the stated boundary, while “nothing in CAPRI ever learns from real data” is not.

## 2. Research lineage and implementation attribution

The project combines established ideas rather than establishing novelty by giving an assembly of components a new name. Table 1 distinguishes direct adaptation, background motivation and work considered but not reproduced.

| Source | Idea used or investigated | What this project did not reproduce |
|---|---|---|
| DeepFont [R1] | Synthetic font-labelled examples, augmentation and multiple image views | AdobeVFR, the complete DeepFont network, its adaptation procedure or its published accuracy |
| SimGAN [R2] | Synthetic-to-real adversarial refinement and restrictions on image changes | Its full architecture, history buffer, training regime or benchmarks |
| Deep CORAL [R3] | Motivation for distribution-statistic alignment | Full covariance alignment; early project losses used means and diagonal standard deviations |
| Vision Transformer [R4] | Patch-token attention for image understanding | Large-scale ViT pretraining or a matched reproduction of ViT results |
| Gridded Transformer Neural Processes [R5] | Efficient organisation of local observations into a global grid | A neural-process probabilistic decoder, the paper's full model or its theoretical properties |
| Conformer [R6] | Communication between local convolutional and global transformer representations | The original network configuration or a reproduction on its benchmarks |
| DINO [R7] | Frozen pretrained visual features; attention-map experiments | CAPTCHA-specific self-supervised DINO training |
| Learning to Count Objects in Images [R8] | Unit-mass density supervision as an alternative to classification | The original MESA formulation and optimisation procedure |
| Vision Transformers Need Registers [R9] | Caution when interpreting attention and feature artefacts | Register-token retraining or a register-equipped replacement model |
| Meta-Sim [R10] | Learning simulator-parameter changes through distribution discrepancy and numerical renderer derivatives | The complete graph-based method, task objective or published benchmarks |
| Learning to Simulate [R11] | Related work on adapting simulation for downstream learning | Its reinforcement-learning optimisation |
| Yet Another Text CAPTCHA Solver [R12] | Prior art demonstrating that GAN-related CAPTCHA synthesis/adaptation is not a new category | Its target-specific training setup or reported recognition performance |
| PARSeq [R13] | Planned downstream recogniser for evaluating corpus utility | Completed recogniser training in the experiments reported here |
| Stroke Width Transform [R14] | Motivation for stroke consistency as a visual cue | Ray-based SWT; project code used MSER and simpler stroke proxies |

The word “transformer” requires particular care. An attention-based vision transformer is a neural architecture. Meta-Sim's distribution transformation is an adaptation concept and does not imply the use of that architecture. The CAPRI refinement adapter is a small multilayer perceptron. Its frozen DINO feature extractor is a transformer, but the adapter itself is not.

The defensible prospective contribution is the evaluated combination of constrained feature inference, procedural mapping, typography approximation and distribution-guided adaptation under restricted target access. Establishing originality and usefulness still requires stronger baselines, genuinely independent evaluation and downstream recognition measurements.

## 3. Data, boundaries and evaluation semantics

### 3.1 Real reference collections

The main development bank contains CryptBB, City, Courier, DarkBay and Outlined Text CAPTCHA. Their file counts changed across experimental scopes. The following distinction resolves an otherwise misleading apparent discrepancy in the records.

| Family | Earlier available style pool | Later full-directory feature inference |
|---|---:|---:|
| CryptBB | 600 | 600 |
| City | 2,000 | 2,000 |
| Courier | 2,510 | 2,510 |
| DarkBay | 799 | 999, including 200 evaluation images |
| Outlined Text CAPTCHA | 800 | 800 |
| Total | 6,709 files before experiment-specific exclusions | 6,909 files |

These are inventory counts, not a universal count of unique, readable or eligible images. An early annotation-selection record reports 6,708 historical unique images. Decode checks, deduplication and per-run selection determine the actual denominator. The additional 200 images in the later 6,909-file run came from DarkBay's evaluation directory, not an enlargement of the outlined collection.

The full-directory dot-scatter run explicitly included those DarkBay evaluation images at the user's request. That did not train the classifier on real text labels, but it exposed previously reserved visual distributions to development review. Later comparisons on those images must therefore not be described as completely untouched evaluations. Preserving a folder named `eval` does not preserve experimental independence once its contents influence decisions.

Additional independent-source probes included Roncuevas, Pymlex and Sloth collections. They were small convenience probes, not a statistically random sample of all CAPTCHA schemes. The Pymlex family-specific refinement report contained only one held-out example after a 2/1/1 split. Such a sample can reveal a failure but cannot estimate reliable generalisation performance.

### 3.2 Synthetic-only training is necessary but insufficient

The style classifiers receive synthetic pixels and synthetic supervision. Target labels, generator identifiers, masks and hidden recipe values must not enter their image input. However, this restriction alone does not eliminate leakage or shortcuts. Synthetic variants can share text, backgrounds, rendering seeds or crop pixels. Training and testing on different crops from the same base image can yield optimistic results even when no real image is used.

The early corpus therefore retained base-group membership and used hash checks. Subsequent trials introduced font, combination and recipe-shift tests. These boundaries improve the experiments, but they do not create a fully independent real benchmark. Repeated inspection of the same five families gradually turns them into development data, even when each new model is frozen before its next inference run.

### 3.3 Metrics that must not be conflated

Average precision measures score ranking. F1 measures a thresholded balance of precision and recall. False-positive rate measures the fraction of negative cases called positive. Calibration coverage describes how often an abstaining classifier makes a confident decision. None is interchangeable with the others.

Likewise, the fraction of real images predicted to contain dots is **prediction prevalence**, not detection accuracy and not dot density. Density is a spatial or count measurement. Accuracy requires trustworthy reference labels. A model can consistently predict a spurious grid throughout a family and produce a stable, wrong prevalence estimate.

Descriptor distance and DINO-space discrepancy measure similarity in their chosen representation. They are neither human perceptual ground truth nor recognition accuracy. A model may reduce a distance by spreading colour around text, while making the image less representative or less readable. Several experiments below exhibit precisely that behaviour.

### 3.4 Historical gate overrides

The original plan required an audited real-image feature review before supervised-model progression. That audit remained incomplete. The user explicitly authorised synthetic-only exploratory training while real evaluation remained pending. Later exploratory refinement also proceeded despite an unresolved Phase 1 gate. These were scope decisions, not retrospective gate passes.

Acceptance thresholds were engineering choices in the experiment protocols, not universal thresholds established by the cited literature. Failed gates are retained as failures. Some later components were packaged for experimentation despite incomplete gates; packaging is an operational milestone, not scientific certification.

## 4. V1: building the feature bank before training

### 4.1 Resource acquisition and adapter pilots

The initial bank investigated Gregwar/Captcha, Securimage, MobiCMS, simple-php-captcha and an existing Python renderer. Each backend rendered 64 initial cases, giving 320 pilot images. These tests examined reproducibility and available operations before attempting to learn their appearance.

| Backend | Successful initial renders | Exact seed replays | Main qualification |
|---|---:|---:|---|
| Gregwar | 64/64 | 64/64 | Edge cases approached text boundaries; warping could also bend interference |
| Securimage | 64/64 | 64/64 | Enabling noise changed text placement, invalidating naive counterfactual pairing |
| MobiCMS | 64/64 | 0/64 | Native random draws were not controlled by the initial seed mechanism |
| simple-php-captcha | 64/64 | 64/64 | Text/shadow clipping concerns; unresolved bundled texture rights |
| Python | 64/64 | 64/64 | Some requested marks were faint; requested operations were not automatically visible labels |

A separate MobiCMS capture-and-replay repair recorded native random draws and checked their bounds and order. It achieved 64/64 exact replays for that instrumented variant. This fixed determinism, not every outstanding corpus-quality issue.

The original repository and historical experiments were preserved. Hash checks covered 164 downloaded source files and 50 original/vendor paths. Unresolved decorative fonts and texture assets were excluded. Securimage's package metadata and licence text required care because their descriptions did not match exactly. A code repository's licence also does not automatically establish the licence of every bundled image or font.

The early catalogue contained more proposed features than validated capabilities. Straight strokes, curves and dots were practical starting points. Hollow lettering, decorative shapes, textures and several typography operations initially lacked a verified eligible mapping. Returning a capability gap was more honest than claiming that every feature could already be recreated.

### 4.2 The 10 × 3 representation

The user's central hypothesis was that different portions of a CAPTCHA collectively reveal its style. The initial view divided an image into **10 columns and 3 rows**, producing 30 overlapping native-pixel bricks. This notation counts cells, not ten vertical cuts and three horizontal cuts. Local crops preserve original pixel scale before descriptor processing.

The appeal is straightforward: one block may expose a line, another a colour patch and another a curved segment. Their combined evidence may be more informative than a single whole-image summary. The immediate difficulty is equally important: small blocks discard context. A curved line is nearly straight over a short interval; a dot can resemble an antialiased character fragment; a coloured rectangle may be part of a glyph.

The initial proposal should therefore be understood as a representation hypothesis, not a guarantee that smaller crops improve recognition. Later experiments repeatedly returned to this trade-off.

The initial overlap extended each brick by 12.5% of its width and height on available sides, rounded up and clipped to the image boundary. Adjacent crops consequently shared approximately 25% overlap. Geometry tests verified exact native-pixel extraction, coverage and reassembly; they did not verify feature recognition. Shared pixels establish adjacency, not that two observations belong to the same physical stroke.

### 4.3 Causal supervision and the first rejected shortcuts

A labelled-brick demonstration used eight independent base configurations and all eight combinations of three effects: straight lines, sine curves and dots. This produced 64 parent images and 1,920 crop occurrences. The effects were instrumented Python/Pillow overlays, not a demonstration of generalisation across the original PHP backends.

The renderer recorded geometry and colours once. Leave-one-effect-out images reused those choices without resampling randomness. A pixel counted as a visible contribution when removing the effect changed an RGB channel by at least 20. This separated a requested operation from an operation that survived in the final image.

The first pilot assigned effect-specific colours and coupled font choice with background polarity. Those shortcuts were rejected before training. A revision sampled all effects from a common colour distribution and used both polarities for each font. Another revision counted visibly distinct connected dots rather than accepting repeated or overlapping draws as multiple marks.

The local label policy also included uncertainty. A crop with insufficient visible context was not forced into the negative class. Curves required local deviation from a straight axis; dots required distinct marks; hidden or faint operations remained uncertain. Later repairs stopped curve fragments borrowing curvature from invisible recorded geometry and required background margin around candidate dots touching text.

These decisions were scientifically useful because synthetic labels can be wrong in a semantic sense even when the generator knows exactly what it drew. A latent “curve enabled” bit does not prove that a human or classifier could identify curvature in every crop.

### 4.4 Expanded candidate corpus

The expanded candidate comprised 5,600 synthetic parents and 168,000 crop occurrences. Quality filtering and exact deduplication left 119,292 unique crops. The initial 1,920-occurrence pilot itself contained only 1,139 unique pixel crops. Neither figure should be presented as an equivalent number of independent observations.

The expanded corpus varied image size, typography, text proportions, colour, polarity and effect strength. Its audits checked parent/crop hashes, base-group boundaries, exclusions and deterministic replay. A perceptual duplicate screen produced 483 candidate cross-partition pairs; none passed its subsequent RGB difference criterion. That is evidence about a particular screening procedure, not proof that all perceptual dependence was removed.

Despite the larger corpus, the original acceptance gate remained partial. The real annotation audit, broader renderer coverage and several local-semantic questions were unresolved. Training proceeded only under an explicitly exploratory scope.

## 5. V1 classifiers, feature pop and real-image failure

### 5.1 What actually trained

The first completed training run took 1,716.9 seconds, approximately 28.6 minutes including descriptor extraction for 84,907 crops. It trained both Logistic Regression and Extra Trees candidates separately for straight interference, curved interference and dot scatter. Each feature used a seeded sample of 6,000 positive and 6,000 negative training crops. Unknown and uncertain labels were omitted per feature.

The representation combined native colour, contrast and connectivity statistics with orientation-pooled HOG on a proportionally padded 256 × 96 canvas. HOG blocks touching padding were excluded. The descriptor schema records 53 base values and 135 total values for the enhanced representation. Inputs were derived from pixels, not generator masks or recipe metadata.

HOG's established descriptor lineage is Dalal and Triggs [R15]. The tree comparison uses the Extra Trees family introduced by Geurts, Ernst and Wehenkel [R16], through scikit-learn. These are method attributions, not claims that their original benchmark experiments were repeated.

Logistic Regression tested C values 0.1, 1 and 10, with training-only standardisation, balanced weights and the liblinear solver. Extra Trees used 256 trees, depths 16 or 32, and minimum leaf sizes 2 or 5. Model and representation selection used synthetic selection AP. Sigmoid calibration and decision thresholds used a separate calibration partition. Logistic Regression was preferred when within 0.01 selection AP of the best tree candidate, and was selected for all three features.

### 5.2 What feature pop did—and did not establish

The enhanced descriptor package added CLAHE, morphological, threshold and feature-pop measurements. Its best selection AP exceeded the ordinary representation:

| Feature | Ordinary LR selection AP | Enhanced LR selection AP |
|---|---:|---:|
| Straight interference | 0.8241 | 0.8755 |
| Curved interference | 0.8523 | 0.9010 |
| Dot scatter | 0.8807 | 0.9514 |

This supports the enhanced package on that synthetic selection distribution. It does not isolate the contribution of one normalised image because several descriptor additions changed together. Feature pop also enhances character edges and background patterns. It is not a clean segmentation mask showing only the desired feature.

| Selected model | Reserved synthetic test AP | F1 at 0.5 | Confident abstaining-decision coverage |
|---|---:|---:|---:|
| Straight LR | 0.8832 | 0.7974 | 94.2% |
| Curve LR | 0.8800 | 0.7657 | 0% |
| Dot LR | 0.9455 | 0.8729 | 0% |

The curve and dot models ranked examples reasonably well but had no accepted separated thresholds under the implemented calibration rule. They therefore could not be described as deployable confident classifiers. The threshold procedure independently selected broad positive and negative sets, then rejected overlapping cutoffs; its failure did not prove that no alternative feasible threshold pair existed. Calibration precision was also estimated on data used to fit the sigmoid, limiting its independence.

Macro F1 at the fixed cutoff was about 0.812, below the planned 0.85 exploratory target. A withheld combination contained no positive curve examples, so its curve AP was not a meaningful generalisation measurement. These qualifications are more informative than reporting only the best dot AP.

### 5.3 Frozen real inference exposed domain shift

A later frozen five-family run processed 840 images, using 128 fit and 40 historical report images per family, and generated 25,200 blocks and 75,600 feature predictions. Straight interference was predicted at image level in every one of the 168 images of every family. Curves were similarly saturated: 168, 166, 168, 158 and 168 positives across CryptBB, City, Courier, DarkBay and Outlined respectively. Dot positives were 110, 27, 9, 56 and 0.

These counts showed that inference executed; they did not establish correctness. If the aggregation says “present if any block is positive,” even moderate crop false-positive rates can become severe image false-positive rates. Under a deliberately simplified independence assumption, 30 blocks with a 5% false-positive probability would yield 1 − 0.95³⁰, approximately 78.5%, image-level false positives. Real blocks are correlated, so this is an explanatory calculation rather than an empirical estimate.

Adding synthetic hard negatives reduced one straight-line false-positive diagnostic from 71.09% to 0.78%, while ordinary test AP decreased slightly from 0.8832 to 0.8709. The analogous curve revision reduced AP from 0.8800 to 0.8063 and failed its criterion. A real-block audit still produced 480/480 straight positives. The central lesson was that success on a repaired synthetic negative distribution did not demonstrate transfer to real backgrounds and typography.

### 5.4 CV measurements and the first end-to-end gate

The procedural mapping grew beyond the three classifiers. It measured palettes, gradients, edge structure, orientations, text-region proposals, tilt proxies, density and relationships between blocks. MSER and connected components proposed text-like regions; distance transforms and contour holes supplied limited stroke/topology cues. These regions were not verified characters, and the method did not implement full SWT [R14].

Block adjacency and global line linking attempted to combine local observations into a coherent family profile. This was explicit CV logic, not a learned scene graph. Measurements were mapped to general rendering controls; they did not identify the original CAPTCHA library.

The first end-to-end Phase 1 pilot improved the declared descriptor distance by at least 20% for all five families. Nevertheless, generated family centroids were nearest their own real family for only 3/5, below the required 4/5. Assistant review judged only 3/5 partial broad matches. Courier showed incorrect blocks/pixel-like fonts, and the outlined family remained filled rather than hollow. The overall gate failed.

The result already foreshadowed the rest of the study: family differentiation and lower summary distance are useful, but insufficient. A generator can become quantitatively less generic while still reproducing the wrong visual mechanisms.

## 6. Rendering regressions, font adjustment and rejected refinement

### 6.1 Why unrelated families acquired similar backgrounds

A conspicuous regression was the appearance of block-like backgrounds in Courier, where they did not represent the observed family. The immediate explanation was not that a GAN had transferred CryptBB's weights into Courier. The baseline renderer already introduced blocks because a CV rule interpreted colour variation too readily as rectangular background structure. A shared renderer can therefore spread the same incorrect visual vocabulary across families without sharing a fitted refinement model.

The repair required positive geometric evidence: median filtering, RGB quantisation, high rectangular fill, limited text overlap, minimum size, sharp sides and multiple supported regions. In an audit of 596 profiles, Courier block support changed from 64/96 to zero. CryptBB support fell from 91 to 17, DarkBay from 87 to 74, while City and Outlined changed from zero to one and two respectively. This is evidence that the rule became more selective; there was no independent labelled recall study proving that it retained every legitimate block.

A separate 160-replay background audit found that changing the font stage did not itself change the recorded background operations. Recipe offsets were a more direct cause of clutter regressions. Example multipliers included approximately 4.08 for Courier line count and 4.03 for City line width. DarkBay and Outlined had zero offsets in that particular comparison, so their mismatch predated those adjustments. Diagnosing the regression required tracing operations rather than attributing every visible difference to the most recently added model.

Randomness was another source of confounding. Changing an earlier operation count can consume different random draws and alter later text placement or colour. Separate random streams reduce this coupling, but later large-scale assertions showed that exact invariants still needed enforcement. “Same seed” alone does not guarantee a valid counterfactual pair.

### 6.2 Early recipe models and direct image refiners

The initial distribution-shift pilot used 96 fit, 32 development and 40 additional historical report images per family. Recipe search considered 32 candidates over 11 controls, including colour offsets, text geometry and clutter parameters. CNN and transformer recipe models learned to imitate search-selected targets. A discriminator scored candidates. These were distinct experiments: the candidate discriminator was not an unrestricted recipe-generating GAN.

The pilot also tested direct CNN, transformer and GAN image refiners. Recipe models ran for 250 steps, the candidate discriminator for 200, and direct models for 400. The pilot was short, about 2.53 minutes in the recorded setting. It cannot justify claims about exhaustive optimisation or the maximum capability of these model families.

| Method | Mean reported descriptor-distance reduction | Main limitation |
|---|---:|---|
| Candidate search | 33.2% | Improved only 3/5 families |
| Recipe CNN | 34.7% | Improved only 3/5; some families worsened |
| Recipe transformer | 31.6% | Improved only 3/5 |
| Candidate discriminator | 36.5% | Improved only 3/5 |
| Direct CNN | 62.6% | Visually rejected: halos and appearance distortions |
| Direct transformer | 61.5% | Visually rejected despite metric gain |
| Direct GAN | 61.9% | Visually rejected despite metric gain |

The direct models' stronger metric gains were not better scientific outcomes. They exposed an objective mismatch: low-level moment alignment could be improved by unwanted changes around protected glyphs. Exact preservation inside a text mask did not prevent disturbing contours outside it. The analogy to SimGAN [R2] motivated an experiment; it did not validate its result. Nor should a means-and-standard-deviations loss be described as implementation of full Deep CORAL [R3].

### 6.3 Why font adjustment moved before optional refinement

Typography repeatedly dominated perceptual mismatch. The project therefore moved font adjustment directly after feature/style extraction, leaving distribution refinement optional. Historical documents used “Phase 3” for fonts, character count and several refinement variants at different times. The current conceptual order is: Phase 1 style extraction/mapping; Phase 2 font adjustment; optional refinement. Historical experiment names are retained as identifiers, not current stage definitions.

The font model used synthetic font-labelled examples from 12 independently sourced OFL font files. Its small CNN and font/effect heads were inspired by DeepFont [R1], but did not reproduce its network or adaptation machinery. Four views were used rather than the paper's full procedure. The task was selection among available fonts, not exact recovery of an unknown target font.

Matched fresh testing showed a useful early improvement: top-1 font accuracy rose from 45.70% to 56.25%, top-3 from 77.47% to 84.77%, and hollow/filled accuracy from 89.06% to 90.23%. Later augmentation revisions improved another matched comparison from 45.96% to 54.56% top-1 and 74.09% to 84.24% top-3. These are separate comparisons on different evaluation conditions, not one monotonic leaderboard.

Low contrast and topology remained serious weaknesses. In one contrast diagnostic, reducing contrast to a gain of 0.15 reduced font accuracy from 55.99% to 18.23% and hollow recognition from 87.5% to 41.67%. A later low-contrast revision improved hollow accuracy from 39.71% to 68.75% but reduced a normal-condition score from 95.31% to 91.93%. Stronger robustness in one stratum could cost accuracy elsewhere.

The outlined real family remained problematic: generated examples still often used filled or heavy text. Available font identity and hollow/filled rendering are different factors. Correctly ranking an available font does not recover missing topology. New-source probes likewise did not establish generalisation: font or hollow mismatches persisted in Pymlex and Sloth examples.

### 6.4 Character-count inference was tried and removed

A count classifier was investigated so that the generator could infer text length without predicting character identity. It used synthetic-only training and achieved 89.29% exact accuracy with MAE 0.161 on an ordinary synthetic test, and 93.53% with MAE 0.085 on a withheld-font test. These values initially appeared promising.

However, shorter real strings exposed weaknesses, and the generator's layout could encode count-dependent shortcuts. A model might infer the occupied band width or spacing convention rather than robustly count characters. Good withheld-font performance does not remove that possibility when the same layout mechanism remains in both splits.

Alternative heatmap approaches compared classical cues, DINO attention and a density CNN. On fresh matched tests, CV reached approximately 50.89%/48.81% and DINO 53.87%/53.27%, while the earlier count classifier reached 88.10%/91.37%. The density CNN reached 38.99%/48.81%. Its Gaussian centre targets had unit mass and the output count was their sum, adapting the density-estimation idea in [R8], not reproducing the original method.

DINO attention did not reliably correspond to one response per character. Lines could attract attention, characters could split into several peaks, and adjacent glyphs could merge. Attention is not an object-count oracle. The count component was ultimately removed from the active pipeline. The current generation policy intentionally spans lengths 2–8 across three character buckets rather than claiming that a failed real-image count estimate is reliable.

### 6.5 DINO-guided parameter search and direct-image Phase 3B

An earlier DINO-guided recipe trial used 15 candidate parameter changes and compared CLS features with an attention-augmented representation. Mean gain on its optimisation metric was about 2.55%; independent CV improved in four of eight sources and worsened in four. Attention changed selection relative to CLS alone for only Courier. This did not show that attention maps supplied a broadly useful control signal.

The later direct-image Phase 3B revisited CNN, transformer and GAN modifications of already generated images. A shared-model trial raised concern about recurring family patterns. Independent per-family fitting then removed shared refinement weights as an explanation. The models used family-specific real fitting data and protected text while modifying the image. The experiment still failed visually.

The per-family audit verified fitting membership and partition-hash separation. Nevertheless, post-hoc DINO distance worsened for seven of eight families. For example, CryptBB changed from 0.2052 to 0.3932, City from 0.1684 to 0.3083 and DarkBay from 0.1346 to 0.3444. Sloth improved only slightly, from 0.4259 to 0.4217. These distances belong to this audit and are not numerically comparable to the later Meta-Sim MMD table.

Specifically, the independent-family continuation used eight separately initialised GAN refiners, with no shared fitted weights. Its protocol allowed up to 1,500 steps, at least 500, validation every 100 and patience four. Selection used own-family grayscale moment distance with identity eligible. The ordinary split was 72 fit, 12 selection and 12 report images where 96 were available. Its residual output was projected to colours from its own synthetic input using a straight-through gradient. These details distinguish it from the earlier shared CNN/transformer/GAN comparison.

Assistant review reported glyph halos, misplaced contours, colour coverage changes and persistent typography mismatch. Restricting outputs to existing RGB values did not preserve perceived colour: reusing a colour across a much larger area changes the image distribution. Exact text pixels at the trial's 192 × 64 resolution also did not establish independent readability. The user stopped the experiment, and it was not promoted.

The useful conclusion is narrower than “GANs do not work.” These particular objectives, protection mechanisms, data amounts and optimisation schedules did not produce convincing improvement. A failed small experiment is evidence against that implementation, not a universal architectural impossibility.

## 7. V2: expanding views, features and supervision

### 7.1 Tier-one scope

V2 was intended to move beyond the minimal three-feature baseline. Its first tier used five target families for development and nine font families for synthetic feature experiments: IBM Plex Mono, Balsamiq Sans, Pixelify Sans, IBM Plex Serif, Bitter, Oswald, Anton, Lora and Caveat. This nine-family experimental split is distinct from the 12-file font-matching bank.

Initial V2 outputs covered lines, curves, dots, rectangles and grain. Six fonts supported training, a seventh selection, and two were withheld. A first small trial collapsed toward negative predictions and zero mask IoU. A larger revision corrected batch-normalisation handling, adjusted positive weighting and increased mask resolution. Architecture was only one potential bottleneck; training implementation and supervision mattered too.

### 7.2 Multi-view did not automatically beat a simpler view

Eight grid layouts were used: 1 × 1, 2 × 1, 3 × 1, 1 × 2, 2 × 2, 3 × 2, 5 × 3 and 10 × 3. Together they supply 63 crops. The early multi-view model shared a CNN encoder and used positional information and tanh-based attention. It was not yet a transformer and did not train 63 independent classifiers.

In one comparison, whole-image, single-view and multi-view F1 values were respectively about 0.706, 0.710 and 0.698 on the ordinary test. Withheld results were about 0.670, 0.680 and 0.679. These values did not support a broad multi-view advantage. Multi-view line, curve and dot false-positive rates remained high, including approximately 0.472, 0.414 and 0.516 in the reported stratum.

A native-resolution revision improved several measures. Dot F1 rose from 0.587 to 0.860 while its false-positive rate fell from 0.575 to 0.028; rectangle F1 rose from 0.585 to 0.828 while false-positive rate fell from 0.252 to 0.016. However, training duration and mask resolution also changed. This was not an isolated proof that preserving native resolution caused the entire improvement.

A subsequent grid-specialist experiment assigned separate heads to views while retaining shared encoding. Aggregate scores remained mixed: the earlier model, shared-local model and grid-specialist variant scored roughly 0.831/0.817/0.807, 0.815/0.819/0.825 and 0.822/0.814/0.813 across their three reported conditions. No clean overall winner emerged. More heads increased modelling options but did not fix ambiguous local labels.

### 7.3 Curve identity requires sufficient context

Curve debugging found that 340 of 341 eligible small 10 × 3 fragments in one diagnostic had at most one pixel of deviation from a straight line. A globally curved primitive can therefore be visually straight inside almost every small crop. Demanding that each fragment confidently predict the parent's curve identity can force the model to learn colour, context or renderer shortcuts.

Local curve AUC fell from approximately 0.845 in a whole-image view to 0.807 in the fine grid. Moreover, clamping a local curve representation changed the global mean probability by only about 0.00221. The model could largely bypass that representation through other appearance features. An attention visualisation alone would not reveal this dependency problem.

Removing a local auxiliary objective improved one curve F1 from 0.754 to 0.793 and reduced false positives from 0.456 to 0.262, but worsened a recipe-shift score from 0.705 to 0.681. This suggests a conflict between the supervision design and local evidence, not a universally beneficial deletion of local learning.

### 7.4 Tilt and neighbouring blocks

To model continuity, another revision added local PCA orientation represented by cos(2θ) and sin(2θ), centroids, coherence and coverage. Doubling the angle respects the fact that an undirected line at θ is equivalent to one at θ + 180°. Eighty-four neighbour relations connected the 63 crops, and a message-passing step supplied relational information.

The trial modestly improved one same-distribution curve F1 from 0.751 to 0.759, but worsened Bezier-shift performance from 0.675 to 0.607 and rectangle performance from 0.798 to 0.672. Most tellingly, 96.98% of predicted orientations lay within ten degrees of horizontal, compared with 37.59% of targets. Angular MAE was 21.503°, slightly worse than a constant-horizontal baseline at 21.312°.

This failed tilt gate is important. Adding a geometrically meaningful target does not establish that the model learns useful geometry. The orientation head mainly reproduced a dominant prior. The user's intuition that neighbouring tilts could reveal a curve remains plausible, but this implementation did not validate it.

## 8. Feature specialists and the mask bottleneck

### 8.1 Curve-specialist failure diagnosis

The specialist direction trained each feature separately and selected views using synthetic evidence. A curve experiment used 20,000 paired examples and withheld curve recipes and rasterisation conditions. Sixteen grid/view combinations and raw versus feature-pop representations were screened.

The confirmed model remained near chance: BCE approximately 0.693 and AP approximately 0.507. Scores on a training diagnostic were almost constant, ranging from 0.496124 to 0.496196 with standard deviation about 8.68 × 10⁻⁶. Yet the same implementation could memorise 16 examples in 100 steps. That argued against a completely broken gradient path, while not proving that the full optimisation setup or targets were adequate.

Threshold selection compounded the problem: coarse 0.01 threshold ties could select 0.99, producing all-negative predictions and zero recall. Fixing threshold search would improve the reporting mechanism, but it would not turn a nearly constant ranking into a useful classifier. Both failures must be acknowledged separately.

### 8.2 Dot scatter: good ranking, failed robustness

A 5 × 3 feature-pop dot specialist achieved selection AP near 0.979. This initially encouraged real-data inference. Locked cluster-shift recall was only approximately 0.484, 0.556 and 0.522 across reported conditions, and raster-shift false-positive rates were approximately 0.192, 0.194 and 0.170. Thus, the apparent selection success did not satisfy the robustness gate.

Full-family inference was still useful as exploratory distribution inspection, provided that its limitations remained explicit. Increasing the number of images reduces sampling variation in a predicted prevalence. It does not remove a systematic bias shared across the entire family. Whole-dataset inference cannot, by itself, validate the classifier that generated those predictions.

### 8.3 Separating stroke detection from shape classification

A two-stage approach first detected a stroke mask and then classified straight versus curved shape. The controlled synthetic trial achieved positive mask IoU 0.7813 and detection recall 0.9975, but absent-image false-positive rate 0.155. Shape accuracy was 0.8525 using oracle masks and only 0.765 using predicted masks.

The gap between oracle and predicted masks directly exposed error propagation. A shape model cannot reliably interpret a curve when its upstream mask retains text fragments and discards the actual interference. The black panels shown during review were mask visualisations; their sparse or incorrect white regions were not proof that the original RGB image lacked a feature.

At the user's request, another trial returned to raw RGB, multiple colours and the 10 × 3 view without feature pop. Thirty patches were obtained by padding 128 × 64 images to 130 × 66 and using 13 × 22 cells. It reached oracle-mask shape accuracy 0.935, but predicted-mask accuracy only 0.525, mask IoU 0.295 and absent-image false-positive rate 0.845.

This was a clear failure of the end-to-end detector, despite a strong oracle result. It also was not a clean ablation of feature pop because view, colour distribution and architecture changed together. The broader lesson is to diagnose where evidence is lost rather than infer that a particular preprocessing method is solely responsible from one gallery.

## 9. Multi-label CNN and pure transformer comparison

### 9.1 Expanding the vocabulary

The next formulation jointly predicted 15 labels from each image. This permits co-occurring features instead of forcing one class per image. The implemented label names require their actual rendering definitions:

| Label | Implemented synthetic meaning |
|---|---|
| Straight interference | Added straight strokes |
| Curved interference | Added parameterised curved strokes |
| Dot scatter | Discrete added dots |
| Decorative shapes | The tested circle-outline operation, not arbitrary shapes |
| Periodic grid | Axis-aligned grid lines |
| Background texture | Gaussian-noise texture, not all textures |
| Background gradient | RGB ramp |
| Background blocks | Rectangular patches |
| Hollow lettering | Hollow glyph rendering |
| Glyph border | Dilation-based border |
| Glyph shadow | Offset glyph shadow |
| Nonlinear text deformation | Sinusoidal text remapping |
| Blur | Gaussian blur |
| Glyph rotation | Per-glyph rotation |
| Multicolour text | Variation in glyph colours |

These definitions constrain interpretation. A texture classifier trained on Gaussian noise is not automatically a detector of stripes, woven patterns or structured background grain. Broad natural-language labels can disguise narrow synthetic coverage.

The corpus contained 8,000 training, 1,200 selection and 1,000 calibration images, plus four locked sets of 800 each: 13,400 in total. Inputs were raw RGB at 192 × 96 without feature pop. Counterfactual visibility checks excluded labels whose removal changed too few pixels. Recipe-shift coverage was also feature-dependent, rather than an independent new renderer for every label.

### 9.2 The CNN and transformer architectures

The multi-label CNN shared visual encoding and used feature-specific attention across 50 spatial tokens derived from 1 × 1, 2 × 2, 5 × 3 and 10 × 3 layouts. The pure transformer used 8 × 8 image patches, giving 288 tokens, an embedding width of 96, three layers, four attention heads, feed-forward width 384, dropout 0.2 and 15 learned feature tokens. It contained 384,783 parameters.

The pure transformer was trained from scratch on the synthetic corpus. It was not a pretrained large ViT [R4]. It is therefore inappropriate to infer its likely performance merely from claims that transformers scale well on large datasets. Scale, pretraining, locality, resolution and optimisation all matter.

The CNN reached selection macro AP around 0.855. Transformer seed results were approximately 0.690, 0.695 and 0.698. On the relevant robustness evaluation, the CNN passed seven feature gates and the transformer four: grid, texture, blocks and blur. This experiment favoured the CNN under the tested budget. It did not establish that all CNNs outperform all transformers, especially because compute-optimal settings and pretraining were not matched.

### 9.3 Real rendering remained the decisive weakness

Both model families were used in real-image inference and matched generation trials, including 6,909 predictions and 105 renders, or 21 cases per family. The renderer exposed errors that synthetic AP did not capture. For example, the CNN predicted periodic grids on about 68.7% of DarkBay images while predicting blocks on only about 0.2%. Qualitative inspection indicated the opposite kind of structure was important.

These are predicted rates, not annotated real-image error rates. Nevertheless, they help locate a concrete mechanism behind poor renders: an incorrect feature estimate becomes a repeated drawing operation. The transformer did not visibly resolve the City/Courier false-grid problem. Stronger architecture alone did not repair the observation-to-operation mapping.

## 10. Joint feature and rendering-parameter learning

### 10.1 Why joint learning was introduced

The next hypothesis addressed a disconnect in the earlier pipeline: feature classification was learned, but much of its conversion into rendering settings was separate. A model saying “curve present” still left amplitude, count and width unspecified. Joint learning added regression targets for controls already known during synthetic rendering.

This is a constrained form of learning the mapping. It learns associations between pixels and named, predefined controls; it does not learn which arbitrary library function to call or invent a new primitive. The renderer schema and the interpretation of each output remain authored code.

### 10.2 Gridded and bidirectionally coupled variants

Two architectures were tested, drawing respectively on gridded representation ideas [R5] and CNN–transformer coupling [R6]. Both used a convolutional stem with channels 3 → 32 → 64 → 64, strides 1, 2 and 2, group normalisation and SiLU. At 192 × 96 input resolution, this produced a 64-channel, 24 × 48 map.

The gridded model padded its feature-map width to 50 and organised it into 3 × 10 cells. Each cell contained 8 × 5 local positions. Local features were projected to width 96 and pooled through learned-query cross-attention before three transformer layers processed the 30 positioned grid tokens. The model had 442,599 parameters. It borrowed a structured grid idea without implementing the full Gridded Transformer Neural Process, its probabilistic prediction task or its claimed properties.

The coupled hybrid added three rounds of local residual convolution and communication with the global tokens. Global information was upsampled to local features and passed through a learned tanh gate, initially small. It had 553,770 parameters. This was a project-specific Conformer-inspired architecture, not the original Conformer reproduced at a smaller input size.

The design permits local evidence to inform the global interpretation and global context to influence local processing. That capacity is not itself proof that the network uses the interaction correctly. Appropriate ablations and interventions remain necessary, as the earlier bypass and tilt failures demonstrated.

### 10.3 Outputs, loss and optimisation

Fifteen feature heads predicted presence. Nine regression heads predicted normalised renderer controls:

| Parameter | Synthetic training range |
|---|---|
| Line count | 1–4 |
| Line width | 1–3 |
| Curve count | 1–4 |
| Curve width | 1–3 |
| Curve amplitude | 9–24 |
| Dot count | 8–64 |
| Dot radius | 1–3 |
| Grid spacing | 9–26 |
| Blur radius | 0.8–1.5 |

The objective combined masked binary cross-entropy with weighted Smooth L1 regression:

```
L = L_feature_BCE + 0.5 × L_parameter_SmoothL1
```

Parameter supervision was conditioned on the owning feature and visibility. This avoids asking the model to infer a meaningful curve amplitude from an image without a visible curve. It does not completely solve identifiability: overlap can hide some drawn dots, and integer width may produce indistinguishable rasterisations.

Training used AdamW with learning rate 0.0005, weight decay 0.02, batch size 32, gradient clipping at 3 and dropout 0.2. The schedule allowed up to 40 epochs, with minimum training and patience constraints. Selection combined macro AP and normalised parameter MAE:

```
selection score = macro AP − 0.25 × parameter MAE
```

This weighted objective encodes a project preference. Its coefficient is not a theoretically optimal balance. The selected checkpoint must also not be confused with the final epoch, and a best single-seed metric must not replace an all-seed robustness claim.

### 10.4 What the completed trials support

Across the required seeds and strata, the gridded model passed six of the 15 feature gates; the hybrid passed seven. The hybrid's additional passing feature was curved interference. Both passed dot scatter, grid, texture, gradient, blocks and blur. Both failed several typography-related features and straight interference under the full gate.

Both architectures passed six of nine parameter criteria against a training-median baseline. Line width, dot radius and curve amplitude failed the all-seed/all-test criterion. Curve amplitude had looked acceptable in one hybrid seed, illustrating why a selected checkpoint's result cannot be reported as robust across training runs.

These results are more encouraging than the pure-transformer experiment but remain partial. Some controls are learnable within the synthetic setup; several are not robust even there. No corresponding labelled real-feature evaluation establishes that the passing controls retain those properties on real CAPTCHA families.

## 11. CAPRI: combining learned inference with V1 measurements

### 11.1 Why fusion was preferable to wholesale replacement

The newer classifiers did not consistently render closer images than V1. The practical response was to retain useful CV measurements rather than force all rendering decisions through neural predictions. CAPRI combined the frozen seed-731 joint model with the earlier palette, density, layout and block measurements.

The fusion policy preserves CV evidence rather than letting a weak learned negative erase it. Unsupported grid and blur additions are suppressed without corroborating measurement. Only selected learned controls are applied, notably curve controls; the existence of 15 heads does not mean all 15 unconditionally drive the renderer.

This makes the system less elegant than an entirely learned inverse renderer, but more faithful to the experimental evidence. It also creates a new attribution problem: improved renders cannot automatically be credited to the neural model when CV measurements dominate important decisions. A CV-only versus fused comparison is therefore essential.

The joint-model real trial produced 420 matched renders across both architectures and CV/fused conditions. Its conclusions remained mixed. The packaged hybrid checkpoint includes a curve-amplitude choice supported by that seed, although the stronger all-seed amplitude criterion failed. The release should consequently be described as an experimental checkpoint with a documented policy, not a fully validated mapping model.

### 11.2 Similar-font variety

The later CAPRI package retained font adjustment but replaced a single chosen font with a small weighted pool. By default, one font is sampled per image from the top three frozen font-model scores. If their scores are p₁, p₂ and p₃, the sampling weights are their normalised values, pᵢ divided by their sum. Fonts are not mixed arbitrarily within one word.

The rationale is to avoid overcommitting the recognition corpus to one imperfect approximation. However, top-three classification probabilities are not verified perceptual distances, and their calibration on real images is unknown. A confidently wrong ranking can still generate an inappropriate pool. “Similar-font variety” is therefore a practical policy name, not a demonstrated perceptual guarantee.

The intended benefit is recognition robustness, but that benefit is untested until a recogniser is trained with and without the mixture under matched conditions. Variety may also increase the domain gap if the extra fonts are less representative. The policy introduces a measurable hypothesis, not a free improvement.

## 12. Meta-Sim-inspired refinement of CAPRI recipes

### 12.1 A deliberately restricted adaptation problem

After direct image refinement failed, a narrower approach adapted six procedural controls while retaining the renderer's structure. This experiment was inspired by Meta-Sim [R10]: learn how simulator parameters should change so that generated outputs better match real observations in a fixed representation. It was not a reproduction of Meta-Sim's full method.

Each family received its own small adapter: eight inputs, a hidden layer of 24 units with tanh, and six output gains. Output log-gains were bounded using ln(4) × tanh, giving multiplicative gains between 0.25 and 4. Initial output was the identity transformation. The six mutable controls were line count, line width, dot count, curve count, curve amplitude and measured block count.

Palettes, gradients, text, fonts and layout were not optimisation outputs. The candidate had to preserve the chosen baseline's exact glyph geometry and RGB. An invalid candidate fell back to the baseline with a recorded reason. This restriction sacrifices flexibility to reduce the likelihood of corrupting labels or repeating the earlier recolouring failures.

Multiplicative control has its own limitation: multiplying a zero baseline count does not create an absent primitive. The adapter cannot generally repair a missing operation outside its permitted support. Nor can it fix an incorrect font topology or invent a stepped background pattern that the renderer does not represent.

### 12.2 Frozen DINO objective and numerical derivatives

The fitting procedure used frozen DINO ViT-S/8 CLS features [R7] with 224 × 80 input processing and the recorded normalisation. DINO was not retrained on the CAPTCHA collections. The distribution loss was a biased Gaussian-kernel maximum mean discrepancy estimate:

```
MMD² = mean k(z_real, z_real)
     + mean k(z_synthetic, z_synthetic)
     − 2 × mean k(z_real, z_synthetic)
```

The kernel bandwidth came from fit-set pairwise distances, with a lower bound. Since the renderer is not differentiable, symmetric finite differences approximated changes in the visual features for parameter perturbations. The adapter then used these numerical sensitivities during optimisation. The trial used 12 optimisation steps, batch size four, Adam learning rate 0.035, gradient clipping at one and finite-difference perturbation 0.2. Identity remained an eligible checkpoint, and selection occurred every four steps.

The protocol is a small pilot. It explores whether a bounded distribution-based adjustment can help, not whether the adapter has converged to the best attainable family model. Quantised operations can make finite differences zero: line-width gains remained at one in the reported trial. More training does not automatically solve an insensitive numerical derivative.

Each family used 48 real fit, 16 selection and 32 report images, with distinct membership in that run. Sixty-four fresh paired outputs per family produced 640 baseline/refined images across five families. These partitions reduce direct within-run reuse, but historical visual exposure means the report sets are not pristine new-family evaluations.

### 12.3 Results and their limits

| Family | DINO MMD reduction | Independent CV distance: before → after | Interpretation |
|---|---:|---:|---|
| CryptBB | 3.79% | 1.86174 → 2.18527 | Mixed: DINO gain but CV worsened |
| City | 4.33% | 1.40547 → 1.25970 | Both metrics improved |
| Courier | 7.20% | 0.77412 → 0.51403 | Both metrics improved |
| DarkBay | 27.71% | 1.01952 → 0.72089 | Largest measured gain; two of 64 candidates fell back |
| Outlined Text CAPTCHA | 5.96% | 0.84504 → 0.68543 | Both improved; typography mismatch persisted |

The CV comparator used 12 dimensions derived from colour, edges and projection-related summaries, standardised using fitting data. It supplied a check outside the optimised DINO representation. It was still a low-dimensional proxy and not an independent human judgement or text-recognition measurement.

A paired bootstrap with 300 resamples gave a CryptBB absolute discrepancy-change interval of approximately [−0.02610, 0.00370], crossing zero. The other reported intervals lay below zero: City [−0.03536, −0.01066], Courier [−0.03560, −0.01718], DarkBay [−0.12952, −0.08816] and Outlined [−0.05264, −0.02941]. These are intervals for absolute discrepancy changes, not percentage improvements. They are conditional on the fitted checkpoint and sampled data; they do not include variation from retraining many adapters.

The learned controls tended to reduce dots and increase some supported blocks. CryptBB and DarkBay block gains were about 3.77 and 3.56. Dot gains ranged from approximately 0.25 to 0.65 across families. These changes are interpretable, but interpretability does not prove correctness: a metric can prefer the wrong change for the wrong reason.

Qualitative inspection remained cautious. DarkBay's broad block/dot balance improved while coherent structure remained imperfect. City still lacked the target's stepped appearance. Outlined text remained too filled. The evidence supports “a bounded recipe adapter improved two chosen distances in four of five development families,” not “Meta-Sim solved the target distribution.”

### 12.4 Adapter binding and the font-mixture caveat

Adapters are bound to the exact profile for which they were fitted. A new family or changed profile does not silently receive another family's adapter. Automatic refinement is available only for the five matching hybrid profiles; forced refinement on an unmatched profile fails. This prevents one obvious form of cross-family misuse.

The adapters were fitted using the single-font baseline. Combining them with the newer font mixture changes the generator distribution. A 378-image smoke validation checked determinism, operation and relevant invariants, including mismatch rejection; it did not re-establish the earlier quality gains for every mixed-font configuration. One DarkBay fallback appeared in that smoke test.

Therefore, the released combination is supported by component evidence and an operational test, but lacks a complete quality ablation. The correct next comparison is a factorial experiment: single versus mixed fonts crossed with refinement off versus on, using matched labels, seeds, budgets and recognition evaluation.

## 13. Tools, implementation and corpus production

### 13.1 Software and assets

| Tool or resource | Role in the work | Reproducibility qualification |
|---|---|---|
| Python | Experiment orchestration, training, inference and generation | Existing environment reused; per-run snapshots retained |
| PyTorch 2.5.1+cu121 | CNN, transformer, font, count and refinement training | CUDA-enabled experiments used the available RTX A4000 |
| NumPy 2.0.2 | Arrays, sampling, descriptors and numerical analysis | Recorded package version |
| Pillow 11.3.0 | Text and primitive rasterisation, masks, image I/O | Rasterisation affects feature appearance and labels |
| OpenCV 4.13.0 / package 4.13.0.92 | MSER, components, geometry and image measurements | MSER proposals are not ground-truth character boxes |
| scikit-learn 1.6.1 | Logistic Regression, Extra Trees, calibration and metrics | Synthetic selection and calibration boundaries matter |
| scikit-image 0.24.0 | Image descriptors and supporting processing | Initial environment record; not every module used in every run |
| PHP backends and instrumented adapters | Initial capability and determinism pilots | Their use must not be confused with the later Python overlay corpus |
| DINO source and weights | Frozen visual embeddings in selected experiments | Pretrained vision model; no language component |
| Google Fonts / audited OFL files | Independent typography bank | Font notices and asset hashes retained |
| JSON, JSONL, CSV, NPZ and joblib/checkpoints | Protocols, memberships, labels, predictions and models | Machine-readable records supplement narrative notes |
| PowerShell and local status files | Launching, monitoring and audit coordination | Status can be stale; corroborate with logs and audit completion |

The project used open-source methods and code with asset-specific checks, rather than assuming every item in an open repository was freely reusable. Package metadata is not a complete transitive licence audit, and proprietary operating-system or driver components should not be misrepresented as open-source research algorithms. The initial paper register's statement that no paper-supplied weights were used became obsolete once DINO was introduced; the chronology in this chapter explicitly corrects that claim.

### 13.2 Release structure and generation contract

The release includes frozen model checkpoints, profiles, renderer code, font assets/notices, integrity manifests and instructions. A fresh profile uses both CV measurements and frozen learned inference. Generating from a cached profile does not require rereading the target images. No recogniser is involved in choosing the synthetic label: the label is the string intentionally submitted to the renderer.

The generation policy covers lengths 2–8 and three buckets: numeric, alphabetic and alphanumeric. Their Cartesian product gives 21 cells. A deterministic cycle balances these cells without assuming the target's hidden alphabet or length. This is a deliberate generic-training policy; it does not reproduce the target family's true text prior.

Text-hash partitioning reserves disjoint string partitions for training, validation and testing. Strings may repeat within an allowed partition, while image hashes are audited for uniqueness. Repeated text is not automatically a duplicate image, and unique image pixels do not prove semantic diversity.

### 13.3 Scaling to 100,000 images per family

The requested production target was 100,000 images for each of five families, or 500,000 total. Generation used shards, per-image label/recipe metadata, aggregate manifests and an audit stage. At the chapter's initial recorded snapshot, DarkBay, City and Outlined each had 100,000 generated and audited images. A subsequent log check found that CryptBB's recovery coordinator had stopped on a Windows status-file replacement error, leaving the bank recovery incomplete; Courier had previously failed and was awaiting recovery. These are dated operational facts, not a permanent description of the eventual dataset. The accompanying status snapshot and any later recovery notes should take precedence over treating this paragraph as a live monitor.

Large-scale generation exposed an important gap in the smoke test. Some refinement candidates violated exact glyph geometry/RGB assertions. A conservative generation wrapper rejects such a candidate and returns the valid baseline with explicit fallback metadata. It does not weaken the invariant to allow a distorted candidate to pass. Historical failed attempts and completed shards are preserved.

DarkBay recorded 2,483 refinement fallbacks among 100,000 outputs, or 2.483%. Consequently, its corpus is a documented mixture of accepted refined outputs and baseline fallbacks. That mixture must be included when interpreting downstream performance. City and Outlined reported zero fallbacks in their completed snapshot.

The recovery mechanism demonstrates operational resilience, not aesthetic success. An audit can establish that files decode, labels and paths match, counts balance and invariant checks pass. It cannot establish that a glyph is independently readable, that every label remains perceptually unambiguous, or that the synthetic distribution resembles the real family sufficiently for recognition transfer.

During preparation of this chapter, the finite recovery coordinator was resumed with bounded retries for status-file replacement. All ten CryptBB shards were already complete, so they were reused for auditing rather than regenerated. Courier remained next in the recovery sequence. This repair did not restart the stopped research loop, alter frozen model weights or interrupt the separate recognition-training processes found in another project.

## 14. Critical synthesis

### 14.1 What demonstrably worked

Several concrete capabilities were achieved. The project constructed reproducible synthetic examples with visible-effect supervision and uncertainty handling. It trained classical and neural feature models while keeping real text labels out of classifier fitting. It identified and repaired a specific shared-background mapping bug. It showed that a hybrid model could learn some feature and parameter targets across synthetic strata. It packaged a reproducible, inspectable procedural pipeline and generated large labelled corpora. Finally, bounded Meta-Sim-inspired adaptation improved both selected distribution metrics for four of five development families.

These achievements justify continued evaluation. They do not establish that the complete system passes every original gate or that the resulting data improves a recogniser. A thesis should present these as separate propositions, supported by separate evidence.

### 14.2 Principal threats to validity

**Generator-induced shortcuts.** Common colour, layout, rasteriser and co-occurrence conventions can make synthetic tests easier than real transfer. Withheld fonts alone do not remove these conventions. The rejected colour-coded pilot and count-layout concern are direct examples.

**Locally unidentifiable labels.** Fine crops may not contain the evidence needed to identify their parent's feature. A globally curved line can be locally straight. Causal masks answer what operation changed the pixels, not always what shape is recognisable from those pixels.

**Repeated development exposure.** The five principal families guided many revisions. Some previously reserved images later entered whole-family inference. Freezing each model before inference does not reset that accumulated exposure or restore a pristine benchmark.

**Objective mismatch.** Synthetic AP, low-level distance, DINO discrepancy, subjective similarity and OCR accuracy are different outcomes. The direct-refiner failures show why improvement on one cannot stand in for another.

**Incomplete ablations.** Resolution, training duration, architecture and palette sometimes changed together. Such experiments identify a better configuration under their conditions, but cannot cleanly assign causal credit to one component.

**Seed and checkpoint selection.** Seed-731 packaging is an engineering choice. The all-seed curve-amplitude failure remains relevant even if that checkpoint looked good. Reporting only the selected seed would conceal instability.

**Representation limits.** Finite font banks, axis-aligned grids, Gaussian texture and bounded curve families cannot express every target style. Optimising within the wrong support can increase confidence without creating the missing structure.

**Assistant-review limitations.** Visual review found useful failures, but was neither a blinded human study nor independently audited feature ground truth. It influenced development and should not be presented as an unbiased final evaluation.

**Small refinement budgets and uncertainty.** Twelve adapter steps and a single fitted run per family are insufficient to characterise optimisation variance. Bootstrap intervals over samples do not include all sources of training or model-selection uncertainty.

**Combined-policy uncertainty.** Font mixture plus refinement was operationally tested after the adapter had been fitted on a different single-font policy. Quality and recognition gains of that combination remain to be measured.

### 14.3 Why intentional underfitting is not the right objective

The motivation for generic recognition was sometimes expressed as making PARSeq underfit the synthetic data. The useful goal is better transfer, not deliberately poor training fit. Underfitting can discard genuinely transferable character structure. A stronger protocol controls synthetic diversity, regularisation, model capacity and stopping criteria, then measures performance on appropriately reserved real data.

Likewise, making every family look different is only a weak sanity check. A renderer could assign a different wrong colour to every family and satisfy that test. Meaningful differentiation must correspond to the relevant visual factors and ultimately support the intended recognition task.

### 14.4 Required downstream study

A rigorous next experiment would compare a generic baseline, CV/font-only generation, CAPRI fusion, font-mixture CAPRI and CAPRI with bounded refinement. Training budgets, string distributions, decoder architecture and random seeds should be matched. A factorial subset should isolate font variety from refinement. Both mean performance and per-family failure should be reported.

PARSeq [R13] is an appropriate planned recogniser, but the protocol must not tune repeatedly on its final real test. Distinct style-fit, development and test roles should be defined before new family acquisition. For the historical bank, exposure should be disclosed rather than concealed by renaming files. A genuinely new source provides a stronger transfer test than another rerender of the five development families.

Feature evaluation should additionally use independently audited real labels for a bounded subset of interpretable features. That would allow precision, recall and false-positive rates instead of prediction prevalence alone. Intervals should respect image/base-group dependence, and recognition experiments should include multiple training seeds where feasible.

## 15. Contribution statement and conclusion

CAPRI is an implemented exploratory framework for converting untranscribed CAPTCHA appearance into constrained procedural generation. Its development demonstrates that local feature classification, renderer mapping and distribution matching are separate problems. Solving one incompletely does not automatically solve the others.

The progression was not a simple replacement of weak classical methods by stronger neural networks. The original 10 × 3 classifiers offered useful synthetic ranking but poor real transfer. Multi-view and specialist models exposed context and supervision problems. A pure transformer did not outperform the tested CNN. The later hybrid learned some useful controls, but the packaged pipeline still benefited from retaining V1's explicit measurements. Direct image refinement was rejected, while restricted recipe adaptation produced modest, measurable and family-dependent improvements.

The strongest current claim is therefore conditional: **within a fixed rendering vocabulary, synthetic-trained inference combined with explicit measurements and bounded unlabelled-real adaptation can produce an operational, partially style-adapted labelled corpus.** Exact family recreation, robust real-feature detection, broad unseen-family generalisation and improved real CAPTCHA recognition remain unproven. Those limits are central findings of this chapter rather than qualifications to be hidden after the results.

## References

**[R1]** Wang, Z., Yang, J., Jin, H., Shechtman, E., Agarwala, A., Brandt, J., and Huang, T. S. (2015). *DeepFont: Identify Your Font from An Image*. ACM Multimedia. [Primary paper](https://arxiv.org/abs/1507.03196).

**[R2]** Shrivastava, A., Pfister, T., Tuzel, O., Susskind, J., Wang, W., and Webb, R. (2017). *Learning from Simulated and Unsupervised Images through Adversarial Training*. CVPR. [Primary paper](https://arxiv.org/abs/1612.07828).

**[R3]** Sun, B., and Saenko, K. (2016). *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*. [Primary paper](https://arxiv.org/abs/1607.01719).

**[R4]** Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR; preprint 2020. [Primary paper](https://arxiv.org/abs/2010.11929).

**[R5]** Ashman, M., Diaconu, C., Langezaal, E., Weller, A., and Turner, R. E. (2025). *Gridded Transformer Neural Processes for Spatio-Temporal Data*. ICML, PMLR 267, 1722–1761. [Published paper](https://proceedings.mlr.press/v267/ashman25a.html).

**[R6]** Peng, Z., Huang, W., Gu, S., Xie, L., Wang, Y., Jiao, J., and Ye, Q. (2021). *Conformer: Local Features Coupling Global Representations for Visual Recognition*. ICCV, 367–376. [Published paper](https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Conformer_Local_Features_Coupling_Global_Representations_for_Visual_Recognition_ICCV_2021_paper.html).

**[R7]** Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., and Joulin, A. (2021). *Emerging Properties in Self-Supervised Vision Transformers*. ICCV. [Primary paper](https://arxiv.org/abs/2104.14294).

**[R8]** Lempitsky, V., and Zisserman, A. (2010). *Learning to Count Objects in Images*. NeurIPS. [Author-hosted paper](https://www.robots.ox.ac.uk/~vgg/research/counting/NIPS2010.pdf).

**[R9]** Darcet, T., Oquab, M., Mairal, J., and Bojanowski, P. (2024). *Vision Transformers Need Registers*. ICLR; preprint 2023. [Primary paper](https://arxiv.org/abs/2309.16588).

**[R10]** Kar, A., Prakash, A., Liu, M.-Y., Cameracci, E., Yuan, J., Rusiniak, M., Acuna, D., Torralba, A., and Fidler, S. (2019). *Meta-Sim: Learning to Generate Synthetic Datasets*. ICCV. [Primary paper](https://arxiv.org/abs/1904.11621).

**[R11]** Ruiz, N., Schulter, S., and Chandraker, M. (2019). *Learning to Simulate*. ICLR; preprint 2018. [Primary paper](https://arxiv.org/abs/1810.02513).

**[R12]** Ye, G., et al. (2018). *Yet Another Text CAPTCHA Solver: A Generative Adversarial Network Based Approach*. ACM CCS. DOI: 10.1145/3243734.3243754. [Author-hosted paper](https://zwang4.github.io/publications/ccs18.pdf).

**[R13]** Bautista, D., and Atienza, R. (2022). *Scene Text Recognition with Permuted Autoregressive Sequence Models*. ECCV. [Primary paper](https://arxiv.org/abs/2207.06966).

**[R14]** Epshtein, B., Ofek, E., and Wexler, Y. (2010). *Detecting Text in Natural Scenes with Stroke Width Transform*. CVPR. [Author-hosted paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/201020CVPR20TextDetection.pdf).

**[R15]** Dalal, N., and Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR. [Author-hosted paper](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf).

**[R16]** Geurts, P., Ernst, D., and Wehenkel, L. (2006). *Extremely Randomized Trees*. Machine Learning, 63, 3–42. [Published paper](https://doi.org/10.1007/s10994-006-6226-1).

## Appendix A. Reading the accompanying evidence

The evidence index links the chapter's stages to local protocols, results and critical reviews. The manifest records hashes and snapshot times; it is an audit trail, not an assertion that every archived sentence is current. The bibliography separates primary papers from project implementation evidence. Earlier planning documents saying “not trained” remain valid historical statements about their own stage, not descriptions of the final release.

The supplied galleries are qualitative illustrations of development data. They are not randomised blinded assessments or sufficient evidence for accuracy. The numerical tables above should be read with their original run's protocol and denominator. In particular, do not average AP, mask IoU, parameter MAE, DINO distance and OCR accuracy into one artificial measure of progress.

### A.1 Experiment progression and decision record

| Stage | Tested idea | Recorded decision or limitation |
|---|---|---|
| Resource pilots | Reuse general-purpose PHP and Python capabilities | Replay and licensing checks first; not all assets eligible |
| Initial labels | Three effects on 30 overlapping bricks | Reject colour shortcuts; add uncertain labels and visible contribution |
| Candidate corpus | Expand to 5,600 parents | Integrity improved; original acceptance audit remained incomplete |
| Classical V1 | LR and Extra Trees with ordinary/enhanced descriptors | LR selected; curve/dot confident thresholds unresolved |
| Frozen real inference | Aggregate local predictions into a family story | Severe saturation; predicted prevalence not accuracy |
| CV mapping | Palette, layout, density and block relationships | Useful partial matches; Phase 1 gate failed |
| Early refinement | Search, recipe models and direct image networks | Direct models rejected despite larger numerical gains |
| Font iterations | Synthetic CNN, views, contrast and geometry augmentation | Some matched synthetic improvements; real topology remained difficult |
| Background investigation | Trace operations and isolate rectangle evidence | Specific Courier regression repaired; recall not established |
| Count and heatmaps | Count classifier, CV, DINO and density CNN | Synthetic success did not transfer reliably; count removed |
| DINO recipe search | Attention/CLS-based parameter candidates | Small mixed gains; attention's added value unproven |
| Direct Phase 3B | Shared models, then independent family GANs | Halos and distribution failures; experiment stopped |
| Multi-view V2 | Coarse-to-fine grids and per-grid heads | Mixed results; small crops lost curvature evidence |
| Neighbour geometry | Tilt and continuity messages | Tilt failed a constant baseline; no gate pass |
| Feature specialists | Dedicated view selection and feature pop | Curve near chance; dot selection success failed shift tests |
| Two-stage strokes | Detect mask, then classify its shape | Predicted masks degraded otherwise stronger oracle performance |
| Raw multicolour grid | Remove pop; return to 10 × 3 | Very high false positives; not a rescue |
| Multi-label CNN | Learn 15 co-occurring feature outputs | Seven synthetic feature gates; real mapping errors persisted |
| Pure transformer | Global attention from RGB patches | Four gates in the tested setup; not better than CNN |
| Joint gridded/hybrid | Learn feature presence and nine controls | Hybrid seven features; both six parameter gates |
| CAPRI fusion | Preserve V1 CV evidence with selected V2 outputs | Packaged experimental system; not all heads trusted |
| Meta-Sim-inspired adapter | Learn bounded per-family recipe gains | Both metrics improved in four of five; CryptBB mixed |
| Similar-font mixture | Sample one of top-three fonts per image | Operationally checked; recognition benefit not yet established |
| Corpus production | 100,000 labelled images per family | Audit and recovery required; completion is not OCR validation |

### A.2 Software and asset sources

The initial PHP source revisions and asset exclusions are preserved in the copied resource-acquisition record. Public upstream sources are [Gregwar/Captcha](https://github.com/Gregwar/Captcha), [Securimage](https://github.com/dapphp/securimage), [MobiCMS Captcha](https://github.com/mobicms/captcha), [simple-php-captcha](https://github.com/yasirmturk/simple-php-captcha) and [Laminas Captcha](https://github.com/laminas/laminas-captcha). Laminas was designated as a withheld renderer; downloading it did not establish completed withheld-renderer testing. Portable PHP 8.5.11 with GD and FreeType was recorded for the initial pilots.

The independently sourced font expansion used [Google Fonts](https://github.com/google/fonts), with revision and per-file licence/hash records. Frozen DINO code/weights came from the [official DINO repository](https://github.com/facebookresearch/dino). Implementation tools are documented by [PyTorch](https://pytorch.org/docs/stable/index.html), [scikit-learn](https://scikit-learn.org/stable/), [OpenCV](https://docs.opencv.org/), [Pillow](https://pillow.readthedocs.io/), [NumPy](https://numpy.org/doc/), [SciPy](https://docs.scipy.org/doc/scipy/) and [scikit-image](https://scikit-image.org/docs/stable/). These general links identify tools; the saved environment and source snapshots, rather than current website defaults, define the versions used. Joblib 1.5.3 and SciPy 1.13.1 are also pinned in the CAPRI release requirements.

One additional font-transfer-learning PDF was consulted historically, but its bibliographic metadata was not verified and no particular method was adopted. It is deliberately not presented as an implemented or fully cited contribution. This is preferable to padding the reference list with an unverified attribution.


## Appendix B. Recorded visual evidence

These are existing experiment exports, not newly selected best outputs. The three final examples per family use the earlier fixed five-character protocol: numeric, alphabetic and alphanumeric; first three fit recipes in their saved order, seeds 63000–63002. Real references are untranscribed development examples, not paired reconstructions. The CAPRI/refined columns preserve the same chosen synthetic text and font within each pair. They illustrate the single-font Meta-Sim trial, not a completed evaluation of the later font-mixture corpus.

### B.1 Original grid and feature-pop representations

![Original V1 grid demonstration](assets/v1_grid.png)

Figure B1. The recorded V1 grid view. Overlap preserves nearby pixels but does not establish object continuity or feature accuracy.

![Feature-pop transformations](assets/feature_pop.png)

Figure B2. Original RGB, CLAHE, feature pop, Canny and morphological views. Character strokes are strongly enhanced alongside interference; this is why feature pop cannot be treated as a feature-only mask.

### B.2 Rejected direct image refinement

![Own-family GAN trial with visible halos](assets/rejected_family_gan.png)

Figure B3. Own-family GAN trial. The protected text survives, but surrounding halos, contours and colour coverage change. Baseline errors also remain. This experiment was rejected; it is included to show a failure mechanism, not a successful refinement result.

### B.3 CAPRI and bounded Meta-Sim-inspired recipe refinement

![cryptbb: real reference, CAPRI and recipe refinement](assets/cryptbb_comparison.png)

Figure B4. cryptbb: real references at left, CAPRI in the middle and bounded recipe refinement at right. Three fixed examples are illustrative, not sufficient for a family-level quality verdict.

![city: real reference, CAPRI and recipe refinement](assets/city_comparison.png)

Figure B5. city: real references at left, CAPRI in the middle and bounded recipe refinement at right. Three fixed examples are illustrative, not sufficient for a family-level quality verdict.

![courier: real reference, CAPRI and recipe refinement](assets/courier_comparison.png)

Figure B6. courier: real references at left, CAPRI in the middle and bounded recipe refinement at right. Three fixed examples are illustrative, not sufficient for a family-level quality verdict.

![darkbay: real reference, CAPRI and recipe refinement](assets/darkbay_comparison.png)

Figure B7. darkbay: real references at left, CAPRI in the middle and bounded recipe refinement at right. Three fixed examples are illustrative, not sufficient for a family-level quality verdict.

![outlined_text_captcha: real reference, CAPRI and recipe refinement](assets/outlined_text_captcha_comparison.png)

Figure B8. outlined text captcha: real references at left, CAPRI in the middle and bounded recipe refinement at right. Three fixed examples are illustrative, not sufficient for a family-level quality verdict.
