# DarkBay: complete experimental record through Model 2

Recorded 6 October 2026. This document describes the experiments actually saved in this workspace, not an idealized protocol. Model 1 means the DS1 PARSeq reader experiment; Model 2 means the DS2 curriculum reader experiment. These names do not mean controller 1 and controller 2, or the older external repositories named `model 1` and `model 2`.

## 1. Outcome and scope

The CAPRI workflow used 100 unlabeled DarkBay reference images to initialize and refine a synthetic rendering domain. A frozen neural hybrid supplied feature predictions; custom, DINO and mixed controllers refined renderer settings from pixels. Their 300 recipe entries generated 100,000 labeled synthetic images for Model 1. Three Model 1 readers then ranked a fresh 100,000-image corpus for Model 2 curriculum training.

| Result | Model 1 | Model 2 |
|---|---:|---:|
| Dataset | Darkbay_ds1_noFont-intel | Darkbay_ds2_noFont-intel |
| Synthetic optimization / validation images | 100,000 / none | 95,000 / 5,000 |
| Reader seeds | 48, 123, 456 | 48, 123, 456 |
| Epochs per seed | 20 | 20 |
| Checkpoint selection | Fixed final epoch 20 | Highest synthetic validation accuracy per seed |
| Selected epochs, seed order above | 20 / 20 / 20 | 18 / 20 / 18 |
| Real ensemble, no filtering | 496/900 = 55.11% | 565/900 = 62.78% |
| Strictest synthetic-derived rejection rule, real result | 232/339 = 68.44% | 411/530 = 77.55% |
| Strictest rule, fraction attempted | 37.67% | 58.89% |

The no-filter improvement is 69 additional correct answers, or 7.67 percentage points. Fresh data, augmentation, curriculum, validation and checkpoint selection all changed; this is not an ablation proving curriculum alone caused the improvement.

No font-intelligence retrieval module was integrated. The renderer still uses fonts and the hybrid still predicts font attributes: “noFont-intel” does not mean “no font information.” No VLM, XGBoost, SAM, DINOv3 or Meta-Sim2 checkpoint is part of the selected CAPRI pipeline. GLM-OCR was a separately requested external benchmark, documented below.

## 2. Evidence map and terminology

Workspace: `C:/Users/pranav/Documents/ChatGPT/thesis`.

| Evidence | Location under workspace |
|---|---|
| Agreed workflow | CAPRI_WORKFLOW.md |
| Maintained implementation | src/capri_controller/ |
| Original completed controller run and DS1 rendering evidence | capri_darkbay_workflow/ |
| Active checkpoint provenance | models/registry.json |
| External source/runtime paths | configs/local.json |
| Frozen 100-reference manifest | configs/targets/darkbay.json |
| Model 1 training, testing and analysis | runs/parseq_darkbay_100k_20261005/ |
| Dataset 2, Model 2 and analysis | runs/Darkbay_ds2_noFont-intel/ |
| Change history | docs/MAINTENANCE_LOG.md |

Paths in subsequent sections are relative to that workspace unless explicitly absolute. The original handoff is in `E:/pranav/Final sem/sourcefree_synth/docs/ITERATION2_CONTROLLER_HANDOFF.md`, not this workspace's docs directory. It describes earlier simple/specialist neural and XGBoost experiments. The selected workflow uses the **simple neural hybrid, epoch 15 EMA**, not the specialist or XGBoost replacements. “Tab 3” in the conversation has no verified model identifier in the inspected artifacts; this record identifies the actual checkpoint instead of assuming a UI tab names a different model.

The controller's job is appearance matching, not reading the target string. The renderer knows its own synthetic text, masks and settings; that does not grant it the real target's text, character positions or rendering labels.

## 3. Real reference images and information boundaries

The saved DarkBay source comprises 1,000 unique real images, each 200×50 pixels. The controller input is exactly 100 unique RGB images. The maintained manifest preserves the first 100 fit references from `controller_mvp_v7_dino/DarkBay/target_splits.json`; that experiment inherited the earlier controller split. It is not a newly sampled set for DS2. Exact membership and RGB hashes are authoritative; this record does not assert an unverified original sampling seed or random-sampling guarantee.

Original source paths include both `pool` and `eval` folders under `E:/pranav/cryptbb_sourcefree/data/darkbay`. Those historical folder names do not define the reader evaluation split. The current reader test excludes all 100 controller-reference RGB hashes from the 1,000 images, leaving 900. Source filenames can contain answers, but the controller's computation loads image pixels and does not parse filename stems as target text. Real test answers were parsed only after predictions were frozen in the evaluation protocol.

| Stage | Real information used | Synthetic labels used? |
|---|---|---|
| Hybrid inference and controller fitting | Pixels, canvas dimensions, inferred regions and appearance | Renderer creates its own strings/settings |
| Corpus generation | Frozen recipes and model-derived length | Yes, renderer supplies correct synthetic strings |
| Reader optimization | No real images or answers | Yes |
| Difficulty ranking | No real answers | Yes, known synthetic string for NLL |
| Checkpoint selection | No real answers | Model 2 synthetic validation only |
| Rejection-policy development | No real answers to choose rules | Discovery/calibration labels |
| Final real scoring | Previously frozen predictions/decisions plus real answers | Not applicable |

The 900-image test is a **historical, repeatedly examined set**, not a pristine unseen benchmark. Exact-pixel separation from the 100 controller inputs does not establish independence from all earlier research.

## 4. Hybrid initialization and cleanup

### 4.1 Selected model

`models/active/hybrid/hybrid.pt` is copied from `E:/pranav/Final sem/sourcefree_synth/runs/iter2_baseline_15_20261003_04/baseline_training/epoch_15/ema.pt`. The exact SHA256 appears in the generated provenance appendix below. This identifies the deployed model more reliably than informal names.

For each image the hybrid emits a structured description: length, font category/stroke/italic/weight/width/x-height/slant, content category, effect presence/severity and text-feature presence/severity. The canvas size comes from the image. The architecture contains a CTC branch, but this controller workflow does not decode it into target text. Raw descriptions and non-CTC logits are retained in `capri_darkbay_workflow/raw_hybrid_predictions.json`.

### 4.2 Deterministic cleanup

`pipeline.clean` resolves conflicting active heads using the highest predicted probability. Text groups are digits/uppercase/lowercase/mixed alphanumeric, tight/wide spacing, condensed/expanded, and two-layer/curved text. Effect groups include gradient/texture/pattern, hollow/bold/thin, blur/pixelate/resample, and wave/morph/perspective/swirl. These are engineering simplifications, not evidence that combinations never occur in nature.

Inactive severity levels become zero; active levels are clamped to 1–10. Initial recipe conversion uses `feature_bank.iter2.controller.initial_recipe`. Renderer ranges are clamped: rotation 0–45, jitter 0–0.2, size jitter 0–0.3, spacing −0.35–0.4, x-scale 0.4–1.6 and curve 0–0.3. Every raw-to-cleaned change is recorded in `refined_predictions.json`, and all 100 initial recipes in `refined_recipes.json`.

### 4.3 Model-derived length

| Predicted length | Votes out of 100 |
|---|---:|
| 2 | 1 |
| 3 | 0 |
| 4 | 1 |
| 5 | 22 |
| 6 | 72 |
| 7 | 3 |
| 8 | 1 |

A predeclared 60/100 majority rule therefore selected length **6**. This is model agreement, not 72% calibrated confidence or measured length accuracy. The chosen length is applied to all generation recipes. The requested future no-majority behavior—equal weight across lengths—is not implemented in this run; current code stops when no majority exists.

## 5. Three controllers and renderer refinement

### 5.0 Upstream training of the frozen scheme models

The simple hybrid's saved configuration specifies 100,000 synthetic training images, 15 epochs, batch128, learning rate0.0005, minimum learning rate0.00001, weight decay0.05, warmup300 steps, gradient clip1, EMA0.999, level balancing and fixed loss weighting. Transformer settings are width160, five layers, four heads, MLP640, effect width128, dropout/drop-path0.1. Its data manifest also records2,000 calibration,2,000 selection,2,000 steering,2,000 seen-test and2,500 generalization images (110,500 unique total). These are upstream feature-model datasets, not the later DarkBay reader datasets. The selected epoch15 EMA was already trained before this controller workflow.

The neural feature head inventory is 37 effect presence heads,37 effect intensity heads,16 text presence heads,nine supported text intensity outputs,seven font attributes and length; a separate CTC branch is not decoded here. The appendix lists names from the saved inference record. Not every named text feature has a separately supported intensity classifier.

The custom encoder was trained in `learned_controller_20261004/train.py`:7,168 synthetic training images and1,024 validation images, paired renders, seed719,24 epochs, AdamW learning rate0.001 and weight decay0.0001. Loss combines continuous-attribute regression, categorical classification, mask BCE/Dice and paired contrastive embeddings; gradients clip at5 and learning rate drops by0.35 at epochs12/20. Synthetic validation selected `encoder.pt`. A separate300,449-parameter adjustment-policy network was trained in that exploratory directory, but the active CAPRI search does **not** load that policy checkpoint.

The active mask model was trained separately on4,096 synthetic images with512 validation images and disjoint saved train/validation font lists, seed63219,20 epochs, AdamW learning rate0.001/weight decay0.0001 and BCE-plus-Dice loss. It was selected by best mean synthetic validation IoU. Target images used for that training:zero. Its mask definition is visible original glyph ink after effects/occlusion, excluding decorative shadows and outline expansion. The recorded groups include random effects, hollow/interference, crossing interference and multicolor. Synthetic IoU does not establish real-mask accuracy. DINO uses the existing official pretrained checkpoint and was not retrained on DarkBay.

All arms pursue resemblance of both text and background while retaining render completeness/usability. These are search procedures using frozen comparators, not three independently trained end-to-end adjustment networks. Their mixture preserves different choices instead of retaining only a global winner.

### 5.1 Custom controller

Uses a frozen custom appearance encoder and synthetic-trained soft text-mask model. Target and synthetic images pass through the same inferred-mask path. No real character boxes are supplied. Pixel descriptors measure text color, layout, glyph shape, ink topology/stroke survival, background color/structure and whole-image embedding. Internal comparison size is 256×64.

Appearance weights: glyph shape 0.25; ink shape 0.20; layout 0.10; text color 0.10; background structure 0.20; background color 0.10; whole-image embedding 0.05. Feature scales use target standard deviations with floor 0.025 and baseline energy-distance scales with floor 0.05. The distribution distance is the nonnegative energy-distance expression normalized by square root of feature dimension. Strict readability is reported separately from the acceptance guard.

### 5.2 DINO controller

Uses official **original DINO ViT-S/8**, not DINOv2 or DINOv3. Frozen patch embeddings and final-layer CLS attention provide whole-image, focus and context features. Attention is saliency, not calibrated text segmentation. RGB residuals at multiple scales supplement attention, retaining colored text that may have low grayscale contrast; tiny/line-like connected regions are downweighted.

Final appearance weights: global embedding 0.10, focus embedding 0.175, context embedding 0.125, attention layout 0.05, region colors 0.05, glyph shape 0.12, ink shape 0.12, layout 0.08, background structure 0.12 and clarity 0.06. A contrast-collapse guard penalizes renders below a target-derived contrast floor (65% of mean target contrast, with numeric floor in the denominator).

### 5.3 Mixed controller

Combines normalized custom and DINO appearance losses equally. Calibration uses baseline renders of the first 24 hybrid recipes, with minimum scale 0.05. It adds DINO's contrast guard. All arms add `4 × (1 − complete_fraction) + 2 × (1 − usable_fraction)`. Mixed adds no separate trained checkpoint.

For each arm, the candidate score is 50% comparison against the corresponding reference image and 50% comparison against the full 100-image target distribution. There is no target-string match objective, pixel-aligned character reconstruction or known-font lookup from real labels.

### 5.4 Search and selection

For each of the 100 initialized recipes:

1. Estimate foreground/background palettes from the image and inferred mask. Quantize to five colors; exclude candidate ink colors too close to background using RGB distance 60, with a fallback when needed.
2. Propose neutral controls, pixel-derived colors, alternating/random/single color modes, background-pool variations, size values 0.50/0.65/0.80, widths 0.75/1.10, effects-off, zero rotation/jitter and an outline variant.
3. Evaluate each unique control candidate on two fresh synthetic strings/seeds. Never substitute the target answer.
4. Mutate each arm's leader: rotation 5/20, spacing 0/0.15, smaller size/jitter and single-color alternatives. Candidates are pooled, but each arm ranks by its own objective.
5. Take the union of each arm's top three candidates and validate on three new render seeds against the same references. Require every validation render to be complete and usable.
6. Each arm selects its own lowest validation-score candidate. Generic recovery is available if none passes, but the saved DarkBay pilot used **zero recovery entries**.
7. Produce one pilot image per selected entry, with logged retries if required.

Recipe initialization seeds use `1100003 + image_index × 7919`; search render seeds start at 1200003, validation at 1300003 and pilot at 1400003 with the exact per-index offsets in `pipeline.py`. Full candidates, scores and selected controls are preserved in `controller_history.json` and `entries.json`; these files, rather than a shortened prose description, specify all 300 settings.

### 5.5 Diversity actually achieved

100 custom + 100 DINO + 100 mixed = **300 entries**, but only **179 parameter signatures** after excluding text and seed. On the same reference image, custom and DINO match for 31/100 choices; mixed matches custom for 67 and DINO for 54. These overlaps are not disjoint categories. Parameter signatures also need not equal perceptually distinct families. All 300 pilot images met basic guards; the workflow does not claim 300 independent CAPTCHA libraries.

## 6. Dataset 1 generation

Name: **Darkbay_ds1_noFont-intel**. Uses the frozen 300 entries in `capri_darkbay_workflow/entries.json`. Assign 333 or 334 images per entry; seeded remainder allocation (1052026) yields custom 33,334, DINO 33,333 and mixed 33,333.

For every output, the renderer draws a new synthetic string according to the recipe's content category and model-derived length. Its saved string is the training label. It does not copy the corresponding real string. Seed formula: `2,000,000,000 + entry_position × 10,000,000 + output_index × 10,000 + retry_index`. Up to 100 retries remain within the same entry. Basic acceptance requires completion, usability and correct generated length; within-entry duplicate RGB renders are rejected. The global audit checks uniqueness across entries too.

Storage: 300 ZIP shards, each with PNGs, `labels.jsonl` and `rejections.jsonl`. Metadata preserves renderer settings, seed, entry/controller, originating reference index, RGB hash, dimensions, attempts and complete/usable/strict flags. Export to lossless LMDB preserves labels and source provenance.

Audit: 100,000 distinct RGB images and seeds, no copied target pixels, balanced arms. Only **36,253/100,000** pass the stricter readability guard; these were **not** the only training samples. Low local character contrast was a major pilot failure mode. A rendered label may be syntactically correct while its image is hard to read. Strict readability is a heuristic, not human verification.

## 7. Reader architecture and training-methodology source

The user requested the newer methodology from `E:/pranav/Final sem/CaptchaThesis`. The saved DS1 protocol identifies `inputs/configs/parseq.yml`, `experiments/six_system_benchmark/protocol.py` and `train_str.py`, with source hashes. The newer 20-epoch/three-seed/final-checkpoint protocol was adapted to PARSeq; the source benchmark's primary model was NRTR. This is an adaptation, not an exact reproduction of every thesis setting.

Reader: OpenOCR **SVTRv2LNConvTwo33 encoder + PARSeq decoder**, 20,076,927 parameters. Encoder stage dimensions 128/256/384, depths 6/6/6, attention heads 4/8/12, convolution/global mixer configuration in saved YAML. Autoregressive decoding with one refinement iteration. Charset is 62 case-sensitive symbols `0–9 A–Z a–z`, spaces disabled, maximum decoder text length 25. Generation length six is not imposed at reader inference.

Common settings: seeds 48/123/456; batch size 128; AdamW learning rate 0.00035 and weight decay 0.05; OneCycleLR with two warmup epochs and cycle momentum disabled; gradient clipping 20; PARSeq loss; mixed precision BF16 during training. Preprocessing uses RGB, 32×128 bicubic resize, tensor conversion and normalization with mean/std 0.5. Windows workers are zero. Exact optimizer filtering, sampler and transforms are in each seed YAML.

All readers were initialized **from scratch**. The requested pretrained checkpoint was unavailable when Model 1 was launched; this departure from the pretrained thesis recipe was disclosed. Model 2 did not resume or fine-tune Model 1 weights. Model 1 was its difficulty teacher only.

## 8. Model 1 training and testing

All 100,000 DS1 images entered optimization. There was no separately constructed validation dataset and no training-time real evaluation. No augmentation was applied. Each seed trained 20 epochs, and `latest.pth` at epoch20 was the predefined evaluation checkpoint. Per-epoch files exist for epochs2–20; epoch1 was not retained separately. Do not infer a validation score from the framework's placeholder `best_acc = 0`.

Training accuracy was not recorded (`cal_metric_during_train: false`). Saved losses are last logged smoothed batch losses, not full-epoch averages. Very low training loss does not establish real generalization; the appendix records all saved DS1 loss values.

Evaluation: the same900 RGB-unique images remaining after excluding controller references; no exact synthetic overlap; case-sensitive exact whole-string match plus aggregate character error rate. Labels from filenames were read after all seed prediction files were hash-frozen. One real label has seven characters (`0UHHAA1`) and was retained; no correction to six characters was made.

| DS1 reader | Correct / 900 | Accuracy |
|---|---:|---:|
| Seed48 epoch20 | 492 | 54.67% |
| Seed123 epoch20 | 439 | 48.78% |
| Seed456 epoch20 | 474 | 52.67% |
| Prediction ensemble | 496 | 55.11% |

The ensemble chooses whole-string majority. If all three disagree, it chooses the prediction with highest saved mean character confidence; exact ties preserve seed order48/123/456. This is not character-by-character voting. Straight equal weight averaging of independently initialized readers was also tested and scored0/900; that failed artifact was retained. Prediction ensembling was used instead.

A subsequent diagnostic evaluated57 saved checkpoints (epochs2–20 × three seeds) and19 ensembles on the historical real set. The best observed ensemble was epoch19,502/900 (55.78%), versus final epoch20,496/900. This was retrospective analysis, not replacement of the predefined final model. It does not prove a common overfitting onset; real scores fluctuate, and DS1 had no contemporaneous synthetic validation curve.

## 9. Dataset 2 and difficulty ranking

Name: **Darkbay_ds2_noFont-intel**. Reuses DS1's **same frozen300 recipes and allocation**, rather than fitting controllers again. Render seed base changes to10,000,000,000; exact seed formula otherwise stays the same. Audit found100,000 unique RGB images, zero exact DS1 overlap and zero target overlap. Strict-readability count36,738;57 rejected render attempts were logged. Fresh pixels do not imply independently fitted target families or no repeated strings.

Difficulty teachers are the three DS1 **epoch20** checkpoints. Each independently decodes the synthetic image. For ranking, take the equal arithmetic mean of their per-position character distributions, exclude EOS and renormalize the character distribution, then sum negative log probabilities assigned to the known synthetic answer's characters. There is no teacher forcing. Ground-truth synthetic labels are appropriate here; real labels are not used. Saved free predictions/confidences are separate from this NLL.

Tertile cutoffs: approximately0.00001852260708 and0.00008649211287. Easy uses the lower third, medium the next third, hard the rest. Counts: easy33,334, medium33,333, hard33,333. Teacher whole-string ensemble accuracy across the100k is95.267%; easy100%, medium100%, hard85.80%. Difficulty can distinguish confidence even when the decoded string is correct. No genuine errors were required for assignment to medium.

Hold out5,000 synthetic images using seed62026, stratified by bucket and controller. Remaining95,000 form the optimization pool. Validation IDs never enter the training sampler. Exact membership is in `curriculum_samples.json`; manifests and LMDB hashes are saved.

## 10. Model 2 curriculum, augmentation and checkpoint selection

The curriculum was adapted from `E:/pranav/model 2/configs/parseq_arc_m2_teacher_v2.yaml` and its training implementation/results. That earlier setup used15 epochs; this run retains the newer20-epoch reader schedule and architecture.

| Epochs | Easy | Medium | Hard | Draws per epoch |
|---|---:|---:|---:|---:|
| 1 | 100% | 0% | 0% | 95,000 |
| 2 | 50% | 50% | 0% | 95,000 |
| 3–20 | 15% | 25% | 60% | 95,000 |

Sampling is with replacement, so “95,000 draws” is not one pass over95,000 distinct samples. The final partial minibatch is dropped. Per-epoch bucket receipts are saved. Seeds are trained sequentially from scratch; all completed20 epochs.

Training-only augmentation reproduces the earlier Model2v2 operations with seeded Albumentations: color jitter (brightness/contrast0.25, saturation0.25, hue0.08, probability0.7); a blur choice (Gaussian/motion/median, overall0.4); Gaussian noise0.3; JPEG-like compression quality50–95 with probability0.4; affine rotation±3°, translation x±2%/y±4%, scale0.95–1.05, x-shear±3°, white fill, probability0.6; coarse dropout1–3 holes of height2–8 and width2–12, white fill, probability0.3. Validation and test use unaugmented deterministic preprocessing.

After each epoch, compute exact whole-string accuracy on5,000 synthetic validation images. Select the best synthetic-validation checkpoint separately for each seed; preserve epoch20 too. All60 epoch files were verified after completion. Validation affects selection, not gradients. Real interim evaluations requested by the user did not alter training, curriculum or selection.

| Seed | Selected epoch | Synthetic validation | Real correct /900 | Real accuracy |
|---|---:|---:|---:|---:|
| 48 | 18 | 96.64% | 566 | 62.89% |
| 123 | 20 | 96.62% | 520 | 57.78% |
| 456 | 18 | 96.58% | 524 | 58.22% |
| Ensemble of selected checkpoints | 18/20/18 | Not separately computed here | 565 | 62.78% |

This final ensemble is **not an all-epoch20 ensemble**. Final real scoring uses the same900-image manifest and decoding as DS1 and the same whole-string ensemble rule. All four final prediction hashes and correct counts were independently checked. An ensemble can perform slightly worse than its strongest member; majority voting does not guarantee improvement.

The shared epoch19 ensemble was separately tested at567/900 (63.00%), compared with DS1 epoch19's502/900. The best requested single-seed interim score was seed48 epoch15 at607/900 (67.44%). Neither retrospective real result replaced synthetic-based final selection. Only the listed interim checkpoints were evaluated; a full DS2 all-epoch real sweep has not been completed.

## 11. Shoot-rate analysis: definitions and the corrected procedure

Accuracy among attempts = correct answers / attempted CAPTCHAs. Shoot rate = attempted CAPTCHAs / all900 real images. Skipped images are excluded from attempted accuracy but remain in the denominator of shoot rate. Synthetic accuracy goals and heuristic thresholds are different quantities.

### 11.1 Earlier confidence-only sweeps

The original DS1 analysis and the automated DS2 final evaluator swept fixed joint-confidence thresholds directly on real predictions. Per-reader joint confidence is the product of emitted-character probabilities, excluding EOS. Ensemble joint support sums the joint confidences of readers agreeing with the winning string and divides by3. No-filter results remain496/900 and565/900 respectively.

For example, at threshold0.99 the DS1 sweep gave211/269 (78.44%,29.89% shoot); DS2 gave178/209 (85.17%,23.22% shoot). These are preserved **historical real-confidence diagnostics**, not the synthetic-derived character-confusion procedure the user subsequently requested. They must not be labeled as synthetic-calibrated thresholds or mixed into the following tables.

### 11.2 Synthetic discovery and calibration

Both readers' corrected policy uses the same DS2 validation5,000 images: deterministic shuffle seed6202601, first2,500 for discovery, remaining2,500 for calibration. Partitions are disjoint and identical between Model1 and Model2. DS1 never trained on these pixels. DS2 excluded them from weight updates but also used the full5,000 for checkpoint selection, so its calibration is not independent of model selection.

For DS1, use its saved epoch20 teacher predictions. For DS2, rerun its selected18/20/18 checkpoints on the original unaugmented validation pixels with FP32, batch64 and the same real-test decoder. Freeze each prediction file with checkpoint and prediction hashes.

Build the usual winning-string ensemble. Its **support** for this method is the sum of saved **mean character confidences** from readers agreeing with the winning string, divided by3. This differs from the earlier joint-confidence sweep.

On discovery only, align each predicted string to its synthetic answer using Levenshtein edit operations. Count correct emitted characters, substitutions and insertions; record deletions separately. Reliability for an emitted character is `(correct_count + 1) / (emitted_count + 2)`; unseen characters default to0.5. Empty strings score0. The deployment score is:

`support × product(reliability of each emitted character)`

This is a heuristic risk ranking, not a calibrated probability. Pairwise confusion counts are recorded for analysis; the deployed rule uses emitted-character reliability rather than changing characters. Synthetic true-answer NLL/p_correct is never an inference feature.

For each predeclared goal90/95/98/99%, search calibration scores for the threshold retaining the most images while retaining at least100 attempts and a Wilson95% lower accuracy bound at or above the goal. Equal-score ties are kept together. These bounds are descriptive after threshold search, not formal post-selection guarantees.

Freeze the policy before loading real prediction files. Compute real scores and attempt/skip decisions without labels and hash-freeze those decisions. Only then load real answers to count correctness. No real-guided threshold adjustment, string repair, case normalization or forced length is performed. The real set had already been examined historically; the freeze concerns this new rule, not a claim nobody previously saw the answers.

### 11.3 Model 1 corrected results

DS1 discovery was2387/2500 (95.48%), and calibration baseline2375/2500 (95.00%).

| Synthetic goal | Score threshold (rounded) | Synthetic correct / attempts | Real correct / attempts | Real accuracy | Real shoot rate |
|---|---:|---:|---:|---:|---:|
| 90% | 0.241105 | 2375/2500 | 496/900 | 55.11% | 100.00% |
| 95% | 0.300290 | 2370/2472 | 494/884 | 55.88% | 98.22% |
| 98% | 0.625424 | 2286/2319 | 353/543 | 65.01% | 60.33% |
| 99% | 0.919712 | 1611/1619 | 232/339 | 68.44% | 37.67% |

### 11.4 Model 2 corrected results

Discovery2428/2500 (97.12%), calibration baseline2429/2500 (97.16%). The90% and95% goals select the same rule.

| Synthetic goal | Score threshold (rounded) | Synthetic correct / attempts | Real correct / attempts | Real accuracy | Real shoot rate |
|---|---:|---:|---:|---:|---:|
| 90% | 0.230200 | 2429/2500 | 565/900 | 62.78% | 100.00% |
| 95% | 0.230200 | 2429/2500 | 565/900 | 62.78% | 100.00% |
| 98% | 0.586452 | 2409/2444 | 535/772 | 69.30% | 85.78% |
| 99% | 0.873029 | 2312/2325 | 411/530 | 77.55% | 58.89% |

Actual decisions use full-precision thresholds in JSON; do not rerun rounded table values expecting exact boundary counts. Thresholds are separately calibrated for each model. Model2's strictest rule attempts more and has higher observed real precision than Model1's, but neither reaches its99% synthetic goal on real images. This is material synthetic-to-real distribution shift.

## 12. External GLM-OCR benchmark

GLM-OCR is a VLM, explicitly requested as an external baseline after the no-VLM CAPRI design. It is not used to generate recipes, label real references, rank DS2 or train readers. Official `zai-org/GLM-OCR`, revision `2e85a62840ccac27daa451df36c736c4636b8628`, is stored in `models/external/GLM-OCR`. Its published size is0.9B parameters; this is the advertised size, not an independent parameter-count audit here. Our single reader is20.08M; all four active scheme-model checkpoints total26.77M, giving46.85M plus one reader or87.00M plus three readers, without double-counting mixed's shared weights.

GLM uses original RGB, official processor, prompt `Text Recognition:`, BF16 SDPA, greedy decoding and max64 new tokens. Only edge whitespace is stripped, no internal-space/case/length correction. Original batch1 real run:645/900 (71.67%), approximately295 seconds. The later batch8 confidence run also scored645/900 but changed28 individual predictions, so the runs are separately preserved.

Its synthetic policy used the same2500/2500 partition and emitted-character reliability mechanism, multiplied by joint generated-token confidence (special tokens/EOS excluded). Token probabilities are not directly comparable to PARSeq character probabilities. Discovery1639/2500; calibration1588/2500. Only the90% goal met the saved support/bound criterion: synthetic695/754 (92.18%,30.16% shoot), threshold0.5500554611137143, then real160/174 (91.95%,19.33% shoot). Goals95/98/99 were unavailable, not zero-accuracy policies. Six synthetic generations reached the token limit; none real. Unknown pretraining overlap and different resource/training budgets limit fairness of comparison.

## 13. Limitations and conclusions supported by the evidence

- Model2 improved the recorded historical DarkBay ensemble test, but no controlled ablation attributes the gain to a single change.
- Pixel-only target conditioning does not mean every component was trained without labels: the synthetic-trained hybrid/mask/encoder have their own upstream supervision, and DINO has external pretraining.
- The100 real references and recipe bank are reused for both datasets. Recipe selection validates fresh render seeds against the same references, not new target-family images.
- Length6 is model-derived, but character recognition remains unconstrained at test time. Majority agreement does not prove length prediction correctness.
- The renderer's font bank and generic typography controls are active; dedicated font intelligence was not integrated. No exact target-font recovery claim is supported.
- Many training renders fail the stricter readability heuristic. Basic guards and labels do not establish human legibility.
- DS1 has no training-time validation and no recorded training accuracy. DS2 validation selects checkpoints and is reused for policy calibration; a new independent calibration set would give cleaner evidence.
- Real tests are historical and repeatedly viewed. They were not used to adjust the saved training or policy rules, but are not an untouched final benchmark.
- Synthetic rejection precision does not transfer directly. Report both correct/attempted and attempted/total.
- Prior Meta-Sim2, SAM, DINOv3 and random-start controller experiments were exploratory; they are not selected in the final DS1/DS2 recipe pipeline. This document does not retrofit those approaches into the completed datasets.

## 14. Reproduction and preservation

Use the recorded old environment for readers: `E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe`. Despite its directory name, the active workflow is not XGBoost. OpenOCR source is `E:/pranav/Final sem/Models/level 1/upstream/OpenOCR`; local package overlay is `.runtime/parseq_packages`. Renderer source is `E:/pranav/Final sem/captcha-effects`. These external dependencies mean this workspace is not a standalone portable release.

Inspect before running. Existing output directories are preserved and several scripts intentionally reject overwrite. Commands describe entry points, not authorization to restart completed runs:

```powershell
$capriPython = 'E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe'
& $capriPython capri.py verify
& $capriPython capri.py controllers --out runs/new_controller_run
& $capriPython capri.py generate --run runs/new_controller_run --workers 8
# Completed DS2 orchestration entry point; do not rerun over existing evidence:
# & $capriPython capri.py darkbay-ds2
# Individual checkpoint diagnostic:
# & $capriPython tools/evaluate_ds2_checkpoint.py --seed 48 --epoch 17
# Completed policy entry points; preserve existing output directories:
# & $capriPython tools/synthetic_shoot_policy.py
# & $capriPython tools/ds2_synthetic_shoot_policy.py
```

The completed DS2 run ended at06:35:58 local on6 October2026. All60 epoch files, selected/final checkpoint hashes, validation records, dataset hashes and final prediction hashes/counts were checked in `completion_verification.json`. The hourly training monitor was paused after completion. Active code/model organization, original checkpoint copies, historical experiments and frozen artifacts remain preserved. This document is a snapshot through the completed Model2 synthetic-calibrated analysis; it does not imply later experiments have been run.

## 15. Deferred differences for a later loop

The user subsequently clarified that these changes are deferred beyond Model 3. Model 3 changes only the curriculum and reuses Model 2's exact dataset, 95k/5k split and other training settings. The items below remain later-loop plans, not changes to the next run or completed results. The current Model 2 run remains final: it optimized on 95,000 images and used 5,000 for validation. The intended allocation was 100,000 optimization images plus a separate validation set; the completed run's smaller optimization pool is retained and disclosed rather than retrospectively relabeled.

| Aspect | Completed Model 2 run | Later loop (deferred) |
|---|---|---|
| Font intelligence | Not integrated | Integrate font intelligence into the scheme-generation workflow |
| Validation generation | Held out from the same three-controller recipe mixture | Add another controller dedicated to generating validation data |
| Training allocation | 95,000 images | 100,000 training images, all eligible for optimization |
| Validation allocation | 5,000 taken from the 100,000-image corpus | A separate 5,000-image validation set, additional to the 100,000 training images |

For that later loop, the planned total is 105,000 synthetic images: 100,000 for training and 5,000 for validation. Validation images must stay outside weight updates. The validation controller's architecture, objective and recipe-selection procedure, and the exact font-intelligence implementation, are not yet specified or implemented. A different controller alone does not prove statistical independence; preserve separate generation provenance and check image overlap.

No retraining, new controller construction or dataset regeneration is being launched as part of this methodology update. Existing numerical results throughout this document describe the completed run. The later sections' evidence tables remain historical records.

## 16. Machine-generated evidence appendices

The following appendices are extracted from saved JSON/CSV/log artifacts to avoid transcription errors. Source paths and hashes are included for verification.

### A. Active checkpoints and feature names

- **hybrid**: [models/active/hybrid/hybrid.pt](C:/Users/pranav/Documents/ChatGPT/thesis/models/active/hybrid/hybrid.pt); SHA256 `ae73375ead99071d55c6394f7c4eb7c4d27c0a3285f8f15d8d84c9df43bc6b32`. Simple hybrid: initial labels and length votes; CTC is not decoded.
- **custom_encoder**: [models/active/custom_encoder/custom_encoder.pt](C:/Users/pranav/Documents/ChatGPT/thesis/models/active/custom_encoder/custom_encoder.pt); SHA256 `0dfac96c3b903cab88bddcfe28dde69550fe53d7d43571a26eb2cd5bd7e3ba38`. Frozen custom appearance encoder; not an adjustment policy.
- **text_mask**: [models/active/text_mask/text_mask.pt](C:/Users/pranav/Documents/ChatGPT/thesis/models/active/text_mask/text_mask.pt); SHA256 `8d7747a31215371a3d0721536bb14e46ffb07cdd00798effeaf732f57785f33c`. Synthetic-trained text-region mask estimator.
- **dino**: [models/active/dino/dino.pth](C:/Users/pranav/Documents/ChatGPT/thesis/models/active/dino/dino.pth); SHA256 `55c8b267f479b614ec20858947ce361ba804846c193e20be74263a096bb6d75e`. Original DINO ViT-S/8, used by controller iteration v2. Not DINOv2 or DINOv3 pretrained weights.

**font**: `category`, `stroke`, `italic`, `weight`, `width`, `x_height`, `slant`.

**effects**: `bg_gradient`, `bg_image_texture`, `bg_blocks`, `bg_grid`, `bg_circles`, `bg_decoy_chars`, `border_frame`, `straight_lines`, `bands`, `curves`, `scribbles`, `dot_noise`, `blob_noise`, `bg_pattern`, `glyph_shadow`, `extrude_3d`, `glyph_outline`, `hollow_outline_glyphs`, `stroke_bold`, `stroke_thin`, `glyph_fill_texture`, `translucent_text`, `strike_through`, `noise_over_text`, `occlusion`, `glyph_cuts`, `wave_warp`, `morph_distortion`, `perspective`, `blur`, `local_warp`, `inverted_regions`, `pixel_noise`, `compression`, `pixelation`, `low_resolution`, `colour_quantization`.

**text_features**: `per_char_rotation`, `vertical_jitter`, `size_variation`, `text_on_curve`, `slant`, `overlap_tight_spacing`, `wide_spacing`, `condensed`, `expanded`, `font_mixture`, `multicolor_text`, `two_layer`, `digits_only`, `uppercase_only`, `lowercase_only`, `mixed_case_alnum`.

### B. Model 1 full saved epoch comparison

Correct whole strings out of 900; retrospective real-test diagnostics. Epoch 1 was not saved separately.

| Epoch | Seed 48 | Seed 123 | Seed 456 | Ensemble |
| --- | --- | --- | --- | --- |
| 2 | 295 | 252 | 223 | 297 |
| 3 | 213 | 358 | 338 | 347 |
| 4 | 299 | 282 | 247 | 317 |
| 5 | 367 | 311 | 284 | 372 |
| 6 | 350 | 291 | 332 | 381 |
| 7 | 297 | 354 | 324 | 366 |
| 8 | 371 | 375 | 358 | 431 |
| 9 | 473 | 377 | 365 | 464 |
| 10 | 426 | 413 | 380 | 458 |
| 11 | 401 | 404 | 403 | 445 |
| 12 | 406 | 406 | 447 | 459 |
| 13 | 478 | 409 | 364 | 463 |
| 14 | 506 | 455 | 372 | 474 |
| 15 | 453 | 411 | 467 | 469 |
| 16 | 488 | 430 | 458 | 492 |
| 17 | 494 | 426 | 461 | 490 |
| 18 | 479 | 433 | 475 | 499 |
| 19 | 493 | 437 | 477 | 502 |
| 20 | 492 | 439 | 474 | 496 |

### C. Model 1 logged losses

Last logged smoothed batch losses, not full-epoch averages or training accuracy.

| Epoch | 48 | 123 | 456 |
| --- | --- | --- | --- |
| 1 | 2.541277 | 3.150942 | 2.767054 |
| 2 | 0.240563 | 0.252018 | 0.264159 |
| 3 | 0.16146 | 0.143008 | 0.145205 |
| 4 | 0.128703 | 0.141281 | 0.115529 |
| 5 | 0.083597 | 0.11929 | 0.083034 |
| 6 | 0.084118 | 0.064469 | 0.071663 |
| 7 | 0.06328 | 0.05568 | 0.062717 |
| 8 | 0.05302 | 0.051961 | 0.044791 |
| 9 | 0.041394 | 0.036016 | 0.033591 |
| 10 | 0.038472 | 0.019394 | 0.027793 |
| 11 | 0.019394 | 0.018329 | 0.020327 |
| 12 | 0.014221 | 0.010557 | 0.010771 |
| 13 | 0.009994 | 0.006864 | 0.009456 |
| 14 | 0.0047 | 0.004363 | 0.005715 |
| 15 | 0.003825 | 0.003054 | 0.002019 |
| 16 | 0.000669 | 0.001635 | 0.000792 |
| 17 | 0.000458 | 0.000546 | 0.000362 |
| 18 | 0.000382 | 0.000239 | 0.00033 |
| 19 | 0.000277 | 0.000219 | 0.000199 |
| 20 | 0.000265 | 0.000201 | 0.000162 |

### D. Model 2 complete synthetic validation accuracy

| Epoch | 48 | 123 | 456 |
| --- | --- | --- | --- |
| 1 | 0.00% | 0.00% | 0.00% |
| 2 | 68.42% | 67.74% | 68.24% |
| 3 | 82.52% | 82.62% | 81.26% |
| 4 | 87.56% | 87.26% | 85.48% |
| 5 | 88.82% | 88.08% | 89.06% |
| 6 | 90.84% | 90.52% | 91.12% |
| 7 | 92.32% | 91.30% | 92.02% |
| 8 | 92.96% | 92.82% | 93.10% |
| 9 | 93.10% | 93.66% | 93.14% |
| 10 | 94.36% | 94.12% | 93.88% |
| 11 | 94.20% | 94.92% | 94.74% |
| 12 | 95.06% | 94.86% | 95.30% |
| 13 | 95.56% | 95.64% | 95.58% |
| 14 | 96.04% | 95.68% | 95.66% |
| 15 | 96.00% | 95.70% | 95.78% |
| 16 | 96.32% | 96.24% | 96.26% |
| 17 | 96.16% | 96.30% | 96.32% |
| 18 | 96.64% | 96.48% | 96.58% |
| 19 | 96.46% | 96.62% | 96.54% |
| 20 | 96.50% | 96.62% | 96.56% |

### E. Model 2 saved interim real tests

| Epoch | Seed | Model 2 correct /900 | Model 1 same epoch correct /900 |
| --- | --- | --- | --- |
| 5 | 48 | 427 | 367 |
| 6 | 48 | 432 | 350 |
| 8 | 48 | 485 | 371 |
| 9 | 48 | 517 | 473 |
| 10 | 48 | 548 | 426 |
| 12 | 48 | 526 | 406 |
| 13 | 48 | 564 | 478 |
| 15 | 48 | 607 | 453 |
| 16 | 48 | 560 | 488 |
| 17 | 48 | 577 | 494 |
| 19 | 48 | 568 | 493 |
| 19 | 123 | 514 | 437 |
| 19 | 456 | 535 | 477 |

Shared epoch19 ensemble: DS2 567/900 versus DS1 502/900. Missing DS2 epochs have not been exhaustively evaluated.

### F. Model 2 logged losses

Last logged smoothed batch loss per epoch; not full-epoch averages.

| Epoch | 48 | 123 | 456 |
| --- | --- | --- | --- |
| 1 | 2.971467 | 2.915630 | 2.952281 |
| 2 | 0.332601 | 0.327713 | 0.324861 |
| 3 | 0.491518 | 0.481538 | 0.450161 |
| 4 | 0.362023 | 0.384578 | 0.394981 |
| 5 | 0.329839 | 0.342087 | 0.334124 |
| 6 | 0.280075 | 0.265641 | 0.298441 |
| 7 | 0.231444 | 0.267491 | 0.244232 |
| 8 | 0.214171 | 0.247653 | 0.236959 |
| 9 | 0.206445 | 0.209955 | 0.234311 |
| 10 | 0.177012 | 0.213932 | 0.188229 |
| 11 | 0.177202 | 0.178385 | 0.167474 |
| 12 | 0.165019 | 0.153015 | 0.163674 |
| 13 | 0.165875 | 0.142013 | 0.159622 |
| 14 | 0.129592 | 0.122421 | 0.143452 |
| 15 | 0.127462 | 0.095017 | 0.118428 |
| 16 | 0.141132 | 0.108824 | 0.135678 |
| 17 | 0.112637 | 0.106652 | 0.121155 |
| 18 | 0.119583 | 0.102474 | 0.134514 |
| 19 | 0.115317 | 0.122680 | 0.111486 |
| 20 | 0.102086 | 0.111073 | 0.119922 |

### G. Original DS2 real-confidence sweep

This is not the synthetic-calibrated policy.

| Joint threshold | Correct / attempted | Accuracy | Shoot rate |
| --- | --- | --- | --- |
| 0.99 | 178/209 | 85.17% | 23.22% |
| 0.95 | 278/332 | 83.73% | 36.89% |
| 0.9 | 319/390 | 81.79% | 43.33% |
| 0.85 | 348/432 | 80.56% | 48.00% |
| 0.8 | 380/479 | 79.33% | 53.22% |
| 0.7 | 406/521 | 77.93% | 57.89% |
| 0.6 | 469/611 | 76.76% | 67.89% |
| 0.5 | 508/693 | 73.30% | 77.00% |
| 0.4 | 532/765 | 69.54% | 85.00% |
| 0.3 | 549/827 | 66.38% | 91.89% |
| 0.2 | 564/889 | 63.44% | 98.78% |
| 0.1 | 565/900 | 62.78% | 100.00% |
| 0 | 565/900 | 62.78% | 100.00% |

### H. Artifact links and snapshot hashes

- [capri_darkbay_workflow/inputs.json](C:/Users/pranav/Documents/ChatGPT/thesis/capri_darkbay_workflow/inputs.json) — SHA256 `c1bbfa8048f0e9b7cfa21fecf88a6031679b565a8f29a03614d52fa34a44f4fc`
- [capri_darkbay_workflow/entries.json](C:/Users/pranav/Documents/ChatGPT/thesis/capri_darkbay_workflow/entries.json) — SHA256 `79e1f11cc462c0f3c16ca4425c6a5d993bdb49b8466fbbef7d27aaa3289f3814`
- [capri_darkbay_workflow/controller_history.json](C:/Users/pranav/Documents/ChatGPT/thesis/capri_darkbay_workflow/controller_history.json) — SHA256 `9024f8122d05a897ef54d487d4ed51a776b6ae8d39e43fb1425d3f528b9cabdd`
- [capri_darkbay_workflow/corpus_audit.json](C:/Users/pranav/Documents/ChatGPT/thesis/capri_darkbay_workflow/corpus_audit.json) — SHA256 `9be5a4b423106477e3ced94b4b9cdb110bfc7576b895c61d54613fa45a70ce6b`
- [capri_darkbay_workflow/refined_predictions.json](C:/Users/pranav/Documents/ChatGPT/thesis/capri_darkbay_workflow/refined_predictions.json) — SHA256 `d1c9ce5925ade1698676336cbb3f878efff647670eb60d0930f6fda421fa3dab`
- [capri_darkbay_workflow/refined_recipes.json](C:/Users/pranav/Documents/ChatGPT/thesis/capri_darkbay_workflow/refined_recipes.json) — SHA256 `886327c1d5075750d5953075e6ffa33c853c3abf09d33691a32d396e43233169`
- [models/registry.json](C:/Users/pranav/Documents/ChatGPT/thesis/models/registry.json) — SHA256 `2eae8b8c6b4942814358b49e51fe05c539bca478e2d031d4f219fb89464a00d9`
- [configs/local.json](C:/Users/pranav/Documents/ChatGPT/thesis/configs/local.json) — SHA256 `64940a55f665cc8b84378cbd06f5f6901b47d9666f1cc153f27491f6449203a4`
- [runs/parseq_darkbay_100k_20261005/protocol.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/parseq_darkbay_100k_20261005/protocol.json) — SHA256 `573c8ce3dd8f139ce4043d559757312c24fa682ca9c546463709bc990179361e`
- [runs/parseq_darkbay_100k_20261005/completion.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/parseq_darkbay_100k_20261005/completion.json) — SHA256 `5f2d057589d767c531b3ceae9aec5507dd4b7bf7d5cb7ed13382f1a19f37a2c5`
- [runs/parseq_darkbay_100k_20261005/synthetic_validation_shoot_policy/frozen_policy.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/parseq_darkbay_100k_20261005/synthetic_validation_shoot_policy/frozen_policy.json) — SHA256 `2e8ec9ba7dde8b2b1d0411d399d7621346a40981a819d879a4bcb401df20593a`
- [runs/parseq_darkbay_100k_20261005/synthetic_validation_shoot_policy/summary.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/parseq_darkbay_100k_20261005/synthetic_validation_shoot_policy/summary.json) — SHA256 `0ec0b0cfc6d39e7ec17ce350ca0b743678cdc6eb082213be33c3d3d15c74ef1d`
- [runs/parseq_darkbay_100k_20261005/seed_48.yml](C:/Users/pranav/Documents/ChatGPT/thesis/runs/parseq_darkbay_100k_20261005/seed_48.yml) — SHA256 `3644765bf1ab2a59a6275a600b0c85268d3b406d630613984fab7a7b9c8af0ea`
- [runs/parseq_darkbay_100k_20261005/seed_123.yml](C:/Users/pranav/Documents/ChatGPT/thesis/runs/parseq_darkbay_100k_20261005/seed_123.yml) — SHA256 `368e6760fba9cc05053311471d74af2705b4a448c14d0643c0c65a62723274ae`
- [runs/parseq_darkbay_100k_20261005/seed_456.yml](C:/Users/pranav/Documents/ChatGPT/thesis/runs/parseq_darkbay_100k_20261005/seed_456.yml) — SHA256 `33076700f86d647f91b4e51a64cc2232791587067c15746e7114ca06304c3fe3`
- [runs/Darkbay_ds2_noFont-intel/protocol.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel/protocol.json) — SHA256 `ff3274f5f590308bd1c6c428b93b162fa5ca82be1727963da1179906607341ad`
- [runs/Darkbay_ds2_noFont-intel/completion.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel/completion.json) — SHA256 `25416951a5a438d5d4f8f92c1071392c6b8bc396677dbe4b0fe179a0e8676cbb`
- [runs/Darkbay_ds2_noFont-intel/synthetic_validation_shoot_policy/frozen_policy.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel/synthetic_validation_shoot_policy/frozen_policy.json) — SHA256 `270059e6c591932d89012d8a6b47d3af4a92f4bfffb539f6066df8a67bad2989`
- [runs/Darkbay_ds2_noFont-intel/synthetic_validation_shoot_policy/summary.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel/synthetic_validation_shoot_policy/summary.json) — SHA256 `75abf5174e16a3f33d1dada249ef8062959c0ee7bfc22dc68f08db91848d293c`
- [runs/Darkbay_ds2_noFont-intel/seed_48.yml](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel/seed_48.yml) — SHA256 `c51c3bcc3f16948cfc8b31c7212ce5c107148712ac858dc984edd8005112f338`
- [runs/Darkbay_ds2_noFont-intel/seed_123.yml](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel/seed_123.yml) — SHA256 `7e428b87e0d0faef36a99b6472f6e5c6d4cbede7b34ecd3ff502ba77753fef46`
- [runs/Darkbay_ds2_noFont-intel/seed_456.yml](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel/seed_456.yml) — SHA256 `662097f8f3ee631d7a453f2a36222396c06b60d3a43e418665695768e9c06256`
- [runs/Darkbay_ds2_noFont-intel/completion_verification.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel/completion_verification.json) — SHA256 `f224a17a1cafee293d7c4342921f2d221bc8dac50a08bc58827183760dcae473`
- [runs/Darkbay_ds2_noFont-intel/bucket_summary.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel/bucket_summary.json) — SHA256 `9eeb8cb3402bccf0f1fc647fd6a7bcbfbf66753d942c61286400a5a9493f6353`
- [runs/Darkbay_ds2_noFont-intel/evaluation_darkbay/summary.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel/evaluation_darkbay/summary.json) — SHA256 `f40744d4fe339533c6ab33309cfca95d5f73aae71327754369bc1fc95231473b`
- [runs/glm_ocr_synthetic_shoot_policy/summary.json](C:/Users/pranav/Documents/ChatGPT/thesis/runs/glm_ocr_synthetic_shoot_policy/summary.json) — SHA256 `218791677be933c708a8d0c15552f58d4fe534b792b2240d8f83ef359ef4037a`

The linked entries/history files preserve all 300 exact control settings and candidate-selection scores. They are part of this reproducibility record; the prose summarizes their common algorithm. Full historical DS1 confidence sweep: [runs/parseq_darkbay_100k_20261005/shoot_rate_analysis](C:/Users/pranav/Documents/ChatGPT/thesis/runs/parseq_darkbay_100k_20261005/shoot_rate_analysis).
