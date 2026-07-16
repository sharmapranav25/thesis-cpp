# Research notes — scene text removal literature

Phase 1 (character extraction) is the **text-localization stage** of the broader scene-text-removal task. Phase 2A (background recreation) is the reconstruction stage. Many published architectures decouple these into separable heads — useful prior art for our fork-join structure.

This file is a working bibliography. Date: 2026-05-06.

## Decision context

- Pretrained scene-text-removal models give us a **universal Phase 1 candidate** without per-type engineering.
- All currently-released models are trained on **natural-scene text** (SCUT-EnsText, Flickr-ST). Our CAPTCHAs are adversarial — built specifically to defeat exactly these models. **Domain-shift risk is real.**
- None of the released repos document a "standalone segmentation mask" output in their inference scripts. To use them for Phase 1 we need to either (a) modify their inference code to expose the intermediate mask tensor, or (b) compute `mask ≈ |input - output| > τ` post-hoc.

## Papers

### Decoupled / progressive STR (closest fit for Phase 1)

#### **PSSTRNet — Progressive Segmentation-guided Scene Text Removal Network** (Lyu et al., ICME 2022)
- arxiv: https://arxiv.org/abs/2306.07842
- code: https://github.com/GuangtaoLyu/PSSTRNet
- weights: ✅ Google Drive + Baidu (in repo README)
- training: SCUT-Syn + SCUT-EnsText (natural scene text)
- architecture: shared encoder + **two decoders** — one for text segmentation, one for text removal. Iterative refinement with a Mask Update module.
- params: smallest of the bunch — described as "about 1/3 of PERT's parameters"
- **Phase 1 fit:** ★★★★★ — only paper with an explicit, named segmentation decoder branch. Best chance of clean standalone mask extraction.
- **Phase 2A fit:** ★★★★ — removal decoder reusable.
- **Domain-shift risk:** medium — natural-scene training set, but two-decoder structure may transfer better than monolithic models.
- **Try first.**

#### **PERT — A Progressively Region-based Network for Scene Text Removal** (Wang et al., TIP 2023)
- arxiv: https://arxiv.org/abs/2106.13029
- code (official): https://github.com/wangyuxin87/PERT
- code (unofficial PyTorch with weights): https://github.com/KangBK0120/PERT
- weights: ✅ unofficial repo has them
- training: SCUT-EnsText
- architecture: progressive region-based erasure with balanced multi-stage Region-base Modification Strategy. 71 FPS, low param count.
- **Phase 1 fit:** ★★★ — text region masks are intermediate but not exposed as a primary output.
- **Phase 2A fit:** ★★★★ — explicit erasure pipeline.
- **Domain-shift risk:** medium-high — paper emphasizes natural scene text.

### Single-stage end-to-end STR

#### **FETNet — Feature Erasing and Transferring Network for Scene Text Removal** (Lyu et al., Pattern Recognition 2023)
- arxiv: https://arxiv.org/abs/2306.09593
- code: https://github.com/GuangtaoLyu/FETNet
- weights: ✅ Google Drive + Baidu
- training: Flickr-ST (3,004 real-world images)
- architecture: encoder-decoder, single-stage. Feature Erasing Module + attention + Feature Transferring Module. Text segmentation feature is internal guidance, not exposed as a primary output.
- **Phase 1 fit:** ★★ — segmentation is internal, would need code modification to surface it.
- **Phase 2A fit:** ★★★★★ — newer architecture, reportedly better quality than PERT.

### Synthetic-data training (most relevant to our project ethos)

#### **Stroke-Based Scene Text Erasing Using Synthetic Data for Training** (Tang et al., TIP 2021)
- arxiv: https://arxiv.org/abs/2104.11493
- code: https://github.com/tzm-tora/Stroke-Based-Scene-Text-Erasing
- weights: ✅ Google Drive (`best.pth`, `best_syn.pth`)
- training: **synthetic only** (improved synthetic text engine), eval on real
- architecture: stroke-mask prediction module + background inpainting module
- **Phase 1 fit:** ★★★★ — has explicit "stroke mask prediction" module
- **Phase 2A fit:** ★★★★ — inpainting module
- **Why this matters most:** **trained without real labels** — same constraint we have. Their technique is the canonical answer to "fill the domain gap with synthetic data." If pretrained STR models fail on our adversarial CAPTCHAs, this paper's paradigm is the path forward.

### Modern (2024+)

#### **DiffSTR — Controlled Diffusion Models for Scene Text Removal** (2024)
- arxiv: https://arxiv.org/abs/2410.21721
- ControlNet diffusion + MAE pretraining + segmentation-based mask refinement
- **Phase 1 fit:** ★★★ — uses diffusion to inpaint, with segmentation as a side product
- **Compute cost:** high — diffusion is slow
- **Use only if pretrained CNN-based methods fail and we have GPU budget for inference**

#### **TMIM — Leveraging Text Localization for STR via Text-aware MIM** (2024)
- arxiv: https://arxiv.org/abs/2409.13431
- two-stream architecture: background modeling stream + text erasing stream. Weak supervision via bbox-level annotations only (no pixel masks needed).
- code: "will be available" at https://github.com/wzx99/TMIM (NOT YET PUBLIC as of 2026-05-06)
- **Phase 1 fit:** ★★★★★ if released — explicitly designed to use cheap supervision (bboxes), which is exactly what PARSeq+CRAFT can give us
- **Action:** monitor the repo; revisit when code lands

#### **OTR — Synthesizing Overlay Text Dataset for Text Removal** (Oct 2025)
- arxiv: https://arxiv.org/abs/2510.02787
- HuggingFace: cyberagent/OTR
- dataset only, no model
- synthesizes overlay text using object-aware placement and VLM-generated content
- **Use:** as additional training data for synthetic-data fine-tuning. Worth pulling once the synthetic-data path is active.

### Older but still cited

#### **EnsNet — Ensconce Text in the Wild** (Zhang et al., AAAI 2019)
- code: https://github.com/HCIILAB/Scene-Text-Removal
- benchmark dataset: https://github.com/HCIILAB/SCUT-EnsText (3,562 images)
- the original benchmark. Not best-in-class anymore but the benchmark itself is canonical.

#### **EraseNet — End-to-End Text Removal in the Wild** (Liu et al., 2020)
- benchmark companion paper for SCUT-EnsText.

#### **"Surprisingly Straightforward Scene Text Removal"** (ECCV 2022)
- pdf: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760436.pdf
- claims a simpler baseline beats more complex methods. Worth reading for the "are we overcomplicating this" sanity check.

#### Survey
- **Visual Text Processing: A Comprehensive Review** — https://github.com/shuyansy/Visual-Text-Processing-survey
- collects scene text detection, removal, editing, and other tasks under one umbrella.

## Domain-gap-filling strategies

If pretrained STR models underperform on our adversarial CAPTCHAs, five paths (in increasing effort):

1. **Self-training / pseudo-labeling.** Run pretrained model on real CAPTCHAs, filter to images where its mask agrees with otsu_cc / SAM, fine-tune on the high-confidence subset. Cheap but can reinforce pretrained biases.
2. **Synthetic-data fine-tuning.** Build a per-type renderer (matching survey observations), generate 5k synth/type with known masks, fine-tune pretrained model. Aligns with project's Phase 3 anyway.
3. **Domain randomization.** Render extreme synthetic — harder than real — to make real captchas an "easy subspace." Robust without per-type code.
4. **Train from scratch on synthetic** (Tang 2021 paradigm). U-Net text segmenter, trained only on synth, evaluated on real. ~2-3 days; produces a model in our domain.
5. **Adversarial domain adaptation (DANN).** Domain-invariant features via gradient reversal. Often unstable; skip for v0.

## Plan for our project

### UPDATE 2026-05-06: Stage A failed — pretrained PSSTRNet does not generalize

Tested both `scut_syn.pth` and `scut_enstext.pth` checkpoints on one image each from DarkBay, City, Courier, CryptBB (upstream-identical PIL preprocessing). **Both checkpoints, all 4 types: `mask_final.mean() ≈ 0.999`** — model says "everything is background, no text anywhere." `text_prob >= 0.5` fraction is 0.0–0.6% across all 16 (image × checkpoint) pairs. Visual prob maps are nearly all black.

The domain gap between natural-scene text (SCUT-EnsText/Flickr-ST) and adversarial CAPTCHAs is **severe, not partial**. Other pretrained scene-text-removal nets (FETNet, PERT, EraseNet) are trained on the same data distribution; expect the same outcome.

The honest paths forward, in order of effort:

1. **One more empirical test:** try Tang 2021's "Stroke-Based Synthetic Data" pretrained model. It was trained on synthetic data instead of natural scene text — possibly generalizes differently. ~2 hrs to wrap.
2. **Fine-tune on synthetic CAPTCHAs.** Build minimal renderers per type, generate 5k samples/type with known masks, fine-tune PSSTRNet (or train from scratch). ~3-4 days. Aligns with project's Phase 3.
3. **Per-type custom methods (original M3 plan).** Classical-CV + SAM-prompted, type by type. ~2-3 days.

**Stage A (DONE — failed):** ~~drop in pretrained PSSTRNet, score on all 4 types via PARSeq+CRAFT pseudo-GT. Expect free wins on DarkBay/City; failure on Courier/CryptBB.~~ Outcome: failed on ALL types.

**Stage B (after Stage A measured):** for failing types, switch to synthetic-data fine-tuning (Strategy #2 above). Renderers are ~half-day each per type, fine-tuning is ~few epochs. Also seeds Phase 3.

**Stage C (long-term):** revisit TMIM and OTR when their codebases mature. Both directly addressable for our problem.

## Open questions

- How does PSSTRNet pretrained perform on out-of-distribution adversarial inputs? (No published OOD evaluation found.)
- Does the "Surprisingly Straightforward STR" baseline actually outperform PSSTRNet/PERT/FETNet in practice? Worth a comparative read.
- Can we build a renderer with enough variation to support domain randomization (Strategy #3) instead of per-type renderers (Strategy #2)?
