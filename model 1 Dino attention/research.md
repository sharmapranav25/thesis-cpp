# Research — synthetic captcha generation via glyph-conditioned diffusion

## Problem framing

Goal: generate synthetic CryptBB-style captchas that are **visually indistinguishable from real** (so a recognizer trained on synthetic transfers without a domain gap), while **guaranteeing the 5-char label is preserved** in every sample.

Past attempt in `model 1/`: CycleGAN style-transfer from renderer output → real-style. Failed because cycle consistency doesn't strongly protect glyph shape — translated samples drifted off-label and hurt recognizer accuracy.

Modern approach: **glyph-conditioned Diffusion Transformers (DiTs)**. Render the glyphs explicitly first as a control signal; diffusion paints captcha-style background, distortion, color, shadow on top. Content fidelity becomes structural (hard-wired in the conditioning), not learned.

---

## Core papers — glyph-conditioned text-image generation

### GlyphControl (NeurIPS 2023)
- **Why it matters here:** ControlNet-style glyph conditioning. Most direct fit for our pipeline — input is a pre-rendered glyph image (which our existing renderer already produces), output is the stylized version.
- Link: <https://openreview.net/forum?id=thPI8hrA4V>

### TextDiffuser & TextDiffuser-2 (Microsoft, 2023)
- **Why:** character-level segmentation control over diffusion. Open weights. "Diffusion Models as Text Painters."
- Link: <https://jingyechen.github.io/textdiffuser/>

### AnyText / AnyText2 (Alibaba, 2023–2024)
- **Why:** OCR-guided positioning + diffusion, multilingual. Strong open-source baseline. SD-1.5 backbone, fits in ~6 GB VRAM at fp16 inference.
- Paper collection: <https://github.com/yeungchenwa/Recommendations-Diffusion-Text-Image>

### EasyText (May 2025)
- **Why:** newer DiT-based controllable text rendering, multilingual. Built on the FLUX/SD-3.5 line of DiTs.
- Link: <https://arxiv.org/html/2505.24417>

### UniGlyph (ICCV 2025)
- **Why:** unified segmentation-conditioned diffusion, "precise visual text synthesis." Latest in the line.
- Link: <https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_UniGlyph_Unified_Segmentation-Conditioned_Diffusion_for_Precise_Visual_Text_Synthesis_ICCV_2025_paper.pdf>

### GlyphDraw2 (2025)
- **Why:** iteration on glyph-conditioned diffusion; cited alongside UniGlyph as state of the art for visual text fidelity.

### FreeText (2026, training-free)
- **Why:** training-free plug-in for DiT models (FLUX, SD, Qwen-Image). Spectral glyph injection + attention localization. If we don't want to train, this is the lowest-effort option.
- Link: <https://arxiv.org/html/2601.00535>

### GLM-Image (Zhipu, 2025)
- **Why:** uses Glyph-ByT5 text encoder that produces tokens representing the *shape* of each character. Two-stage layout-then-fill architecture.
- Link: <https://www.deeplearning.ai/the-batch/zhipus-glm-image-blends-transformer-and-diffusion-architectures-for-better-text-in-images/>

### Refining T2I via Glyph-Enhanced Generation (2024)
- **Why:** training-free improvements to glyph fidelity in existing T2I models.
- Link: <https://arxiv.org/html/2403.16422v2>

---

## Captcha-specific papers

### Diff-CAPTCHA (2023, arXiv 2308.08367)
- Uses denoising diffusion to *generate* CAPTCHAs (defender side — making harder captchas). Inverse direction from us, but useful for the architectural pattern: text + background fused via diffusion.
- Link: <https://arxiv.org/abs/2308.08367>

### CAPTCHA Image Generation: Two-Step Style-Transfer Learning (Kwon-Yoon)
- Most directly captcha-specific. Two-step style transfer to generate captchas. Pre-DiT (CNN-based) but the workflow maps closely to ours.
- Link: <https://www.semanticscholar.org/paper/CAPTCHA-Image-Generation:-Two-Step-Style-Transfer-%E2%80%A0-Kwon-Yoon/5d646b325d53aee9b869a80f006947e83f60b39a>

### COGNITION (2025)
- "From Evaluation to Defense against Multimodal LLM CAPTCHA Solvers." Confirms current MLLMs (e.g. Qwen-VL, GPT-4V) can solve many captcha types — useful as a confidence baseline / pseudo-labeler.

### An Experimental Investigation of Text-based CAPTCHA Attacks (ACM CSUR 2022)
- Survey of text-CAPTCHA attack methods and robustness.
- Link: <https://dl.acm.org/doi/10.1145/3559754>

---

## Domain adaptation / OCR-side context

### LightOnOCR-1B (LightOn, 2025)
- Pixtral-ViT encoder + Qwen3-based text decoder, distilled from open VLMs. Trained on real + synthetic. Domain-adaptable for receipts, forms, etc. Possible OCR fine-tune target instead of PARSeq if we want a stronger pretrained recognizer.
- Link: <https://huggingface.co/lightonai/LightOnOCR-1B-1025>
- Blog: <https://huggingface.co/blog/lightonai/lightonocr>

### Vision Transformers in Domain Adaptation and Generalization: A Study of Robustness (2024)
- Survey of how ViTs handle synthetic-to-real shift.
- Link: <https://arxiv.org/html/2404.04452v1>

### TransAdapter
- ViT for feature-centric unsupervised domain adaptation.
- Link: <https://openreview.net/forum?id=ATdshE4yIj>

### Efficient Domain Adaptation for Text Line Recognition via Decoupled Language Models (2026, arXiv 2603.28028)
- Lightweight visual character detection + linguistic correction module trained entirely on synthetic noise. Annotation-free domain adaptation.
- Link: <https://arxiv.org/abs/2603.28028>

---

## Background / scaling context

### PixArt-α (2023)
- Fast training of Diffusion Transformer for photorealistic T2I. Foundation for several text-rendering follow-ups.
- Link: <https://pixart-alpha.github.io/>

### Scaling T2I Diffusion Transformers with Representation Autoencoders (2026)
- Notes that targeted data composition matters more than raw scale for text-in-image fidelity. Validates the "fine-tune on real captchas" sub-step.
- Link: <https://arxiv.org/abs/2601.16208>

---

## Concrete pipeline (chosen direction)

1. **Render** — keep `model 1/src/render_synthetic.py` as the glyph generator. Each sample → (glyph image, label).
2. **Style transfer via glyph-controlled diffusion** — feed the rendered glyph image as the control signal to GlyphControl (or AnyText). Output is the stylized captcha.
3. **Style fine-tune** — LoRA fine-tune the diffusion backbone on the 600 unlabeled real CryptBB captchas (unconditional, just to absorb the visual style). ~5 GB VRAM, fits the A4000.
4. **Generate** — produce 50–100k stylized samples with chosen labels. Labels are exactly preserved by glyph conditioning.
5. **Train recognizer** — PARSeq or Qwen-VL fine-tune on the new dataset.
6. **Evaluate** — score on the 649 labelled set in `data/eval_labelled/`.

### Why this beats CycleGAN
- **Content fidelity:** explicit glyph control → labels are guaranteed (CycleGAN had no such guarantee).
- **Style fidelity:** diffusion fine-tuned on real images → matches the real distribution at the texture/color/distortion level.
- **Sample efficiency:** style fine-tune needs only the 600 real images we already have.

### Why this beats raw renderer + heavy augmentation
- Augmentation can't introduce features that aren't in the rendered base — it can blur, jitter, distort, but it can't synthesize the *kind* of speckle, grid, or chromatic shadow patterns the real PHP library produces.
- Diffusion-style transfer can.

### Open risks
- 600 real images may be too few for a clean style LoRA — may need to upweight or use augmentation on the real set itself.
- Diffusion inference is slow (~1–3 s/image on this GPU). 50k samples ~14–40 hours — manageable overnight.
- GlyphControl was trained on natural-scene text; CryptBB style (heavy speckle, vivid color, chromatic shadow) is out-of-distribution for the base model. Style LoRA must close that gap.
