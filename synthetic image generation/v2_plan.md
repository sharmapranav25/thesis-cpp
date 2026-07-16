# synth_v2 plan

> See `src/synth_v1/STATUS.md` for the v1 deliverable and what's already
> working. This document covers what v2 needs to add and why, with a phased
> implementation and citations to the open-source tooling we'll use.

## Goal

Make synth recreate **any** captcha scheme, not just CryptBB. v1 proved that
font replication via char-crop recombination + LaMa bg works **once you have
labeled char crops**. v2's only new job is producing those labeled crops for
arbitrary schemes — i.e. replacing v1's CryptBB-specific perception layer
(`detector_v2` + `parseq_v2`) with a scheme-agnostic one.

Constraints carried over from earlier discussions:

1. **Open-source models only.** No API dependencies (no Claude / GPT-4o /
   Qwen-VL-Max). Local inference, Apache-2.0 / MIT licensed.
2. **Disjoint unlabeled pool per scheme.** v1 already exposed the bg leak
   problem on DarkBay; same constraint applies to v2's bank-build inputs.
3. **No human-labeled data in the recognizer-training loop.** v2's bank labels
   come from VLMs running on unlabeled real samples.
4. **Difficulty knob and curriculum machinery stay dropped.** v2 inherits v1's
   "render N labeled samples" interface — no `t`, no scheduler.

## What v1 already settled (don't redo)

- **Compose half (font + bg recombination).** `compose.py` is scheme-agnostic
  given a populated bank + a bg pool. Carries forward unchanged into v2.
- **Bank format.** Per-char alpha-keyed RGBA crops in case-encoded folders
  (`L_a/`, `U_A/`, `0/`). `font_bank.py`'s `CharBank` reads it as-is.
- **Bg pipeline.** LaMa-extracted bg pool from disjoint unlabeled real
  samples. Already works on CryptBB (600 imgs); applies to any scheme with
  unlabeled data + a detector that finds char regions to inpaint.
- **Output schema.** `labels.csv` + `bboxes.jsonl` + RGB PNGs at native size.

## Architecture

### v2 pipeline (replaces v1's `build_bank.py` only)

```
[unlabeled real captcha]
        │
        ├──► CRAFT detector ──► per-character bboxes
        │                           │
        ├──► Qwen2.5-VL-7B  ──► char string + per-char JSON bbox
        │                           │
        ├──► Qwen2.5-VL-7B  ──► (second pass, prompt variant)
        │       (consistency)       │
        ▼                           ▼
       merge & filter (Consensus Entropy)
                  │
                  ├─► alpha-keyed crop  ──► bank/<char>/<source>_<idx>.png
                  └─► reject log         (with reason: disagreement / bbox mismatch)
```

Two specialist models, paired:

- **CRAFT** ([Baek et al. 2019](https://arxiv.org/abs/1904.01941),
  [clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch), MIT,
  ~20 MB). Built around a **character-region awareness** map plus a
  character-affinity map; this is the only well-known detector designed to
  emit *per-character* bboxes rather than word/line bboxes.
- **Qwen2.5-VL-7B-Instruct** ([Qwen team 2025](https://qwenlm.github.io/blog/qwen2.5-vl/),
  Apache 2.0, ~14 GB bf16 / ~5 GB 4-bit). Strong general-purpose VLM with
  JSON-grounded OCR. Pixel-coordinate bboxes via the `bbox_2d` field. 88.8 on
  OCRBench. Captcha is not its strongest domain (low-frequency text per
  [OCRBench v2](https://arxiv.org/html/2501.00321v2)) but combined with CRAFT
  it covers the gap.

The ensemble matters because the failure modes are uncorrelated:
- CRAFT can return merged regions on touching chars but always emits
  *something* per region.
- Qwen-VL can read distorted glyphs but is poor at fine spatial localization.
- When CRAFT's box count matches Qwen's char count, both signals agree on
  the segmentation — high confidence.

### Quality filter — Consensus Entropy (CE)

[CE-OCR (Zheng et al. 2025)](https://arxiv.org/abs/2504.11101) is a
training-free protocol: when multiple VLMs agree on the OCR string, entropy
is low → trust; when they disagree, entropy is high → reject or escalate.
Reported gains: **+42.1% F1 over single-model VLM-as-judge.**

For v2's first cut we use a cheap variant — same model, two prompt seeds:

```
pred_A = qwen(image, prompt_A)
pred_B = qwen(image, prompt_B)   # different temperature / phrasing
keep if pred_A == pred_B (string-level)
keep position i only if box_A[i] IoU box_B[i] > 0.5
```

If single-model dual-pass isn't enough (i.e., acceptance rate too high but
quality too low), upgrade to two distinct VLMs:
**Qwen2.5-VL-7B + InternVL2.5-4B** ([OpenGVLab/InternVL3](https://github.com/OpenGVLab/InternVL),
Apache 2.0). Different vision encoders → genuinely independent failure modes.

## Phased implementation

Each stage has a **go/no-go gate** — don't start the next until the previous
hits its target.

### Phase 0 — environment & smoke (≈2 h)

- [ ] Pin Qwen2.5-VL-7B-Instruct, CRAFT-pytorch, simple-lama-inpainting in
      `model 1/.venv` (or new `.venv_v2/` if the existing one breaks).
- [ ] 4-bit quant Qwen via bitsandbytes (memory budget: 16 GB A4000).
- [ ] Run Qwen on 5 sample CryptBB images, confirm it emits parseable JSON
      with `bbox_2d` and `content` fields.
- [ ] Run CRAFT on the same 5, confirm it returns 5 boxes per image.

**Gate:** both models inferenceable, no environmental issues.

### Phase 1 — `perceive.py` (≈4 h)

- [ ] `src/synth_v2/perceive.py` — `perceive(image_path) -> List[(bbox, char, conf)]`.
      Calls CRAFT for boxes, Qwen for string, pairs them by left-to-right
      sort. Returns `None` if `len(boxes) != len(string)`.
- [ ] CE filter as a separate function: dual Qwen pass → consensus check.
- [ ] Unit smoke: 5 hand-picked CryptBB images, eyeball output for sanity.

**Gate:** `perceive.py` runs end-to-end without errors on at least 4/5 sample
images, with reasonable boxes + correct strings.

### Phase 2 — validation against labeled eval (≈3 h)

- [ ] Score `perceive.py` against `model 1/data/eval/` (651 labeled CryptBB).
- [ ] Side-by-side metrics with v1's `detector_v2 + parseq_v2` output:
      char-acc, string-acc, mean per-char IoU, CE acceptance rate.
- [ ] Result table → `src/synth_v2/STATUS.md`.

**Gate:** v2 perception meets one of two bars on CryptBB:
1. Char-acc within 5% of v1, OR
2. CE filter acceptance rate × char-acc-on-accepted ≥ v1's char-acc
   (i.e., a noisier model but a strong filter still produces clean labels).

If neither bar is met, v2 is *worse than v1 on CryptBB* and the open-source
VLM path is invalidated. Re-evaluate before continuing.

### Phase 3 — v2 bank build for CryptBB (≈2 h)

- [ ] Refactor `src/synth_v1/build_bank.py` → `src/synth_v2/build_bank.py`
      that calls `perceive.py` instead of `detector_v2 + parseq_v2`. The
      crop-extraction + alpha-keying half stays identical to v1.
- [ ] Run v2 bank build on `model 1/data/raw/` (600 unlabeled CryptBB).
- [ ] Compare bank to v1 bank: per-char count, sample-crop QA sheet.

**Gate:** v2 bank covers ≥55 of 62 charset chars with min_count ≥ 5 (v1
covers 61). If much sparser, the CE filter is too aggressive — relax.

### Phase 4 — v2 generate.py + 100k smoke (≈1 h)

- [ ] `src/synth_v2/generate.py` reuses v1's `compose.py` and `font_bank.py`
      unchanged; only the bank dir changes.
- [ ] Render 100k → `E:/pranav/synth_data/v2/cryptbb/`.
- [ ] QA sheet: synth_v2 vs synth_v1 vs real, 4×8 each.

**Gate:** v2 synth visually indistinguishable from v1 synth (we expect this —
compose half is unchanged; bank quality is the variable).

### Phase 5 — second scheme (≈4 h, the real "any captcha" test)

- [ ] Pick a scheme with disjoint unlabeled data: **City** (2000 unlabeled,
      324×81) is the cleanest test because layout differs from CryptBB and
      its char count is unknown a priori.
- [ ] LaMa-extract bg pool from City unlabeled (no detector needed —
      `extract_bgs_lama.py`'s mask can come from CRAFT instead of
      `detector_v2`).
- [ ] Run `perceive.py` on 100 City samples, eyeball CRAFT bboxes + Qwen
      strings.
- [ ] Build bank, render 100 synth, side-by-side QA against real City.

**Gate:** The honest "any captcha" test. If City synth visually matches real
City (font + layout + bg) without any new training, v2 has delivered the
goal. If not, the failure mode tells us where to improve (CRAFT can't
segment that scheme's chars / Qwen can't read its glyph style / both).

## Validation strategy

Three pseudolabel pipelines, scored on the same 651 labeled CryptBB:

| Pipeline | What it tells us |
|---|---|
| `detector_v2 + parseq_v2` | v1 baseline (CryptBB-trained, ceiling) |
| `CRAFT + Qwen2.5-VL-7B` (single pass) | v2 raw open-source perception quality |
| `CRAFT + Qwen + CE filter (dual pass)` | v2 quality after consensus filtering |

Metrics:
- **char-acc** — fraction of characters correctly labeled (matches ground truth)
- **string-acc** — fraction of images where all 5 chars match
- **mean IoU** — per-char bbox overlap between predicted and (where
  available) detector_v2 ground-truth boxes
- **acceptance rate** — what fraction of images survive the CE filter
- **clean-yield** = acceptance × string-acc-on-accepted — the practical
  metric for bank quality

This is fair use of the labeled eval set: we're scoring a *pseudolabeler*,
not training a recognizer on it. The "no human labels in recognizer training"
constraint is preserved.

## Open questions / risks

1. **Captcha is low-frequency text for Qwen-VL.**
   [OCRBench v2](https://arxiv.org/html/2501.00321v2) explicitly flags
   captcha as a weak point for current VLMs. We may discover Qwen reads
   ~70% of CryptBB chars correctly, vs. parseq_v2's ~87%. If so, the CE
   filter has to do a lot of heavy lifting to keep bank quality high.
2. **CRAFT trained on scene text, not captchas.** Crayon-style
   anti-aliased glyphs over textured bg are not in CRAFT's training
   distribution. Region/affinity scores may be noisy. Cheap mitigation:
   threshold sweep on CRAFT's score map; expensive: fine-tune CRAFT on a
   small captcha corpus.
3. **Qwen2.5-VL bbox quality is coarse.** Qwen's pixel coordinates are
   reportedly accurate at object scale, less so at character scale within
   a 200×60 image. Plan: rely on CRAFT for bboxes, Qwen for strings only.
4. **CE filter may be too strict on hard captchas.** If CryptBB acceptance
   drops below ~60%, we'll have too few crops per char (v1 had ~50/char from
   600 imgs; a 60% acceptance rate halves that). Workaround: source more
   unlabeled real samples; or relax filter (single Qwen pass + CRAFT
   confidence threshold only).
5. **Quantization may degrade Qwen's bbox emission.** 4-bit quant is fine
   for chat but VLMs sometimes lose grounding precision. Bench fp16 vs.
   int4 on the CryptBB validation set; if int4 drops > 10% acceptance,
   stay at fp16 and accept slower throughput.
6. **GLM-OCR (0.9B) is the wildcard.** Z.ai's March 2026 release tops
   OmniDocBench v1.5 at 0.9B params. If it works on captchas, it could
   replace both CRAFT *and* Qwen — single model, MIT-ish license,
   document-tuned grounding. Worth a 10-image smoke test in Phase 0.
7. **DarkBay / contamination.** v2 doesn't fix the data leak problem —
   it's a data-sourcing issue, not a model issue. Until disjoint unlabeled
   DarkBay exists, DarkBay is excluded from v2 just like from v1.

## What v2 does *not* do

- No FFG / DG-Font training. The char-crop bank approach is good enough
  for in-distribution chars; FFG only matters for chars absent from the
  unlabeled pool, which we haven't seen.
- No refiner net. v1's refiner showed marginal lift; not load-bearing.
- No curriculum or difficulty knob. Permanently dropped.
- No new schemes beyond Phase 5's smoke test on City. Adding a third
  scheme is a Phase 6 / v2.1 problem if v2 succeeds.

---

## Citations

### Open-source VLMs and OCR models

- Baek et al., *Character Region Awareness for Text Detection (CRAFT)*,
  CVPR 2019. [arxiv:1904.01941](https://arxiv.org/abs/1904.01941),
  [code](https://github.com/clovaai/CRAFT-pytorch).
- Bai, Bai, Yu, et al. (Qwen team), *Qwen2.5-VL Technical Report*, 2025.
  [blog](https://qwenlm.github.io/blog/qwen2.5-vl/),
  [HF model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
- Chen et al. (OpenGVLab), *InternVL3 / 2.5*, 2025.
  [code](https://github.com/OpenGVLab/InternVL).
- Florence-2 (Microsoft), 2024. [Roboflow OCR guide](https://blog.roboflow.com/florence-2-ocr/),
  [HF model](https://huggingface.co/microsoft/Florence-2-large).
- GLM-OCR (Zhipu AI / Z.ai), Mar 2026. [Decode the Future overview](https://decodethefuture.org/en/glm-ocr-explained/).
- DeepSeek-OCR, 2025. [Analytics Vidhya comparison](https://www.analyticsvidhya.com/blog/2025/11/deepseek-ocr-vs-qwen-3-vl-vs-mistral-ocr/).
- PaddleOCR PP-OCRv5 (Baidu). [TildAlice benchmark](https://tildalice.io/paddleocr-vs-easyocr-benchmark/).
- EasyOCR. [Comparison vs PaddleOCR / Tesseract](https://medium.com/@shah.vansh132/comparison-of-text-detection-techniques-easyocr-vs-kerasocr-vs-paddleocr-vs-pytesseract-vs-opencv-44c2bc22b133).

### Ensemble / consensus methods

- Zheng et al., *Consensus Entropy: Harnessing Multi-VLM Agreement for
  Self-Verifying and Self-Improving OCR*, Apr 2025.
  [arxiv:2504.11101](https://arxiv.org/abs/2504.11101).
- *Vote-in-Context: Turning VLMs into Zero-Shot Rank Fusers*, 2025.
  [arxiv:2511.01617](https://arxiv.org/html/2511.01617).

### Benchmarks

- Liu et al., *OCRBench v2: An Improved Benchmark for Evaluating Large
  Multimodal Models on Visual Text Localization and Reasoning*, Jan 2025.
  [arxiv:2501.00321](https://arxiv.org/html/2501.00321v2).
- *Best Open-Source OCR Models 2025* (E2E Networks).
  [link](https://www.e2enetworks.com/blog/complete-guide-open-source-ocr-models-2025).
- *Multimodal AI: The Best Open-Source Vision Language Models in 2026*
  (BentoML).
  [link](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models).

### Project state references (in this repo)

- `src/synth_v1/STATUS.md` — what v1 delivered.
- `CONTEXT.md` — broader project state, schemes, eval contamination notes.
- `future.md` — deferred items including FFG / DG-Font (not v2's path).
- `notes.md` — alternatives considered earlier.
