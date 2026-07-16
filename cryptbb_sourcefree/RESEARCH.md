# RESEARCH — source-free, GAN-based captcha data factory

Survey + recommendation for building the "render → GAN-restyle" pipeline
(`GOAL.md`). Written 2026-06 for the Fable build agent. Covers: the canonical
prior art, the content-preservation method families, what already failed in THIS
repo, and the recommended architecture with citations.

---

## 0. The problem, precisely

We want a function `restyle: (clean_text_image, mask) → scheme_styled_image` such
that the output (a) looks like scheme X (learned from unlabeled X images) and
(b) still legibly says the exact text we rendered. Label = the text we typed.
**Everything rides on (b): content preservation.** Style is easy; keeping the
letters exact is the hard, thesis-defining part.

This is a **supervised-source, unsupervised-target** translation: we fully
control the source (we render it), but the target domain is only unlabeled real
captchas. So it's *unpaired* on the real side.

---

## 1. Canonical prior art — and exactly how we differ

**Ye et al., "Yet Another Text Captcha Solver: A GAN-Based Approach", ACM CCS 2018.**
The blueprint. They: (1) train a GAN **captcha *generator*** to mimic a target
scheme, (2) train a base solver on that synthetic data, (3) **fine-tune the
solver on ~500 real labeled captchas** via transfer learning. Beat 4 SOTA solvers
across 33 schemes (incl. schemes used by Google/Microsoft/eBay). Code:
`github.com/yeguixin/captcha_solver`.

- **Why it matters:** proves GAN-synth → solver works across many schemes. This
  is our north star and validates the whole direction.
- **How WE differ (the contribution):** Ye needs **500 human-labeled real
  captchas** for the transfer step. **Our rule forbids any human labels.** So our
  job is to get as far as possible *without* that transfer step — pure
  zero-real-label. That deletion is exactly what's novel and hard. (If pure-zero
  underperforms, we have a fallback ladder — see §7.)

Sources: [Ye et al. CCS'18 PDF](https://eprints.lancs.ac.uk/id/eprint/126984/1/ccs18.pdf) · [code](https://github.com/yeguixin/captcha_solver)

---

## 2. Content-preserving unpaired translation (the generator core)

- **CycleGAN (Zhu 2017)** — cycle consistency. ⚠ **Already failed in this repo**
  (`model 1/src/cyclegan/`): cycle loss doesn't protect glyph identity → content
  drift → hurt the recognizer. **Do not use as the core.**
- **CUT / FastCUT (Park, ECCV 2020)** — replaces cycle with **PatchNCE**
  contrastive loss: a patch of the output must match the *same* patch of the
  input more than other patches. One-way (no second generator), lighter, and
  **markedly better content preservation than CycleGAN.** → **Recommended core.**
- **DCLGAN** — dual contrastive (content AND style) for better disentanglement.
- **Attention/mask-guided translation** (AttentionGAN, InstaGAN) — restrict
  changes to/around a region. We already have the mask, so we can hard-condition.

Content-preservation is a whole subfield now — see the review *"Unpaired
Image-to-Image Translation with Content Preserving Perspective"* (2025).

Sources: [CUT review 2025](https://arxiv.org/html/2502.08667v1) · [Enhanced CUT + HOG 2024](https://arxiv.org/abs/2409.16042)

---

## 3. Scene-Text-Editing (STE) — the most on-point prior art

STE = "change the text in an image, keep the font/color/background." That's
*exactly* our restyle, and these methods already solve content preservation:

- **SRNet (2019)** — divide-and-conquer: text-conversion + background-inpainting + fusion modules. The original template.
- **MOSTEL (2023)** — ⭐ closest to our setting: **semi-supervised, trains on labeled *synthetic* + unpaired *real*** images, uses a **recognition loss** and a **Background Reconstruction Module (BRM)**, guided by stroke maps. This is almost literally our recipe (labeled synth source + unpaired real target + recognizer anchor). Study its loss design.
- **TextStyleBrush (Meta 2021)** — StyleGAN2-based; encodes style into a 512-d vector, transfers to a content image. Self-supervised. Good for style-vector conditioning if we want per-image style variety.
- **Recognition-Synergistic STE (2025)** and **TextCtrl (NeurIPS 2024, diffusion)** — newer, explicitly couple a recognizer for legibility. Confirms the recognition-loss anchor is current SOTA practice.

**Takeaway:** borrow MOSTEL's structure — background-reconstruction + style
imitation + **recognition loss** on unpaired real — but simplified for
fixed-size captchas.

Sources: [Visual Text Processing review 2025](https://arxiv.org/html/2504.21682v1) · [Recognition-Synergistic STE 2025](https://arxiv.org/html/2503.08387) · [TextCtrl NeurIPS'24](https://arxiv.org/html/2410.10133)

---

## 4. The legibility anchor — OCR / recognition-consistency loss

Force a recognizer R to read the correct text out of the GAN output. Proven
effective (OCR-VQGAN, TextSR "recognition-guided super-res", multi-layer
OCR-aware losses in TextPixs/CharGen).

- **How:** `L_rec = CTC_or_CE( R(G(x)), label_of_x )`. R can be frozen (a general
  STR net) or co-trained. This is the single strongest content anchor — it
  directly optimizes the thing we care about (readability of the *right* text).
- ⚠ **Caveat (from the OCR-loss papers):** if perceptual/recognition loss
  outweighs pixel fidelity, generators **hallucinate plausible-but-wrong text**.
  → keep mask/pixel-fidelity weight high; recognition loss moderate; and
  **measure legibility directly** (R's accuracy on held-out G(x)), don't just
  trust the loss going down.
- **Cold-start:** R must read *something* early or L_rec gives no gradient.
  Bootstrap R on the clean source renders first (trivial to read), then it guides
  G as G learns to preserve legibility while restyling. Periodically refresh R on
  (G(x), label) pairs (self-training).

Sources: [OCR-VQGAN](https://arxiv.org/pdf/2210.11248) · [TextSR](https://arxiv.org/pdf/1909.07113) · [TextPixs OCR-guided](https://arxiv.org/pdf/2507.06033)

---

## 5. Supporting modules (optional, per need)

- **Background GAN (safe — no letters to wreck):** StyleGAN2-**ADA** generates
  infinite scheme backgrounds from limited data (works from ~1k imgs; we have
  600 → borderline, ADA's adaptive aug is designed for exactly this; can also
  train on 200×60 → many patches). Or just sample the existing LaMa-extracted bg
  pool (`auto_synth/data/cryptbb_unsolved/bg_lama/`). Start with sampling; add a
  bg-GAN only if variety is the bottleneck.
- **Few-shot font generation (fills missing glyphs / adds glyph variety):**
  DG-Font, MX-Font, FS-Font, VQ-Font, FontDiffuser. Only needed if the restyler
  can't produce a character we never showed it. Defer until it's the blocker.

Sources: [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada) · [DG/MX/FS-Font review](https://arxiv.org/html/2604.13797v1)

---

## 6. What already FAILED in this repo (do not repeat)

| Approach | Where | Why it failed |
|---|---|---|
| CycleGAN synth↔real translation | `model 1/src/cyclegan/` | content drift; letters change |
| GlyphControl glyph-diffusion + LoRA | `model 1 Dino attention/` | content drift, 0/5 read-back; heavy |
| Refinement GAN as a *polish* | `synthetic image generation/src/refine/` | marginal — but it was polishing an already-good composite, NOT used as the primary restyler with a recognition anchor. **Reuse the code, change the job.** |
| Composite (real bg + pasted text) | `model 1` compbg | edge/anti-alias artifacts, 1.23% word |
| Off-the-shelf OCR/VLM as solver | `model 1 Dino attention/` | ≤2.5% word (the floor we must beat) |

The through-line: **content fidelity is the whole ballgame, and only approaches
that guarantee it by construction (renderer-chosen text + a legibility anchor)
work.** CUT + recognition loss is precisely that guarantee for a GAN.

---

## 7. Recommended architecture (what Fable should build)

**"Recognition-anchored CUT restyler."**

```
random string s  ──►  neutral renderer  ──►  x = clean text img (200x60) + mask m
   (we know s)         (generic font, NOT MyBB)          │
                                                          ▼
                         G (CUT generator, input = x⊕m)  ──►  ŷ = restyled captcha
                                                          │
        real 600 CryptBB ──► D (PatchGAN, +ADA aug) ◄─────┤  adversarial: ŷ looks real
                                                          │
                         PatchNCE(x, ŷ)  ◄────────────────┤  content preserved
                         mask-L1 (soft) ◄────────────────┤  letters stay put
                         R(ŷ) == s  (CTC/CE) ◄───────────┘  letters stay LEGIBLE
                                                          │
                                                          ▼
                                     (ŷ, label=s)  ──►  labeled synth pool
                                                          │
                                     train PARSeq ──► eval on 651 real ──► vs 57.60%
```

Loss: `L = λ_adv·L_GAN + λ_nce·L_PatchNCE + λ_mask·L1_mask + λ_rec·L_rec`.
Start weights ≈ adv 1.0 / nce 1.0 / mask 5.0 / rec 0.5, then tune (keep fidelity
high, rec moderate per §4 caveat).

**Why this specific stack:** CUT (not CycleGAN) fixes the known content-drift
failure (§2, §6); recognition loss is the SOTA legibility anchor (§3, §4); mask
conditioning hard-constrains letter location; it reuses ~70% of existing repo
code (`refine/` GAN + `parseq/` recognizer). Compute fits the A4000 (CUT/PatchGAN
train in hours; no diffusion).

**Fallback ladder if pure zero-real-label underperforms:**
- Tier 1 (goal): zero real labels — L_rec uses only self-rendered labels.
- Tier 2 (still no *human* labels): add a weak real-domain signal via a
  VLM/OCR-**consensus** pseudo-label on a few reals (report as "no human labels").
- Tier 3 (baseline, report separately): Ye-style ~500 real labels transfer — the
  known-good upper bound, to quantify the cost of the no-label constraint.

---

## 8. Evaluation & ablations (how we judge it)

- **Primary:** PARSeq trained on GAN synth, word/char acc on the 651 real. Compare to Model 1 (39.94%), Model 2.5 (57.60%), and the OCR floor (≤2.5%).
- **Legibility check:** R's accuracy reading held-out G(x) (must be high, else labels are lies).
- **Realism check:** FID / DINOv2-cosine between G(x) and real; visual A/B panels.
- **Ablations:** CUT vs CycleGAN · with/without L_rec · with/without mask cond · bg-sample vs bg-GAN · λ sweep.
- **Generality:** re-run the whole pipeline on City (2000 unlabeled, 324×81) changing only the unlabeled folder + canvas size. The real "any captcha" test.

---

## Citations (all sources)
- [Ye et al., Yet Another Text Captcha Solver, CCS 2018](https://eprints.lancs.ac.uk/id/eprint/126984/1/ccs18.pdf) · [code](https://github.com/yeguixin/captcha_solver)
- [End-to-End Attack on Text CAPTCHAs (CycleGAN synth)](https://arxiv.org/pdf/2008.11603)
- [CUT / content-preserving translation review 2025](https://arxiv.org/html/2502.08667v1) · [Enhanced CUT+HOG 2024](https://arxiv.org/abs/2409.16042)
- [Visual Text Processing review 2025](https://arxiv.org/html/2504.21682v1) · [Recognition-Synergistic STE 2025](https://arxiv.org/html/2503.08387) · [TextCtrl NeurIPS 2024](https://arxiv.org/html/2410.10133)
- [OCR-VQGAN](https://arxiv.org/pdf/2210.11248) · [TextSR](https://arxiv.org/pdf/1909.07113) · [TextPixs OCR-guided](https://arxiv.org/pdf/2507.06033) · [CharGen](https://arxiv.org/pdf/2412.17225)
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada)
- [Few-shot font gen (DRG-Font review)](https://arxiv.org/html/2604.13797v1) · [FontDiffuser](https://arxiv.org/pdf/2312.12142) · [VQ-Font](https://arxiv.org/pdf/2308.14018)
- [CAPTCHA Two-Step Style-Transfer](https://pmc.ncbi.nlm.nih.gov/articles/PMC7085644/) · [Diff-CAPTCHA](https://arxiv.org/pdf/2308.08367)
