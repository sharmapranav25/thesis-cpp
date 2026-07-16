# The Synthesis Graveyard — A Consolidated Post-Mortem of Failed CAPTCHA-Synth Attempts

> **Scope of this document.** The thesis has two problems. The *easy* one (mostly
> solved) is **recognizing** a captcha once you have matched training data. The
> *hard* one — the actual research contribution — is **generating** distribution-matched,
> labeled synthetic training data (“synth”) for an **arbitrary** scheme
> **without that scheme's source code, without its font, and without human labels**
> (“source-free, label-free synth for *any* captcha”).
>
> This file consolidates every **failed, dropped, parked, superseded, or only-marginal**
> attempt at that generation problem found across `e:/pranav`. It is the “dead-ends
> chapter”: what was tried, the concrete mechanism, the **root cause** of failure,
> the **evidence** (real numbers from logs/metrics, not memory), and where the code lives.
> Recognizer-architecture failures are excluded unless the recognizer failed *because
> of the synth it was fed*.
>
> Compiled 2026-07-06 from a fresh directory-by-directory crawl (10 extractor passes +
> a completeness critic). All metrics quoted from the referenced files.

---

## 0. How to read this

- **36 distinct synthesis attempts failed or stalled** before the two things that
  actually worked were found (see §“What actually worked”). The graveyard is the
  point: the difficulty of *generation* — not recognition — is the thesis.
- Every failure traces to one of **six recurring root causes** (§1). If you are about
  to try a new synth idea, check it against those six first.
- Reference baselines on the **651 real CryptBB eval set** (used everywhere for
  apples-to-apples), so the failure numbers below have meaning:

  | Milestone | Word acc | Char acc | Note |
  |---|---:|---:|---|
  | Off-the-shelf OCR/VLM floor | ≤2.47% | ≤27.5% | best = Qwen2-VL-2B zero-shot; the “nothing reads these” floor |
  | Model 1 v1 (PIL, 32 generic fonts) | 31.18% | — | first distribution-*approximating* renderer |
  | Model 1 v2 (MyBB's own 2 fonts) | **39.94%** | 84.28% | PARSeq-alone; the +8.76pt font swap |
  | Model 1.5 v2 (ensemble) | 49.16% | 86.86% | shipped baseline |
  | **Model 2.5 (flagship)** | **57.60%** | 89.04% | fixed pool + teacher-NLL curriculum + ensemble |
  | Source-free + label-free VLM (CryptBB) | 30.4% | 71.9% | **no source, no font, no labels** — the real prize |
  | Source-free + label-free VLM (DarkBay) | 91.0% | 97.3% | same pipeline, easier scheme |

---

## 1. The six root causes that killed almost everything

Read these first. Nearly every entry in the catalogue is an instance of one (or two) of these.

1. **The Font Wall.** Source-free rendering cannot reproduce a scheme's *exotic glyph
   shapes* (e.g. CryptBB's MyBB font). No open font exceeds **~37% char-agreement** with
   the real font (vs 90.2% for the real font itself). Worse, a **font-specific**
   recognizer (PARSeq/CRNN/TinySTR) memorizes whatever glyph shapes it trained on and
   **cannot read an unseen font** — so source-free synth transfers at **0–4% word**. This
   single wall is the root cause of most GAN/render/self-train failures on CryptBB.
   *Escape:* a **font-general pretrained VLM** (Qwen2-VL) reads the real font
   out-of-the-box and sidesteps the wall entirely. This is the one true pivot.

2. **Content preservation ≠ style transfer — and content is the whole game.** Every
   method that tried to produce *content + style at once* (CycleGAN, CUT/PatchNCE,
   GlyphControl, AnyText, QwenStyle, B-CvT) drifted the **character content**: the
   adversarial/style objective fights glyph fidelity, so the generated image no longer
   contains the label you assigned it. A single wrong stroke breaks the label. *“Style is
   easy; keeping the content is the whole game.”* The winning philosophy **split the
   problem**: a renderer *guarantees* correct text; style is handled cheaply/separately.

3. **Pretrained perception doesn't transfer to adversarial captchas.** Scene-text
   detectors/removers (PSSTRNet, CRAFT) and OCR (easyOCR) are trained on natural-scene
   text — the exact thing captchas are engineered to defeat. Dropped onto captchas they
   emit garbage: PSSTRNet calls the **entire image background** (mask ≈ 0.999); easyOCR
   **hallucinates 91 junk characters** on CryptBB. Any pipeline that gates on pretrained
   perception inherits this.

4. **Distribution-match beats sophistication *and* visual fidelity.** The boring,
   line-by-line-ported renderer using the target's *actual fonts* beat every fancy
   generator. Conversely, *any* perturbation that drifts the synth off the real manifold
   — supersampling, a difficulty ladder, a background-free variant, a “prettier” wave —
   **hurts**, even when it looks better to a human. A stronger, *uglier* wave that matches
   the real distortion pipeline transferred better than a milder pretty one:
   **pipeline-match > fidelity.**

5. **Compositing and inpainting leave tells.** Alpha-blending rendered text over a
   background creates **anti-aliasing boundary artifacts** the recognizer keys on
   (real captchas render text+bg in one pass). Inpainting to *recover* a clean background
   (LaMa, cv2.inpaint) **hallucinates** char-shaped texture and leaves **ghost letters** —
   which, if labeled with your word, silently poison the label. Better masks (SAM,
   GroundingDINO) don't fix it because the residue is *inpainting hallucination*, not a
   mask deficiency.

6. **Label-free recovery has a cold-start / minimum-pool floor.** Self-training on
   unlabeled reals only compounds if the bootstrap already produces a *seed* of correct
   whole-captcha pseudo-labels. On the pathological worst case (CryptBB, where even OCR is
   0% word) the seed is empty, so the flywheel never turns. Even when it turns, a
   **too-small pool** (~16 confident labels) *declines* accuracy via confirmation bias;
   you need a floor of ~89+ correct pseudo-labels before it climbs.

---

## 2. Master index of failed attempts

Status legend: **FAILED** (built, ran, lost) · **SUPERSEDED** (worked partially, replaced by a better idea) · **DROPPED** (built/triaged then abandoned) · **PARKED** (started, never validated) · **MARGINAL** (works a little, not load-bearing) · **PARTIAL** (works but has a systematic gap).

| # | Attempt | Family | Scheme | Status | Killer evidence | Dir |
|---:|---|---|---|---|---|---|
| 1 | Generic 5-PHP-lib corpus (“Model −1”) | Generic renderer | generic | FAILED | 1.08% word on *own* synth val | `model 0/` |
| 2 | PIL 32-generic-font renderer | Generic renderer | CryptBB | SUPERSEDED | 31.18%→39.94% word when fonts swapped to MyBB's | `model 1/` |
| 3 | Background-free synth + heavy aug | Generic renderer | CryptBB | FAILED | never broke 1.5% word | `model 1/` |
| 4 | Parametric TTF renderer + DINOv2 auto-tuner | Parametric | DarkBay | SUPERSEDED | search plateaued (dist 0.189); per-char vs per-image style mismatch | `synthetic image generation/` |
| 5 | Difficulty-knob generator (RefinedGenerator, `t∈[0,1]`) | Curriculum-knob | DarkBay | DROPPED | no valid curriculum metric (DINOv2 = similarity, not difficulty) | `synthetic image generation/` |
| 6 | Difficulty-mode MyBB renderer (entangled `d` knob) + var-length | Curriculum-knob | CryptBB | SUPERSEDED | 42.82% synth→**10.45% real** word; var-len → 0% word (decoder collapse) | `model 2/` |
| 7 | v3 synth: 2× supersample + mixed-scale | Renderer perturbation | CryptBB | DROPPED | killed on visual A/B; drifted off real distribution | `model 1/` |
| 8 | CycleGAN synth→real translation (v1/v2) | GAN-translation | CryptBB | FAILED | worse than raw synth; v2 D-loss collapse @ep8 | `model 1/` |
| 9 | CycleGAN-translated set → TinySTR cold-start (v2/v3/v4) | GAN-translation | CryptBB | FAILED | frozen at random baseline (1.5% char) for 7 epochs | `tiny STR/` |
| 10 | CUT/PatchNCE recognition-anchored restyler (“Route B”) | GAN-translation | CryptBB | FAILED | reads 82% but **D still fully separates**; real transfer 0.92% vs 4.15% raw | `cryptbb_sourcefree/` |
| 11 | GlyphControl glyph-diffusion + style LoRA | Diffusion | CryptBB | FAILED | 0/5 PARSeq read-back (content drift + wrong style) | `model 1 Dino attention/` |
| 12 | GlyphControl / AnyText / QwenStyle bundle | Diffusion | CryptBB | PARKED | GlyphControl 0/5; AnyText unfinished; QwenStyle 20B won't fit A4000 | `model 1/` |
| 13 | B-CvT skin-lesion style-transfer framework | Style transfer | generic | DROPPED | rejected at paper triage (domain-specific; joint content+style) | `papers/` |
| 14 | Compbg composite (LaMa real bg + PHP text overlay) | Composite | CryptBB | FAILED | 1.23% word / 72.62% char (boundary artifacts) | `model 1/` |
| 15 | Alpha-compositor over LaMa bg | Composite | CryptBB/DarkBay | PARKED | LaMa hallucinates ~20%; ghost fragments survive → double text | `synthetic image generation/` |
| 16 | Source-free composite renderer (styled glyphs + procedural bg) | Composite | CryptBB | MARGINAL | caps ~4.15% word (font wall) | `cryptbb_sourcefree/` |
| 17 | Real-bg reuse via LaMa inpainting | Composite | CryptBB | PARKED | ghost letters poison labels; blob filter keeps only ~124/200 | `cryptbb_sourcefree/` |
| 18 | `bg_syth` + `Font_syth` reconstruction tracks | Composite | all 4 | DROPPED | never started (gated on part-extraction that never passed) | `captcha regenerator/` |
| 19 | Part-extraction attack (regenerate-from-parts, whole strategy) | Part-extraction | all 4 | SUPERSEDED | phase-1 never cleared IoU gate on any scheme | `captcha regenerator/` |
| 20 | PSSTRNet scene-text-removal net | Part-extraction | all 4 | FAILED | mask ≈ 0.999 (“all background”) on all 16 runs | `captcha regenerator/` |
| 21 | otsu_cc / SAM / SWT extraction zoo | Part-extraction | all 4 | FAILED | all OUT_OF_SCOPE; mean IoU 0.13–0.36 vs 0.7 gate | `captcha regenerator/` |
| 22 | `masking/` VLM-color-gate → HSV / DINO-attention | Part-extraction | all 4 | PARKED | no method reached PASS; never scored vs GT | `captcha regenerator/` |
| 23 | Real-glyph-crop recomposition (cluster 2411 crops) | Part-extraction | CryptBB | DROPPED | clusters impure (mix 2–3 chars); labeling is circular | `cryptbb_sourcefree/` |
| 24 | Mask-aware refiner GAN (U-Net + PatchGAN, ×4) | Refiner-GAN | DarkBay | MARGINAL | lift within noise; not load-bearing | `synthetic image generation/` |
| 25 | From-scratch / MAE-init PARSeq on source-free synth | Font-specific transfer | CryptBB | FAILED | 0% word / 18% char on real (font wall) | `cryptbb_sourcefree/` |
| 26 | MAE pretrain + progressive self-training | SSL / self-train | CryptBB | FAILED | bootstrap 0% word → pseudo-labels 1/600 → stalls | `cryptbb_sourcefree/` |
| 27 | Disjoint 100-real self-train subset (min-pool) | SSL / self-train | CryptBB | FAILED | 30.31→29.40→27.59% (declines below pool floor) | `cryptbb_sourcefree/` |
| 28 | synth_v2 CRAFT + Qwen-VL + Consensus-Entropy | VLM-synth | any | SUPERSEDED | never logged a result; Qwen miscounts chars | `synthetic image generation/` |
| 29 | Generalized measured-style auto-fit generator | VLM-synth | CryptBB/DarkBay | DROPPED | visual regressions; reverted, never scored | `cryptbb_sourcefree/` |
| 30 | OCR-direct discovery (easyOCR reads whole captcha) | Discovery | CryptBB/Joey | SUPERSEDED | “discovered” 91 junk chars; length spread 1–9 | `auto_synth/` |
| 31 | Courier composite synth (LaMa-cleaned bg) | Composite | Courier | MARGINAL | underline survives inpainting (weakest match) | `auto_synth/` |
| 32 | MobileSAM box-prompt bg extraction | Part-extraction | all 4 | DROPPED | marginal mask gain; doesn't fix hallucination residue | `auto_synth/` |
| 33 | Qwen2.5-VL-3B detector + SAM bg | VLM-synth | Courier | PARKED | run only on Courier; never adopted | `auto_synth/` |
| 34 | GroundingDINO / Grounded-SAM iterative bg | Part-extraction | Courier | PARKED | same hallucination wall; never generalized | `auto_synth/` |
| 35 | CC+DINOv2+KMeans glyph bank (OCR-vote mapping) | Part-extraction | all 4 | PARTIAL | digit **‘6’ never recovered** for Courier/City; CryptBB injects misread ‘{’ | `auto_synth/` |
| 36 | Naive compose layout + under-counted length | Composite | all 4 | PARTIAL | even-slot layout ≠ real wave/overlap; CryptBB len mode 4 (true 5) | `auto_synth/` |

---

## 3. The catalogue, by family

### Family A — Generic & parametric renderers (wrong distribution by construction)

**A1. Generic 5-PHP-library corpus — “Model −1” · FAILED · `model 0/`**
The earliest idea: buy generality with *breadth*. One PHP dispatcher
(`generators/gen_corpus.php`) round-robins **5 off-the-shelf captcha libraries**
(Securimage, Gregwar/Captcha, KCAPTCHA, Cool PHP Captcha, Simple PHP Captcha), forcing a
known random label into each (a `SecurimageForced` subclass overrides `createCode()`;
Gregwar takes the label in its constructor; KCAPTCHA is read back via `getKeyString()`)
and randomizing **every** knob: size (150–280 × 40–100), length 4–8, one of 8 charsets
(incl. a case-sensitive 62-char set), colors, ~130 fonts, and each library's distortions.
100k images; a 23.8M PARSeq trained on it, no real labels.
- **Why it failed:** the union of 5 renderers × random sizes × 8 charsets × variable
  length × every distortion is *maximal entropy* — there is no consistent distribution to
  fit. Since word-acc ≈ char-acc^length, even decent glyph recognition collapses at word
  level (root cause #4).
- **Evidence:** peaked at **1.08% word** on its *own* synth-val split
  (`checkpoints/parseq/.../metrics.csv`, ep14 `val_accuracy=1.08` — a 0–100-scale figure,
  i.e. essentially zero, *not* 108%). Char-NED did climb 3.1→37.9 (it learned glyph
  shapes but never words). Never even tested on real captchas.
- **The lesson that founded the thesis:** its successor abandoned breadth and did a
  line-by-line port of the *target's own* MyBB `captcha.php` → **39.94%** real word,
  PARSeq-alone. *Match the target distribution; don't train on a union of everything.*

**A2. PIL 32-generic-font renderer · SUPERSEDED · `model 1/`**
`src/render_synthetic.py` renders 5-char captchas with **32 generic Windows TTF fonts** +
approximate MyBB distortions, on the premise that a downstream CycleGAN would close the
residual gap.
- **Why it failed:** wrong fonts = wrong distribution, and the CycleGAN rescue (§C) also
  failed. It only worked once the fonts were replaced by MyBB's **actual 2 bundled fonts**
  (MINYN + edmunds).
- **Evidence:** 32-font PIL synth = **31.18%** word (Model 1 v1); the one-line font swap →
  **39.94%** (**+8.76pt**, ~26pt of synth-real gap closed). The single biggest jump in the
  whole project came from a *font list*, not a model.

**A3. Background-free synth + heavy augmentation · FAILED · `model 1/`**
Render characters on plain white (`data/synth_nobg/`), rely on a heuristic cleaner
(saturation+darkness+Canny+morphology) to strip real backgrounds at inference so reals
“look like” the clean synth; bridge the rest with heavy albumentations.
- **Why it failed:** the bg-free distribution is too narrow; augmentation adds distortions
  the model can't generalize from and the cleaner never makes reals match clean-white synth.
- **Evidence:** word bounced **0.7–1.5%**, never broke 1.5%; `synth_nobg/` deprecated.

**A4. Parametric TTF renderer + DINOv2 cosine auto-tuner · SUPERSEDED · `synthetic image generation/`**
`render_darkbay_v2.py` (16 knobs) + `dinov2_search.py` random-searches RendererConfigs,
scoring each by `1 − cos(DINOv2(real), DINOv2(synth))`; best config becomes the “hard”
anchor.
- **Why it failed:** random search **plateaued early** and 4 Windows TTFs only
  *coincidentally* resemble DarkBay serifs (“a coincidence of the schemes we have, not a
  property of the architecture”). Structural mismatch remained: the renderer styles
  **per-character** (box/plain) while real DarkBay is uniformly all-boxed or all-plain
  **per image**; a single-mean real anchor blurs distinct subtypes.
- **Evidence:** DarkBay 200-trial search best dist **0.189 @ trial 51, plateaued**; no
  recognizer accuracy ever produced. Superseded by char-crop recombination.

---

### Family B — Perturbing the winning distribution-matched renderer (curriculum / difficulty knobs)

Once the MyBB port worked, several ideas tried to *improve* it by adding a difficulty axis.
All of them **drifted the synth off the real manifold** and lost (root cause #4).

**B1. Difficulty-knob generator (`RefinedGenerator`, `t∈[0,1]`) · DROPPED · `synthetic image generation/`**
A single scalar `t` linearly blends an EASY RendererConfig toward a HARD one
(`difficulty.py::interpolate`), meant to feed a 4-model curriculum study.
- **Why it failed:** there was **no valid optimization target for the difficulty axis**.
  The only per-config score (DINOv2 cosine) is a *similarity* metric — it can pick the one
  “looks-real” HARD config but cannot define intermediate *difficulty*. Declared dead:
  *“Difficulty knob and curriculum machinery stay dropped … Permanently dropped.”* No
  recognizer number was ever produced; the curriculum idea survived only *reframed* into
  the recognizer arc (M2's teacher-NLL buckets over a **fixed** pool).

**B2. Difficulty-mode MyBB renderer (entangled `d` knob) + variable-length pools · SUPERSEDED · `model 2/`**
A *separate* PHP fork (`generate_mybb_captcha.php`) plumbs one scalar `d∈[0,1]`
simultaneously into **every** axis — rotation (±30·d), size (26±6·d), color_max (200·d),
shadow, font-pool size, dot count, **and** the background-pattern Bernoulli gate — to
render an easy→hard ladder (`synth_buckets`), plus a `range:3-7` variable-length pool.
- **Why it failed (two causes):** (1) the single `d` **entangles two independent inductive
  biases** — background/figure-ground density and per-glyph distortion — so low-`d` “easy”
  samples are *simultaneously* clean-bg and undistorted-glyph, a distribution that lies
  **off** the real (always-`d=1.0`) manifold rather than a sub-difficulty of it. Documented
  post-mortem: *“the d knob conflates bg-density and char-distortion; ramp them
  independently.”* (2) When your synth already distribution-matches deployment,
  **“the synthetic data IS the test data”** — there is nothing to scaffold, so any ladder
  is just off-distribution data.
- **Evidence:** d-ladder bucket curriculum → **42.82% synth val but only 10.45% real** word;
  another bucket run 60.83% real *char* but **1.08% real word** (joint distribution
  destroyed). Variable-length Run A → **0.00% real word / 1.75% char**, autoregressive
  decoder collapse (EOS at position 0, empty string on 94% of reals). All these pools
  marked *“Stale; safe to delete.”*
- **The negative control that proved the thesis:** the flagship threw the ladder away,
  rendered a **single fixed `d=1.0`/len-5** pool, and bucketed it only by a teacher's
  post-hoc NLL — reaching **54.22%** real word. *Manipulate the learning **order**, never
  the synth's **difficulty distribution**.*

**B3. v3 synth: 2× supersample + mixed-scale · DROPPED · `model 1/`**
Render at 2× and mix 200×60 with 400×120 to “sharpen edges / add scale diversity.”
- **Why it failed:** produced sub-pixel-thin bg lines and over-smoothed text edges real
  200×60 captchas don't have — broke the exact distribution-match that made v2 work.
- **Evidence:** killed on visual A/B (`data/_v2_v3_vs_real.png`) before any training run.

---

### Family C — GAN translation / restyle (synth → real)

The dream: render text you control, then let a GAN repaint it into the real look. Three
independent implementations, three failures — all instances of root cause #2 (style fights
content) compounded by the font wall (#1).

**C1. CycleGAN unsupervised synth→real translation (v1/v2) · FAILED · `model 1/`**
CycleGAN (Zhu 2017) translates PIL synth toward the ~600 unlabeled reals; PARSeq then
trained on `data/translated/`.
- **Why it failed:** adversarial pressure consistently fights content preservation.
  v1's content-preserving checkpoints barely restyled; v2 **collapsed** (D-loss below
  threshold, early-stopped @ep8) — preserved characters but barely changed style. Either
  way the translation injects more noise than the style shift is worth.
- **Evidence:** PARSeq on translated data < PARSeq on raw synth; no config beat the plain
  renderer. Superseded once the PHP MyBB renderer landed (31.18% word).

**C2. CycleGAN-translated set → TinySTR cold-start (v2/v3/v4) · FAILED · `tiny STR/`**
The *same* translated data fed to a from-scratch TinySTR transformer. A clean controlled
comparison: **v1 (raw synth) vs v4 (translated), byte-identical architecture, only the
data differs.**
- **Why it failed:** the translation's high visual entropy means an *untrained* ViT can't
  separate character signal from translated texture during feature formation — it never
  reaches its phase transition. The v1-vs-v4 control isolates the **data**, not the model,
  as the cause.
- **Evidence:** v4 loss frozen at **~20.64** (random baseline = 5·ln62) and val-char
  **1.5–1.7%** (word 0.0000) across all 8 epochs; v1 on raw synth crossed the cliff by ep4
  (5.33→41.30% char) and hit 93.88% char synth-val. (Curriculum pretrain-then-finetune was
  proposed as the fix but never run.)

**C3. Recognition-anchored CUT/PatchNCE restyler — “Route B” (M1→M2→M3) · FAILED · `cryptbb_sourcefree/`**
The most sophisticated attempt: a mask-conditioned U-Net + PatchGAN trained CUT-style
(PatchNCE content loss + mask-L1 + adversarial), and in M2 an **`L_rec` recognition-
consistency loss** forcing a frozen PARSeq to read the correct label out of `G(x)`.
- **Why it failed:** CUT/PatchNCE (and the M2 residual generator) **preserve glyph SHAPE by
  construction** — they repaint color/texture but *cannot morph* a generic-font letter into
  CryptBB's exotic MyBB glyph. So the discriminator never stops separating synth from real,
  and the restyle adds **no font signal**. The font wall (#1) is impassable by a restyler.
- **Evidence:** M1 — as adv weight grew, glyphs faded, readback fell to 3%. M2 — `L_rec`
  drove readback to **82.5% (= source composite's 82.4%, zero legibility loss)** *yet D still
  fully separates* synth from real. M3 transfer diagnostic — the M2-GAN pool **crashes real
  word 39.94%→0.92% in one epoch**; ±GAN ablation: raw composite peaks **4.15%** real vs the
  GAN's 0.92% (**the GAN actively HURT** — it blurred glyphs). Realism probe: GAN pool 18.6%
  parseq_v2-char vs 90.2% on real.

---

### Family D — Diffusion, glyph-conditioned generation & whole-image style transfer

**D1. GlyphControl glyph-conditioned diffusion + style LoRA · FAILED · `model 1 Dino attention/`**
Two-stage, label-*guaranteed* synth: (1) `diffusion_render.py` turns a random 5-char label
into 5 spaced glyph bboxes (± yaw/jitter) fed to a vendored **GlyphControl** (SD-2.0 + glyph
ControlNet) under a fixed CryptBB style prompt — the ControlNet branch was supposed to
hard-wire content, fixing the CycleGAN drift problem. (2) `lora_train.py` fits rank-8 LoRA
adapters on the U-Net cross-attention (`to_q/k/v/out`) to ~600 reals for 2000 steps to
absorb the speckle/vivid-hue/chromatic-shadow *style*.
- **Why it failed:** the base model was pretrained on **natural-scene text**
  (TextCaps/LAION-glyph); CryptBB's heavy speckle/color/shadow is far OOD. Zero-shot it
  produced wrong-style outputs **and drifted content despite the glyph branch**; the LoRA on
  600 images was too weak to pull the pixel distribution to the captcha domain. Result:
  neither realistic nor reliably legible → unusable as training data.
- **Evidence:** **0/5 on PARSeq read-back** — of 5 sampled outputs, PARSeq recovered the
  intended label on **zero**. (Small-sample qualitative probe; no per-step LoRA loss logged.)
  *“The base model was trained on natural-scene text … Track abandoned.”*

**D2. GlyphControl / AnyText / QwenStyle bundle · PARKED · `model 1/`**
A literature sweep of glyph/style diffusion as a renderer alternative.
- **Why it failed:** GlyphControl 0/5 (above); **AnyText** setup never finished;
  **QwenStyle** threw an HF-demo runtime error *and* is 20B params (won't fit the 16GB
  A4000); B-CvT (below) medical-specific. Logged as “not productive.”

**D3. B-CvT skin-lesion style-transfer framework · DROPPED · `papers/`**
The lone PDF: a Bidirectional CvT style-transfer net built to restyle **skin-lesion**
images while preserving diagnostic content — pulled in as a candidate for “restyle renderer
output into an arbitrary scheme's look, content-preserving.”
- **Why it failed:** rejected at **paper triage**, never implemented. (1) Domain-specific —
  its content-preservation is defined around lesion morphology / skin-tone fairness, not
  sparse high-frequency glyphs; nothing suggests it transfers to text where one wrong stroke
  breaks the label. (2) It belongs to the **“content + style jointly”** family the thesis
  concluded is the wrong framing (root cause #2). Code release also unclear.
- **Evidence:** no experiments/metrics exist (dir is just the PDF). Rejection recorded in
  `MASTER_CONTEXT.md`, `model 1/CONTEXT.md`, `synthetic image generation/CONTEXT.md`. It is
  best read as one data point in a **rejected-candidate family** (CycleGAN, GlyphControl,
  AnyText, QwenStyle, B-CvT) all discarded for the same reason.

---

### Family E — Composite (background + pasted text)

Guarantee the label by pasting text you rendered; get realism from a real/inpainted
background. Every variant hit root cause #5 (compositing tells / inpainting hallucination).

**E1. Compbg composite — LaMa real bg + PHP text-only overlay · FAILED · `model 1/`**
Extract real backgrounds by LaMa-inpainting the text away, render CryptBB glyphs separately
(`--text-only`), alpha-composite. 100k pool; full Model 1.5 stack retrained.
- **Why it failed:** the alpha step creates **anti-aliasing artifacts at the text/bg
  boundary** that one-pass real captchas don't have — the recognizer keys on them; and
  sampling bg independent of text **decouples** their interaction, uniformizing errors so
  every captcha gets ≥1 wrong char.
- **Evidence:** **1.23% word / 72.62% char** on real (below even v1's 31.18%). Char signal
  survives; word collapses because errors spread evenly.

**E2. Alpha-compositor over LaMa-extracted bg · PARKED · `synthetic image generation/`**
The generalized version (`compose.py` + `extract_bgs_lama.py`), scheme-agnostic.
- **Why it failed:** two coupled bg-quality defects — LaMa **hallucinates ~20%** of CryptBB
  bgs (bg-pattern blocks at former text positions), and the removal mask is incomplete when
  chars and bg share grey values, so **char fragments survive → double text**.
- **Evidence:** ~65% strict-clean / 80% clean+borderline; fixes (mask dilation,
  detector-union, local inpaint) all deferred. Only reliable once fed the char-crop bank +
  cleaner bg (the synth_v1 win).

**E3. Source-free composite renderer (styled glyphs + procedural/LaMa bg) · MARGINAL · `cryptbb_sourcefree/`**
Generic DejaVu fonts, per-char HSV colors + rotation, over a procedural or filtered-LaMa bg;
labels correct by construction. Used both standalone and as the GAN source.
- **Why it failed:** the **font wall** — no open font matches MyBB glyphs, so a
  font-specific reader caps out. A deterministic realism probe (run `parseq_v2`, 90.2% on
  real, on each pool) ranks all open fonts far below real; rotation makes it worse.
- **Evidence:** best open font (verdana-bold-italic) **37% char**, DejaVu-Bold 31%, arial
  14% — vs real 90.2%; rotation ±12 → 20.8%; full composite 18.7%. Best real transfer
  **~4.15% word** — clears the ≤2.5% OCR floor but caps ~4%. (This is the raw material the
  VLM pivot later turned into 30.4%.)

**E4. Real-background reuse via LaMa inpainting · PARKED · `cryptbb_sourcefree/`**
Harvest authentic backgrounds by erasing real text with LaMa, paste your text over them.
- **Why it failed:** LaMa leaves faint **ghost letters**; a ghost that survives into a bg you
  then label with your word means the image **secretly contains an uncounted letter → your
  own label is WRONG → training poison** (root cause #5). Reliable ghost rejection would
  itself require a reader for the exact missing font — circular.
- **Evidence:** the dark-blob ghost detector keeps only ~124/200 bgs and still “misses faint
  ghosts, false-positives on legit dark bg.” Fell back to purely procedural bg.

**E5. `bg_syth` + `Font_syth` reconstruction tracks · DROPPED · `captcha regenerator/`**
The intended synth half of “regenerate-from-parts”: reconstruct the background *and* recreate
the font, then recomposite with fresh labels.
- **Why it failed:** **never started** — both dirs are empty. Strictly gated on the
  part-extraction phase (§F) producing tight masks, which no method achieved. The concept
  resurfaced (LaMa bg) inside `auto_synth/`.

---

### Family F — Part-extraction / regenerate-from-parts

Rather than *generate*, decompose a real captcha into parts (char mask + bbox + isolated bg
+ font) and re-render from them. The whole line stalled at extraction (root cause #3).

**F1. Part-extraction attack (whole strategy) · SUPERSEDED · `captcha regenerator/`**
Phase 1 = produce a binary char mask + per-char crops + bboxes, scored against a hard gate
(≥95% of images meet count ±1 **AND** IoU≥0.7 **AND** mask P,R≥0.85; pseudo-GT from
Claude/Qwen VLM bboxes). Phase 1 never cleared the bar on **any** scheme, so no clean parts
ever reached the downstream synth stages. Superseded by `auto_synth/`.

**F2. PSSTRNet pretrained scene-text-removal net · FAILED · `captcha regenerator/`**
Drop in PSSTRNet (scut_syn + scut_enstext checkpoints) as a universal char-localizer.
- **Why it failed:** **severe domain gap** — it's trained on natural-scene text; adversarial
  captchas are built to defeat exactly such models. It classifies the **entire image as
  background**. (Other released STR nets — FETNet/PERT/EraseNet — share the training
  distribution and were expected to fail identically, so none were tried.)
- **Evidence:** both checkpoints × all 4 schemes (16 runs, 2026-05-06): **`mask.mean() ≈
  0.999`**; text-prob≥0.5 fraction only **0.0–0.6%**; prob maps nearly all black.

**F3. otsu_cc / SAM / SWT extraction zoo · FAILED · `captcha regenerator/`**
Three “universal” (no per-type code) extractors: Otsu+connected-components+geometry;
saliency/Otsu CCs → box-prompted MobileSAM → union mask; a Stroke-Width-Transform variant.
- **Why it failed:** structurally sane but **geometrically far too loose** — bbox non-overlap
  and L2R order are ~100%, but bbox tightness (IoU) and mask P/R collapse everywhere vs the
  0.7/0.85 gate. No universal method transfers; per-type engineering is unavoidable. Courier
  breaks color assumptions (digits not always darkest) so SAM misses nearly all digit pixels.
- **Evidence:** **all OUT_OF_SCOPE** (0.00% per-image pass). otsu_cc mean IoU DarkBay 0.276 /
  City 0.146 / Courier 0.352 / CryptBB 0.300; SAM 0.214/0.167/0.126/0.361 (Courier SAM
  catastrophic: 0% count acc, mask-R 3.8%); SWT 0.245/0.187/0.121/0.198. (Caveat: pseudo-GT
  bboxes are eyeballed with ~5–10px error and n=10/type, so absolute IoU is a **noisy lower
  bound** — but the method *ranking* and the qualitative verdict are firm.)

**F4. `masking/` — VLM color-gate → HSV or DINO-attention mask · PARKED · `captcha regenerator/`**
“Gate first, mask second”: Qwen-VL returns char/noise colors; if color known → HSV inRange +
morphology; if unknown → DINO ViT-S/8 self-attention + Canny threshold; a coverage validator
loosens params until ≥80% of attention blobs are covered.
- **Why it failed:** **never scored against the IoU/P/R gate** and never reached PASS; the
  only aggregation script just tallies VLM color/noise distributions, not mask accuracy. It
  inherits brittle assumptions (VLM color guess can be “unknown”; DINO attention is coarse
  saliency, not a tight stroke mask). Effort pivoted to `auto_synth/` before validation.
- **Evidence:** *“Parked. No method reached PASS.”* Stage-2.5 only tracks a self-referential
  “final_coverage,” not agreement with any GT.

**F5. Real-glyph-crop recomposition (cluster real crops) · DROPPED · `cryptbb_sourcefree/`**
Bypass the font wall directly: extract 2411 real bg-removed glyph crops (the *exact* real
font), cluster into ~50 groups, assign a char per cluster, composite your text from real
glyphs.
- **Why it failed:** unsupervised clustering **cannot produce pure single-char groups** —
  pixel clusters mix 2–3 chars, and MAE-feature clustering is no purer (MAE encodes
  color/style/distortion, not character identity). Reliable per-crop labeling would need a
  model that already reads the real font — the exact missing thing. **Circular.**
- **Evidence:** pixel cluster 0 (“9”) holds g/Q/8/b; cluster 3 is the 1/I/l bar group;
  MAE-space clusters mix q/Q/I/u/S/K/X/A/b. Crops usable; label assignment unsolved.

---

### Family G — Refiner GANs

**G1. Mask-aware refinement GAN (U-Net + PatchGAN), 4 versions · MARGINAL · `synthetic image generation/`**
A pixel refiner: a 4-down/4-up mask-aware U-Net (RGB composite ⊕ text mask → refined RGB,
v3+ residual) with a 70×70 PatchGAN discriminator vs real captchas and L1 on the text-mask
region to protect content — meant to add the un-parameterizable realism (JPEG noise,
anti-alias kernel, color bleed) the renderer can't.
- **Why it failed:** it **wasn't load-bearing** — accuracy contribution within noise despite
  stable training (no mode collapse). It also depended on real DarkBay for the discriminator,
  which was the eval-contaminated bg set. Dropped from the go-forward plan.
- **Evidence:** *“v1 refiner showed marginal lift; not load-bearing … Refiner dropped.”*
  v4 trained cleanly (step 15000: G=5.90, G_l1=0.065 flat, D=0.216, no divergence) but **no
  recognizer-accuracy delta was ever logged** for any of the 4 checkpoints.

---

### Family H — Label-free recovery from unlabeled reals (SSL / self-training)

The most principled route to “get the real font's shapes without labels.” All three hit the
cold-start / minimum-pool floor (root cause #6) on the pathological CryptBB.

**H1. From-scratch / MAE-init PARSeq on source-free synth · FAILED · `cryptbb_sourcefree/`**
Train a downsized PARSeq (from scratch or from a source-free MAE encoder) on 60–100k
generic-font synth, score 651 real.
- **Why it failed:** a **font-specific** recognizer learns the generic-font shapes exactly
  and cannot read the unseen MyBB font (font wall). The synth being wrong-font is the root
  cause; the recognizer just faithfully overfits it.
- **Evidence:** **0% word / 18% char** on real (18.5% after MAE); `metrics.csv`
  `val_accuracy=0.000` across logged epochs. Contrast: the *same synth* into a font-general
  VLM reached 22–30.4% word — proving the wall is specific to font-specific recognizers.

**H2. MAE self-supervised pretrain + progressive self-training · FAILED · `cryptbb_sourcefree/`**
GeeSolver-style: MAE-pretrain the encoder on 600 unlabeled reals (learn real-font features
label-free), bootstrap on source-free synth, then progressively pseudo-label the 600 reals
(confidence-filter, add, iterate) to close the font gap.
- **Why it failed:** **cold-start barrier** — the bootstrap is 0% *word*, so at conf≥0.9 it
  pseudo-labels essentially nothing and has no confident seed to compound. CryptBB is the
  pathological worst case (even OCR ≤2.5% word).
- **Evidence:** MAE itself worked (recon-mse 0.27→0.015; lifted char reading 4%→18.5%) **but
  bootstrap = 0% word**; self-train round 1 pseudo-labeled only **1/600** reals at conf≥0.9
  (mean weakest-char conf 0.39) → blocked. Only the font-general Qwen path could seed the
  flywheel (40→98→123 pseudo-labels).

**H3. Disjoint 100-real self-train subset (minimum-pool test) · FAILED · `cryptbb_sourcefree/`**
Self-train the *working* Qwen on a small disjoint 100-image pool to probe the pool-size floor.
- **Why it failed:** below the flywheel floor — at ~30% word only ~16–19 reals cross the
  confidence bar, some are **confident-but-wrong** (confirmation bias), and `real_frac=0.5`
  draws half of every batch from those ~16 images (memorization).
- **Evidence:** round 0 **30.31%** → R1 (16 labels) **29.40%** → R2 (19 labels) **27.59%**
  (declining). The full 600-pool run (~89–141 correct labels) instead *climbs* to 30.41%.
  **Conclusion: a hard floor of ~89+ correct pseudo-labels.**

---

### Family I — VLM-perception synth pipelines that underperformed / were superseded

**I1. synth_v2 — CRAFT + Qwen2.5-VL-7B + Consensus-Entropy bank builder · SUPERSEDED · `synthetic image generation/`**
The planned scheme-agnostic replacement for synth_v1's CryptBB-specific perception: CRAFT
bboxes + two Qwen-VL passes filtered by consensus-entropy (keep only where both passes agree
on string and per-char IoU>0.5); survivors become the alpha glyph bank.
- **Why it failed:** **never carried to a validated result** — `synth_v2/STATUS.md` is empty,
  no char/string-acc ever logged. The plan itself flags the fatal risks: captcha is
  low-frequency text for Qwen-VL (expected ~70% char vs parseq_v2's ~87%) and CRAFT is
  scene-text, not captcha glyphs. `auto_synth/` executed the same idea end-to-end and
  replaced it.
- **Evidence:** smoke output only — Qwen returned 4 chars (N,I,M,i) for a sample (miscount).

**I2. Generalized measured-style auto-fit generator · DROPPED · `cryptbb_sourcefree/`**
`analyze_pool.py` measures color/size/wave/bg stats from the unlabeled pool and auto-drives
`php_gen.php`, to remove the hand-tuned per-scheme `SCHEME_CFG` and make the generator
point-at-a-folder self-configuring.
- **Why it failed:** introduced **visual regressions** (DarkBay bg came out CryptBB-like,
  letter size over-read) needing several fix rounds, and — never trained/scored — had **no
  evidence it moved the number** (image realism is only a low-single-digit-word% lever on the
  VLM path). Reverted at the user's request to the hardcoded generator.
- **Evidence:** built + visually iterated then **reverted (2026-07-05)** to the versions that
  produced the 29.8% (CryptBB) / 86.5% (DarkBay) checkpoints. No auto-fit train/score numbers
  ever produced. *(Verdict noted as “measure-don't-train is sound but unproven on the score.”)*

---

### Family J — `auto_synth`: the near-miss (partial failures + the unrun validation)

`auto_synth/` is the newest, **mostly-working** label-free system (Part A alphabet/length
discovery, Part B glyph bank, Part C LaMa bg). It runs end-to-end on 4 schemes in ~30 min and
**City is “nearly indistinguishable from real.”** But its wins are **visual only**, and it
carries real partial failures.

> **⚠ THE SINGLE BIGGEST OPEN WEAKNESS — the gold-standard test was NEVER run.**
> `auto_synth`'s entire quality claim rests on visual A/B panels (`notebooks/*_compare.png`).
> The stated validation criterion (`GOAL.md`) is: *train a PARSeq on
> `auto_synth/data/<scheme>/synth_pool/` and eval on a held-out labeled set — head-to-head vs
> Model 2.5's 57.60% word.* Both `RESULTS.md` and `CONTEXT.md` state plainly: **“No training
> of a recogniser on auto_synth output yet … Until that runs, ‘the synth is good’ rests only
> on visual A/B panels.”** The pools hold only **2000 samples each** (plan: bump to 10k–50k).
> *Every “works” claim for auto_synth is unvalidated against labels.* **This is the top
> priority to close.**

**J1. Courier composite synth — LaMa can't erase the wavy underline · MARGINAL.** Part C adds
long-thin horizontal components to the LaMa mask to wipe Courier's underline, but LaMa fills
*small holes* well, not *thin curving paths* — it hallucinates underline-shaped texture back
in. Courier is the “weakest match” (visual only; proposed fix — *render a matching synthetic
underline instead of erasing the real one* — not done). `src/part_c/extract_bgs.py`.

**J2. OCR-direct discovery (easyOCR reads whole captcha) · SUPERSEDED.** On distorted schemes
easyOCR hallucinates: CryptBB → **91 junk chars** (full punctuation incl. `!"#$%&`, a
replacement char), mean conf 0.477, length spread 1–9 — poisoning both the char bank and the
length distribution. Superseded by a hybrid rule (use OCR only if `mean_conf>0.5`, else CC).
`src/part_a/ocr_discover.py`, `run_ocr_discovery.log`.

**J3. Four Part-C background variants for the underline — three parked/dropped.** All attack
the *same* Courier residue; only the simplest **CC-based `extract_bgs.py` feeds the final
pool**:
- **MobileSAM box-prompt** (`extract_bgs_sam.py`) — DROPPED: ran 200/scheme but only a
  marginal mask gain for +2.5GB / ~5× slower; A/B only, never used downstream.
- **Qwen2.5-VL-3B detector + SAM** (`extract_bgs_vlm.py`) — PARKED: run **only on Courier**,
  never scaled, never adopted; log truncates mid-run.
- **GroundingDINO / Grounded-SAM iterative** (`extract_bgs_grounded.py`) — PARKED: recommended
  in `setup.md` but only run on Courier, never wired into the pool.
- **Root wall shared by all four** (`setup.md`): *“the residue isn't a mask problem … bigger
  SAM doesn't address LaMa's hallucination.”* A better detector cannot fix inpainting
  hallucination (root cause #5).

**J4. CC+DINOv2+KMeans glyph bank — systematic coverage gaps · PARTIAL.** The bank is a
*sampler over real crops with no generative model*, so it can only render chars that were both
segmented **and** OCR-mapped. **Digit ‘6’ was never recovered for Courier or City** → those
synth pools *literally cannot contain a 6*; CryptBB injected an OCR-misread ‘{’ as a fake
glyph. `src/part_a/map_clusters.py`, `src/part_b/glyph_bank.py`.

**J5. Naive compose layout + under-counted length · PARTIAL.** `compose_one()` lays glyphs in
even horizontal slots (±3px y-jitter), but real captchas use wave/overlap/non-uniform spacing —
the geometric distribution diverges. CC-count length inference under-counts under heavy noise
(merged components): **CryptBB true length 5 comes out modal 4** (only 33% at 5).
`src/compose.py`, `src/part_a/infer_length.py`.

---

## 4. What actually worked (so the failures have meaning)

The graveyard exists because exactly a few boring things beat everything fancy:

1. **Distribution-matched renderer.** A line-by-line PHP port of MyBB's `captcha.php` using
   the target's *own 2 fonts* → Model 1 (39.94% w), Model 1.5 ensemble (49.16%), Model 2.5
   (57.60%). The **+8.76pt font swap** was the biggest single jump in the project.
2. **Char-crop bank recombination (synth_v1).** Pseudolabel reals → per-char alpha crop bank →
   recompose over LaMa bg. Visually matches real CryptBB; the conceptual seed of `auto_synth`
   Parts A/B/C. (Blocker: hardcoded 5 boxes + CryptBB-trained perception → CryptBB-only.)
3. **The font-general VLM pivot — the actual answer to “synth for any captcha.”** Fine-tune a
   pretrained **Qwen2-VL-2B** on source-free synth, then self-train on unlabeled reals.
   Source-free **and** label-free: **CryptBB 30.4% word** (~12× the OCR floor; ~76% of
   source-*based* Model 1), **DarkBay 91.0% word / 97.3% char** — one pipeline, swap the
   unlabeled folder. This is the only route that walked *through* the font wall instead of
   dying on it.
4. **Two independent source-free levers (~+4–5% word each on CryptBB):** self-training on
   ~600 unlabeled reals (25.4→30.4), and a faithful **PHP-GD render** (imagettftext + sine
   warp + noise) (25.4→29.8). Finding: **render-pipeline match > visual fidelity.**

---

## 5. Do-not-retry list (unless you have a genuinely new idea)

- ❌ **Generic multi-library / multi-font corpora** for breadth (model 0: 1.08%). Match one
  scheme's distribution instead.
- ❌ **Perturbing a working distribution-matched renderer** — supersample, difficulty ladders,
  bg-free, entangled `d`-knobs. They drift you off the real manifold (10.45%, 0%, 1.5%).
  If you want curriculum, reorder a **fixed** matched pool (teacher NLL), don't re-render.
- ❌ **GAN translation/restyle to change the font** (CycleGAN, CUT/PatchNCE). Shape-preserving
  by construction → cannot cross the font wall → 0.92–3% word, often *hurts*.
- ❌ **Glyph-conditioned diffusion / whole-image style transfer** (GlyphControl, AnyText,
  QwenStyle, B-CvT). Content drifts (0/5); base models are natural-scene-text OOD; some don't
  fit 16GB.
- ❌ **Pretrained scene-text nets / OCR as perception on captchas** (PSSTRNet → all background;
  easyOCR → 91 junk chars). They're trained on the exact thing captchas defeat.
- ❌ **Alpha-composite (real/inpainted bg + pasted text) as a primary route.** Boundary
  artifacts (1.23%) and inpainting **ghost letters that poison your labels**. Only viable
  atop a *clean* bank + carefully-masked bg.
- ❌ **Unsupervised glyph clustering to auto-label real crops.** Clusters are impure; correct
  labeling is circular (needs the reader you lack).
- ❌ **Self-training on the pathological scheme from a 0%-word bootstrap, or on a <~89-label
  pool.** Cold-start stalls; small pools *decline* via confirmation bias. Seed with a
  font-general VLM first.
- ⚠ **Refiner GANs** buy marginal, non-load-bearing lift — don't build the pipeline around one.
- ⚠ **auto_synth's visual wins are unvalidated.** Before trusting any of them, **run the
  gold-standard downstream test** (train on `synth_pool/`, eval vs 57.60%). And note the
  known holes: missing digit ‘6’ (Courier/City), Courier underline, under-counted length,
  naive layout.

---

### Provenance
Compiled from a fresh crawl of `model 0/`, `model 1/`, `model 1 Dino attention/`,
`model 2/`, `model3/`, `captcha regenerator/`, `synthetic image generation/`,
`cryptbb_sourcefree/`, `auto_synth/`, `tiny STR/`, `papers/`, `synth_data/`,
`recreate syth/` (empty), plus `MASTER_CONTEXT.md`. Every number above is quoted from that
directory's `CONTEXT.md` / `research.md` / `notes.md` / `RESULTS.md` / results-`.md` /
`metrics.csv` / run logs. Where a number is a small-sample probe (GlyphControl 0/5) or a
noisy lower bound (extraction IoU), it is flagged inline.
