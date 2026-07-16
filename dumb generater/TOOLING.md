# Dumb Generater — Per-Stage Tooling Map (from a deep sweep of all prior attempts)

> Companion to `PLAN.md`. Produced by mining every prior CAPTCHA-synth attempt under
> `e:/pranav` (generation/, cryptbb_sourcefree/, auto_synth/, synthetic image generation/,
> captcha regenerator/, model 0/1/2/3, tiny STR/) plus the 638-line failure post-mortem
> `generation/FAILED_SYNTHESIS_ATTEMPTS.md`. Every claim below cites a file or a logged
> number. Status tags: **reuse-as-is** / **declassify** (strip a VLM/OCR/real-label call, keep
> the classical core) / **reference** / **do-not-reuse** (VLM-contaminated or R1-violating).
>
> Constraint recap: **R1** no labels on real (not even pseudo-labels); **R2** no LLM/VLM/
> language-supervised encoder (CLIP/SigLIP/BLIP/Qwen); **R3** input is one folder of unlabeled
> reals; **R4** output is labeled synth good enough to train a non-trivial solver;
> **R5** no captcha-specific libraries (no `captcha`/`claptcha`/`securimage`/`gregwar`/`kcaptcha`/
> `coolcaptcha`/`simplecaptcha`/`multicaptcha-gen`, or any vendored copy) — build only from
> general graphics/CV primitives (Pillow, OpenCV, NumPy, raw PHP-GD `imagettftext`).

> **What R5 filters out of this map (added 2026-07-10):** the A2 proxy route and Stage-3 "optional
> lib wrap" that leaned on captcha libraries are **removed** — a proxy must now be **hand-rolled**
> (Pillow/GD + your own fonts/distortions). What is **unaffected:** the hand-written renderers
> `cryptbb_sourcefree/src/{render_source,distort,background}.py`, `php_gen.php`/`php_gen_ext.php`
> (GD `imagettftext` is a graphics primitive, not a captcha lib), and all classical Stage 1/2/4/5/6
> code. Only the *generator library* dependency is cut, not the machinery. Font `.ttf` files remain
> usable data (a font is not a library); see the font note in §3 for the captcha-lib-bundled fonts.

---

## 0. The one finding that dominates the whole plan: the Font Wall vs. R2

Everything that *worked* on the hard scheme in prior work relied on the exact thing this plan
bans. This is not a detail — it sets the ceiling.

- **The wall (root cause #1, `FAILED_SYNTHESIS_ATTEMPTS.md` §1):** source-free rendering cannot
  reproduce an exotic scheme's glyph *shapes*. Measured: no open font exceeds **~37%
  char-agreement** with CryptBB's real MyBB font (verdana-bold-italic 37%, DejaVu-Bold 31%, arial
  14%) vs **90.2%** for the real font itself (`cryptbb_sourcefree/notes/RESEARCH_FONTWALL.md`).
  And a font-specific solver (PARSeq/CRNN/TinySTR) memorizes its training glyphs and **cannot read
  an unseen font**.
- **The consequence, quantified:** a source-free generic-font composite renderer caps at
  **~4.15% word** on CryptBB (E3), and a from-scratch/MAE PARSeq trained on that synth reads
  **0% word / 18% char** on real (H1). Generic 5-library union (model 0) peaked **1.08% word**.
- **The only escape in the entire 36-attempt graveyard was a font-general pretrained VLM**
  (Qwen2-VL-2B → **30.4% word** CryptBB). **R2 bans exactly that.**
- **So:** on a **hard exotic-font scheme, this plan is capped at single-digit word accuracy**, and
  no amount of Stage 4/5 tuning moves it (Stage 5 tunes *appearance*, not glyph shape — "image
  realism is a low-single-digit-word% lever", I2). On an **easy common-font scheme it works well**:
  the same pipeline hit **DarkBay 91.0% word / 97.3% char** and City is visually
  "near-indistinguishable" — because a classical font search can land a near-exact open font
  ("Windows TTFs coincidentally resemble DarkBay serifs").

**Strategic recommendation (do this or the thesis measures nothing):**
1. Bring up M1–M6 on an **A2 proxy scheme you hand-roll yourself** (Pillow/GD + your own fonts and
   distortions — **not** a captcha library, per R5) — you control the font, so the wall vanishes and
   you can measure the pipeline objectively.
2. Then prove generality on an **easy real scheme** (DarkBay 799-pool / 201-eval, or City).
3. Run **CryptBB as the honest worst case** and *report the ceiling* — do not expect to beat the
   30.4% VLM number, and **do not let an A2-proxy "win" masquerade as solving the source-free hard
   problem.** The proxy only proves you can match a font you already have.

---

## 1. Environment / setup (Stage 7 — do this first)

Verified live against the shared venv `model 1/.venv` (Python 3.9.13, torch 2.5.1+cu121, RTX A4000
16 GB, HF_HOME=E:/hf_cache, PHP 8.2.12 GD+FreeType at C:/xampp/php).

**Build a fresh isolated venv (ideally py3.10+) for this project — do NOT extend `model 1/.venv`.**
Two reasons: (a) it keeps `qwen_vl_utils` + the transformers/Qwen VLM cluster out **by
construction**, satisfying R2 structurally; (b) it avoids numpy-2.0.2 / py3.9 conflicts when adding
`lightly`/`cma`/`kornia`. Note the shared venv's Anaconda Python has a **broken PIL** (`_imaging` DLL
load) — another reason to start clean.

| Need | Status in `model 1/.venv` | Action |
|---|---|---|
| opencv, numpy, scikit-learn, **scikit-image (Sauvola/Niblack)**, scipy, Pillow | ✅ INSTALLED (all Stage 1–3 needs verified importable) | reinstall in fresh venv |
| torch/torchvision cu121, timm, pytorch-lightning, torchmetrics, albumentations | ✅ INSTALLED | reinstall |
| `lightly` or `solo-learn` (Stage 4 SSL) | ❌ absent from all 3 venvs | `pip install` **or hand-roll SimCLR on torch+timm** (zero-dep) |
| `cma` (Stage 5A CMA-ES) | ❌ absent | `pip install cma` (tiny, pure-Python) |
| `kornia` (optional diff-aug) | ❌ absent | optional |
| CUT / CRNN / parseq pip pkgs | ❌ no pip pkg exists | reuse in-repo trees (see Stage 6) |
| `captcha` / `claptcha` / any captcha lib | ❌ absent | 🚫 **BANNED by R5 — do not install.** Hand-roll the A2 proxy in Pillow/GD instead. |

Gotchas: **opencv is the headless build** → the M3 eye-test and M5 t-SNE must save PNGs (Pillow/
matplotlib), never `cv2.imshow`. A4000 16 GB is plenty for ResNet-18/ViT-tiny SSL + CRNN/PARSeq (no
VLM co-resides). If you touch the old venv, torch-adjacent imports need
`os.add_dll_directory('C:/ProgramData/Anaconda3/Library/bin')` before import (sqlite shim).

---

## 2. Per-stage tool map

### Stage 1 — Ingest & normalize
**Tools:** OpenCV (resize/letterbox, Canny, Otsu), NumPy (mean/std, **FFT log-magnitude** — add
this, it's missing everywhere), scikit-learn KMeans (8-color palette), Pillow (PNG round-trip).

**Reuse (all VLM-free):**
- **declassify** `generation/director.py:75-98` — `_classical_difficulty` (Canny density + color
  count + gray std) and `_dominant_colors` (border-median bg + farthest-15% fg → bg/glyph/noise
  hex). Drop the `/2` VLM fuse; swap the median for k-means k=2 per the plan.
- **reuse** `generation/pipeline.py:57-72` `_list_images` + `_median_canvas`; `font_atlas.py:45-64`
  `to_black_on_white` (Otsu→autocrop→square-pad→resize, the canonical normalizer).
- **reuse** `cryptbb_sourcefree/src/data.py` (resize-to-canvas + tensor helpers); `tiny STR/src/data.py`
  `CaptchaDir` (canonical resize + normalize + PNG round-trip); `captcha regenerator/.../methods/classical.py`
  (to_gray/otsu_mask/detect_polarity/morphology).

**Do NOT:** upsample above native resolution. The plan's proposed 64×192 canvas is an *upsample* for
a 200×60 scheme; 2× supersample "produced sub-pixel-thin bg lines + over-smoothed edges real captchas
don't have — killed on visual A/B before training" (B3). **Match native size** (CryptBB/Courier
200×60, City 324×81, DarkBay 200×50). Detect aspect ratio per scheme; `_median_canvas` samples only
first 30 paths — histogram all sizes and **letterbox, never crop** (glyph clipping).

### Stage 2 — Unsupervised style profiling
**Tools:** cv2 `connectedComponentsWithStats` + morphology (glyph count/bbox), `distanceTransform`
(stroke), PCA on CC pixels (rotation), `HoughLinesP` on CC-removed residual (clutter), NumPy FFT
(noise type), `scipy.optimize.least_squares` (sine-warp `y=A·sin(ωx+φ)` fit to CC centroids),
**skimage `threshold_sauvola`/`threshold_niblack`** (uneven lighting), sklearn KMeans in **Lab space**
(colored-on-colored). All verified installed.

**Reuse (the star module first):**
- **reuse-as-is** `generation/extract.py:20-102` — `_components`/`char_crops`/`count_chars` (dual-
  polarity Otsu → MORPH_OPEN → CC → geometry filters → L→R mask-applied crops, picks polarity by
  best CC-count fit) **and** `length_modes` (weighted length-mode voting). This *is* the plan's
  Stage-2 CC segmentation + length distribution, already written and VLM-free.
- **reuse-as-is** `auto_synth/src/part_a/{segment.py,cluster.py,infer_length.py}` — CC segmentation,
  **KMeans shape clustering that produces ANONYMOUS clusters (`cluster_000…`) with no char names**
  (exactly the plan's "cluster shapes without naming"), and CC-count length histogram. R1-clean.
- **reuse-as-is** `captcha regenerator/char_extraction/src/.../postproc/components.py` +
  `methods/otsu_cc.py` + `methods/classical.py` (`remove_horizontal_lines` = the Hough-clutter
  precursor; `top_hat_dark_on_light` for thin strokes).
- **declassify** `generation/font_atlas.py:67-131` + `configs/font_pool.json` — `render_glyph` (Pillow
  draw of a candidate font) + **69 vetted generic TTFs**. This is the plan's "render ~200 candidate
  fonts, score by IoU/shape-context" font-guess. **Replace the DINOv2 scorer with classical
  IoU/shape-context** (Hu moments / HOG / Zernike / shape-context).
- **declassify** `generation/director.py:38-69` `Spec` dataclass — its fields (bg pale/dark, texture,
  serif, weight, case, wave, noise, colored, colors{bg/glyph/noise}, length_modes) **are** the
  `scheme_profile.json` schema; populate from classical measurements instead of the VLM.
- **reference** `cryptbb_sourcefree/src/php_pool.py` `SCHEME_CFG` (hand-authored profile dict) — a
  template of *which* parameters matter; and `background.py` bg palettes as sanity ranges.

**Do NOT:**
- Assign a *character identity* to any real crop. auto_synth's `map_clusters.py`/`ocr_discover.py`
  run easyOCR on real crops to name them → **R1 violation**. Identity must flow **generated→real**
  ("I rendered 'A' — which cluster matches?"), never real→label.
- Trust classical font-guess as truth on exotic schemes — it returns a ~37%-match font that a solver
  can't read (there is no classical fix; the VLM escape is banned). Use `RESEARCH_FONTWALL.md` as a
  **pre-flight screen**: measure whether *any* open font is close enough before committing to a scheme.
- Trust CC length blindly: it **under-counts** under noise/merges (CryptBB true len 5 → modal 4, only
  33% at 5; digit '6' never recovered for Courier/City). Use vertical-projection valley cuts + an
  expected-length prior; **force full alphabet coverage in Stage 3** (you render your own strings).
- Average a multi-modal scheme into one style mean (all-boxed vs all-plain DarkBay subtypes blur; the
  DINOv2 search plateaued at dist 0.189). Cluster real embeddings before trusting a single anchor.

*Optional fallbacks (R2-clean but not the default):* MobileSAM (`auto_synth/.../extract_bgs_sam.py`,
`configs/mobile_sam.pt`) for dark-bg glyph masks — marginal (IoU 0.36 CryptBB, ~0 Courier); the
spectral-residual saliency prompt-gen in `universal_sam.py` is pure classical and reusable.

### Stage 3 — Procedural labeled generator
**Tools:** Pillow `ImageDraw.text` + TTF per-char alpha layers, NumPy/OpenCV for sine-warp, per-char
rotation, gradient/noise/texture bg, JPEG re-encode. Optional raw PHP-GD `imagettftext` (the strongest
prior renderer — a graphics primitive, R2/R5-clean). **No captcha-library wrappers (R5).**

**Reuse (richest vein — `cryptbb_sourcefree/src/`):**
- **reuse-as-is** `render_source.py` (per-char glyph strip: HSV color, per-char rotation, size/spacing
  jitter, binary mask, generic DejaVu — labels correct by construction) + `distort.py`
  (`sine_warp`/`add_lines`/`add_arcs`/`add_dots` + named PHP-GD `PROFILES`, pure NumPy/scipy, **no
  xampp dependency**) + `background.py` (`procedural_bg`/`dark_block_bg` + `_has_ghost_glyph` label-
  safety filter) + `data.py` `SourceDataset` (wires renderer+bg+distort → `(rgb, mask, label)`).
- **reuse-as-is** `synthetic image generation/src/auto_tune/renderer_config.py` (16-knob dataclass +
  `sample_random` + JSON round-trip — doubles as the Stage-5A search space) + `render_darkbay_v2.py`
  (Pillow per-char RGBA + rotate-expand + boxed/plain + labels.csv) + `composite/compose.py`.
- **declassify** `generation/php_gen_ext.php` (one-pass `make_bg`+`imagettftext`+`wave_warp`+`add_noise`,
  **renders text+bg in ONE pass — no compositing seam**) + `pipeline.py:80-184` orchestration
  (sample length→sample text→render→labels.csv; strip the one `director.direct()` VLM call) +
  `compose.py:23-70` `to_alpha_smart` layout (feed it **TTF glyphs, not real crops**).
- **reference** the char-crop banks (`synth_v1/`, `auto_synth` GlyphBank, `generation/build_bank.py`).
  These were the repo's *winning* strategy — but they paste **real captcha pixels** (forbidden by the
  plan's TTF-render Stage 3) and their identities came from easyOCR (**R1**). Lift only the classical
  augment primitives (color/size/rotation jitter, `to_alpha`).

**Do NOT (each backed by a logged number):**
- **Alpha-composite text over a sampled bg** (PLAN steps 3–5 do exactly this). Boundary anti-aliasing
  tells → **1.23% word / 72.62% char** (E1/compbg). **Render text+bg in one pass** (php_gen_ext /
  imagettftext family). *This is the single most important correction to the current PLAN.*
- Union multiple renderers/fonts/lengths for "breadth" → **1.08% word** (model 0). Match **one** scheme.
- Add a difficulty knob / curriculum axis → **42.82% synth but 10.45% real** (B2). "When your synth
  distribution-matches deployment, the synth IS the test data — nothing to scaffold."
- Render **variable length** → **0.00% word** decoder collapse (EOS at position 0). Lock length.
- Over-rotate: ±12° cut char transfer 30.2%→20.8%; keep `max_rotation` modest.
- Chase prettiness: **pipeline-match > visual fidelity** — a stronger/uglier wave that matched the
  real PHP-GD pipeline gave **+4.4% word** (25.4→29.8) over a milder pretty one.
- Forget per-worker RNG reseed in the DataLoader, or every worker renders identical images
  (`data.py` fixes this; replicate it).

### Stage 4 — Self-supervised critic (from scratch)
**Tools:** `lightly` or `solo-learn` (SimCLR/BYOL) **or hand-roll on torch+timm**; ResNet-18 or
ViT-tiny from scratch; mild color jitter + crop + blur + JPEG augs (skip glyph-destroying augs).

**Reuse:**
- **declassify** `cryptbb_sourcefree/src/mae_pretrain.py` — **from-scratch MAE** (mask 0.75) on the
  unlabeled pool, **R1 guardrail already baked in** (pretrain on unlabeled, never eval). Proven
  feature learner: recon-mse 0.27→0.015, lifted char reading 4%→18.5%. Swap its strhub-ViT *body* for
  a standalone ViT-tiny to fully decouple. A valid Stage-4 alternative to SimCLR/BYOL.
- **reuse** `tiny STR/src/model.py` `TransformerEncoderBlock` — a ~150-line from-scratch ViT (Conv
  patch-embed + 6× pre-norm attention), a ready ViT-tiny body for a SimCLR/BYOL critic (drop the head,
  attach a projection head). Random-init → R2-clean.
- **reuse** `auto_synth/src/part_a/cluster.py` — feature-source-agnostic KMeans/silhouette; point it at
  your from-scratch SSL embeddings unchanged.

**Do NOT:** reach for pretrained **DINOv2** as the shortcut critic. It is R2-clean *by the letter*
(self-supervised, no language) but **violates the plan's "from scratch" mandate**, and empirically it
plateaus as a domain signal (dist 0.189). Allowed only as a throwaway diagnostic baseline, never in
the tuning path. Expect: the critic learns real-font *features* but **cannot read words by itself**
(MAE features working, still 0% word) — it is a *style yardstick* for Stage 5, nothing more. Guard
against overfitting to JPEG-level/mean-color (strong color aug); fold synth into the SSL pool
(label-free, fair) to pad the small real pool (<500 is noisy; CryptBB is only 600).

### Stage 5 — Unsupervised domain refinement
**Ship 5A only.** Tools: `cma` (CMA-ES over discrete knobs: font, bg type) + Adam on continuous knobs;
**custom MMD or CORAL loss over Stage-4 critic features** (synth batch vs real batch — both unlabeled,
R1-legal). Label-preserving because you re-generate from Stage 3; pixels never touched.

**Reuse:**
- **declassify** `synthetic image generation/src/auto_tune/dinov2_search.py` — the "render candidate →
  score against real anchor → keep best config JSON" loop **is** the Stage-5A tuner template. Swap:
  DINOv2 encoder → from-scratch critic; cosine → MMD/CORAL; random search → CMA-ES.
- **reuse** `renderer_config.py` search ranges + `generation/director.py:23-69` discretized knob schema
  ("no free-form floats — they make critic loops oscillate").
- **5B only if 5A plateaus AND the gap is clearly appearance:** `cryptbb_sourcefree/src/model.py`
  (UNetRefiner + PatchDiscriminator + PatchSampleF) + `losses.py` (PatchNCE) + `train_restyler.py`
  (the recognition-anchored CUT loop = the guard the plan prescribes); or `model 1/src/cyclegan/`.

**⚠ Critical — the plan's Stage-5B guard is INADEQUATE (sharpest finding of the sweep):** the exact
experiment the plan proposes (recognition-anchored CUT) is in the graveyard (Route B, C3). It drove
CRNN readback to **82.5% — content fully preserved, the plan's ">2% char-drop" guard would PASS** —
**yet the discriminator still fully separated synth from real and real transfer CRASHED 39.94% →
0.92%** (worse than the un-refined 4.15% composite; the GAN *actively hurt*). The CRNN guard only
detects content *drift*; it is structurally blind to "content preserved but font still generic/wrong",
which is precisely how CUT fails on a font-mismatched scheme. **CUT/CycleGAN are shape-preserving by
construction and cannot cross the font wall.**

**Do NOT:** build the pipeline around a refiner (all 4 prior refiner-GAN versions gave lift "within
noise", no accuracy delta ever logged, dropped). Don't expect Stage 5 to rescue a font-wall miss —
MMD/CORAL is a low-single-digit-word% lever; the prior measure-then-tune auto-fit was **reverted with
"no evidence it moved the number"** (I2). **Gate every tuning decision on the 200-image eval, not the
MMD/t-SNE number.** Note: **no MMD/CORAL exists anywhere in the repo — build it fresh.**

### Stage 6 — Solver training + evaluation
**Tools:** **TinySTR** (from-scratch, preferred for fixed-length) or **PARSeq** (variable-length /
word-acc) or a custom CRNN; `albumentations` (**mandatory** augmentation); ResNet-18 char-CNN ensemble
for a cheap word-acc boost; report **word AND char** on the 200 hand-labeled eval.

**Reuse (all R2-clean — random-init, no pretrained STR weights, no language supervision):**
- **reuse-as-is** `tiny STR/src/{model.py,train.py,data.py}` — 5.67M from-scratch transformer;
  matches PARSeq char-acc at 1/4 params; 93.88% synth char / 73.17% real char / 22.27% real word
  alone; ~75 min on the A4000. **Default for fixed-length schemes.**
- **reuse-as-is** `tiny STR/src/eval_ensemble.py` + `model 1/src/charcls/model.py` (ResNet-18 char-CNN,
  `pretrained=False`) — ensemble lifted TinySTR **22.27% → 31.03% real word**.
- **declassify** `model */src/parseq/train.py` + vendored `third_party/parseq` — heavier (23.8M) but
  its AR decoder buys ~+9pt word and native variable-length. Point the loader only at synth + the
  200-eval, never real pseudo-labels. `model 1/src/eval/case_decode.py` for force-len + case-folding.
- **reference** `cryptbb_sourcefree/src/eval_scoreboard.py` scoring methodology (force-len decode, eval
  scored exactly once).

**Do NOT:**
- Train with **augment:false** → val_word collapses to **1.30%** (catastrophic overfit); augment:true
  → **78.26%**. Augmentation is non-negotiable. But keep it **light photometric** — heavy geometric
  aug on already-distorted synth stalled loss at ~3.3 ≈ random.
- Use **PARSeq `refine_iters=1`** — corrupts the first char (5%→90% fixed by `refine_iters=0` + force-len,
  for both decode *and* checkpoint selection).
- Select checkpoints on **real** (eval peeking) — select on synth-val.
- Use a **self-paced/adaptive curriculum** — model 3 peaked ~21.5% synth-val then collapsed, never
  reached a reportable real number. Prefer plain shuffled or a teacher-fixed static curriculum.
- Report char-acc alone: **word ≈ char^length** — a run hit 60.83% char but 1.08% word.

**Baselines to calibrate against (651 labeled CryptBB, `model 1/data/eval`):** OCR/VLM floor ≤2.5%
word; **VLM-free target to beat = 30.4% word CryptBB / 91.0% word DarkBay** (the banned approach);
source-based Model 1 39.94%; flagship 57.60% (uses the scheme font — *not* a fair target).

---

## 3. Data assets (Part A) — what's safe, what's poison

| Asset | Path | Size / geom | Role |
|---|---|---|---|
| CryptBB unlabeled | `CryptBBCaptchas(unsolved)…/` | 600 · 200×60 · ~5 alnum | ✅ **R3 input**; scoreable |
| CryptBB eval | `model 1/data/eval` | 651 (labels=filenames) | ⚠ **EVAL-ONLY** scoreboard |
| DarkBay pool | `cryptbb_sourcefree/data/darkbay/pool` | 799 · 200×50 | ✅ **R3 input** (pixels only) |
| DarkBay eval | `cryptbb_sourcefree/data/darkbay/eval` | 201 · 6-char | ⚠ EVAL-ONLY |
| City live | `City (live)…/` | 2000 · 324×81 · digits | ✅ input; **no eval** (visual only) |
| Courier live | `Courier Market (live)…/` | 2510 · 200×60 · digits | ✅ input; **no eval** (visual only) |
| **DarkBay_Labeled** | `DarkBay_Labeled…/` | 1000 · 200×50 | 🚫 **BLOCKED** — eval-only, ≠ the 799 pool |
| **JoeyRetitling** (both copies) | `JoeyRetitling…/`, `JoeyRetitling_extracted/` | 651 | 🚫 **THE CryptBB ANSWER KEY** (651/651 filename subset of eval, 0 unique). Never an input. |

**Only CryptBB and DarkBay are end-to-end scoreable** (clean pool + labeled eval). City/Courier are
generality/visual only. Manifest: `cryptbb_sourcefree/{DATASETS.md,configs/corpus.yaml}` (has a
`blocked:` section). **Every prebuilt glyph bank on disk is R1-tainted in its label field only** —
the DINO+KMeans crop *clusters* are R2-clean shapes (reuse if you drop `cluster_to_char`); the char
assignments came from easyOCR-on-real (discard, re-derive from your rendered strings). Dedup the
unlabeled pool against the 200-eval by file hash (eval-contamination is a *documented* prior failure —
the refiner trained its discriminator on the eval-contaminated DarkBay set).

**Fonts on disk (a font `.ttf` is typography data, not a captcha library — usable under R5):**
`model 1/data/_mybb_fonts` (MINYN___.ttf + edmunds.ttf = the *actual* MyBB fonts — still off-limits
for **rendering shipped synth** because it's the scheme's own font/answer-key under R1/R3; use only to
**validate** that your Stage-2 font search lands the right shape; +8.76pt evidence: 31.18→39.94% word);
38 matplotlib TTFs; system fonts (C:/Windows/Fonts); **Google Fonts** (the field's recommended
large-library source — not a captcha lib, fully available).

> **R5 font caveat:** `generation/configs/font_pool.json` (69 TTFs) and `model 0/vendor/{gregwar,
> coolcaptcha,kcaptcha,…}` fonts were *bundled inside captcha libraries*. Under the literal reading
> of R5 ("no captcha **libraries**") a standalone `.ttf` copied out is fine; under a strict reading
> (nothing sourced from captcha tooling) they're excluded. **Default here: prefer Google Fonts +
> system + matplotlib faces** (unambiguously clean and a far larger pool than the ~4 fonts prior work
> was stuck on) and treat the captcha-lib-bundled TTFs as optional/gray. Tell me if you want the
> strict reading and I'll drop them from the candidate pool entirely.

---

## 4. Ambiguous-tool rulings (from the constraint audit)

| Tool | Ruling | Why |
|---|---|---|
| **SimCLR / BYOL from scratch** | ✅ R2-clean | The Stage-4 prescription. Build fresh (no impl in repo). |
| **MAE from scratch** | ✅ R2-clean | Self-supervised, no language; impl at `cryptbb_sourcefree/src/mae_pretrain.py`. |
| **scikit-image Sauvola/Niblack** | ✅ R2-clean | Classical adaptive thresholding. Not yet in repo — trivial add. |
| **kornia augmentations** | ✅ R2-clean | No pretrained weights, no language. Keep augs glyph-preserving. |
| **SAM / MobileSAM** (box/point-prompt) | ✅ R2-clean | Vision-only. But marginal on captchas (IoU 0.13–0.36). *GroundingDINO / Grounded-SAM text-prompted variants are **banned** (language-grounded).* |
| **LaMa inpainting** | ✅ R2-clean but ⚠ | Vision-only, but a proven **label-poisoning trap** (ghost letters → wrong labels; keeps only ~124/200 bgs). Prefer procedural bg. |
| **CUT / CycleGAN** | ✅ R2-clean but ⚠ | GAN, no language. But graveyard-net-negative on exotic fonts (39.94→0.92%). 5A first; skip on exotic fonts. |
| **pretrained DINOv2** | 🟡 borderline | R2-clean *by letter* (self-supervised) but **violates "from scratch" spirit**. Diagnostic-only, never in the path. |
| **easyOCR** | 🚫 banned in practice | CTC OCR (R2-clean) but **R1-violating the instant it labels a real captcha** (and hallucinates 91 junk chars, conf 0.477). OK only as synth-side readback QA. |
| **Qwen-VL / CLIP / SigLIP / BLIP / TrOCR / GroundingDINO** | 🚫 R2 banned | Language-supervised. This is the whole point of the plan. |
| **captcha / claptcha / securimage / gregwar / kcaptcha / multicaptcha-gen** | 🚫 R5 banned | Off-the-shelf captcha generators/wrappers — banned as A2 proxy and as Stage-3 wrappers. Hand-roll with Pillow/GD. (Raw PHP-GD `imagettftext` is a graphics primitive, allowed.) |

**Contaminated code to never import** — R2/VLM: `generation/vlm.py`, `director.py` VLM path,
`qa.discriminate`; `auto_synth/src/part_a/vlm_alphabet.py` + `part_c/{vlm_qwen,extract_bgs_vlm,
grounded_dino,extract_bgs_grounded}.py`; `captcha regenerator/masking/src/masking/stage0.py`;
`cryptbb_sourcefree/src/{qwen_finetune,qwen_selftrain,trocr_finetune,trocr_selftrain}.py`.
R1 (pseudo-labels real): `auto_synth/.../{map_clusters,ocr_discover}.py`, `generation/build_bank.py`
(easyOCR), `cryptbb_sourcefree/src/{self_train,qwen_selftrain}.py`, `synth_v1/build_bank.py`
(detector_v2 + parseq_v2), all `cluster_to_char` manifest fields.

---

## 5. Recommended build order under R1+R2

1. **Fresh venv** (§1), classical stack + torch/timm + `lightly`/`cma`. No easyocr/transformers.
2. **M1 data:** hand-roll an A2 proxy scheme first (Pillow/GD + Google/system fonts — **no captcha
   library**, R5). Stage easy real (DarkBay 799) + hard real (CryptBB 600) as the generality +
   worst-case tests. Hash out the 200-eval.
3. **M2 Stages 1+2:** port `extract.py` + `director` classical priors + Sauvola/Niblack + sine-fit →
   `scheme_profile.json`. Eyeball vs real.
4. **M3 Stage 3:** wire `render_source`+`distort`+`background`+`SourceDataset`. **One-pass render, no
   alpha seam, fixed length, native size.** M3 eye-test checks *pipeline* match (wave/noise/JPEG), not
   beauty.
5. **M4 Stage 6 baseline:** TinySTR (or CRNN) on 50–100k synth, light aug, score 200-eval. Expect
   ~90%+ on proxy/easy, single-digit on CryptBB.
6. **M5 Stage 4+5A:** from-scratch SSL critic (MAE or SimCLR) → CMA-ES + fresh MMD/CORAL over its
   features. Gate on the eval number, not t-SNE.
7. **M6 Stage 5B:** only if 5A plateaus *and* the gap is appearance not structure. Honor the guard,
   but expect it to fail on exotic fonts — the sweep says don't build around it.

**Bottom line:** ~90% of the classical machinery this plan needs already exists, VLM-free, scattered
across the repo (extract.py, distort.py, background.py, render_source.py, mae_pretrain.py, tiny STR).
The VLM was the discardable ~10% control loop. What you *cannot* recover by removing the VLM is the
**font wall on exotic schemes** — so scope the thesis honestly around proxy + easy schemes, with
CryptBB reported as the measured ceiling.
