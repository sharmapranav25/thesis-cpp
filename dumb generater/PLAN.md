# Dumb Generator — Synthetic Labeled CAPTCHA Generator (no labels, no VLM/LLM)

Clean-slate design. Independent of `generation/`, `auto_synth/`, `cryptbb_sourcefree/`.

## Constraints (hard)

1. **No labeled data** on real captchas. Ever. Not for training, not for pseudo-labeling.
2. **No LLMs or VLMs** anywhere in the pipeline. No Qwen-VL, no CLIP, no BLIP, no GPT.
3. Input: a folder of unlabeled real captchas from one target scheme.
4. Output: a labeled synthetic dataset that mimics the scheme well enough to train a solver whose accuracy on real is meaningfully non-trivial.
5. **No captcha-specific libraries.** No off-the-shelf captcha generator or wrapper anywhere in the pipeline — not `captcha`/`claptcha` (PyPI), not `securimage`/`gregwar/captcha`/`kcaptcha`/`coolcaptcha`/`simplecaptcha` (PHP), not `multicaptcha-gen` (Node), and not any vendored copy of these in this repo. The generator is built only from **general-purpose graphics/CV primitives** — Pillow, OpenCV, NumPy, and raw PHP-GD `imagettftext` (GD is a graphics library, not a captcha library). This also governs Part A: a self-made proxy scheme must be **hand-rolled**, not produced by a captcha library. (Font `.ttf` files are typography data, not captcha libraries, and remain usable.)

The only "labels" that ever exist are (a) the ground-truth strings the generator itself decided to render (free by construction), and (b) an optional ~200-image hand-labeled eval set used **only** for scoring.

---

## Part A — Where to get the unlabeled captchas

Three options, ranked by effort/risk.

### A1. Public captcha image datasets (zero effort, generic)
- **HuggingFace Datasets** — search "captcha": `hammer888/captcha-data`, `project-sloth/captcha-images`, `parsawar/captcha_dataset_v3`. Ignore their labels.
- **Kaggle** — `fournierp/captcha-version-2-images`, `akashguna/large-captcha-dataset`.
- **GitHub** — `AakashKumarNain/CAPTCHA-Cracker`, `RustingSword/captcha_break`, various forks. Grab their `images/` folders.

### A2. Hand-roll a proxy scheme yourself (best for R&D)
Write your **own** small renderer (Pillow/OpenCV, or raw PHP-GD `imagettftext`) to produce a "target scheme" that you *pretend* is unknown. Per constraint 5 you may **not** use an off-the-shelf captcha library for this. Pick the hidden ground truth yourself: a font (system / Google Fonts), a background style, a distortion (sine warp + noise), a length, a charset. Because you control it, you can measure the pipeline objectively without touching a labeled real dataset.

Render ~2000 images with fixed seed(s), discard the labels, treat as unlabeled. Then generate an eval set with fresh seeds — labels are correct because you rendered them. Keep the proxy's parameters in a sealed file you don't look at until scoring, so Stage 2 genuinely has to rediscover them.

### A3. Scrape a live site (highest fidelity, most risk)
Only with permission or on your own property. Selenium/Playwright: load page, screenshot the captcha element, save. Rate-limit + rotate. Skip until pipeline works.

**Recommendation:** start A2 (a hand-rolled Pillow/GD proxy scheme). Once M1-M5 work end-to-end, prove generality by swapping in an A1 dataset.

---

## Part B — The six stages

Each stage is one increment. Each produces something you can look at with your eyes before moving on.

### Stage 1 — Ingest & normalize

**Purpose:** consistent tensor shape + a dataset-level fingerprint for downstream.

**Input:** folder of PNGs/JPGs, variable size.
**Output:** normalized RGB tensors (canonical H×W, e.g. 64×192), `stats.json`.

**Steps:**
- Detect dominant aspect ratio; resize to canonical shape (letterbox or crop).
- Per-image: mean, std, 8-color k-means palette, Canny edge density, FFT log-magnitude spectrum, Otsu foreground fraction.
- Aggregate to dataset-level distributions in `stats.json`.

**Tools:** OpenCV, NumPy, scikit-learn.

**Done when:** `stats.json` prints sensible numbers; a random resized tensor round-trips to a viewable PNG.

---

### Stage 2 — Unsupervised style profiling (classical CV, no learning)

**Purpose:** reverse-engineer the parametric fingerprint of the scheme.

**Input:** normalized folder from Stage 1.
**Output:** `scheme_profile.json`.

**What to extract per image, then aggregate:**

| Property | Method |
|---|---|
| Background color / gradient | Border-pixel k-means (k=2). Test flat vs gradient (plane fit) vs textured (variance). |
| Foreground/background mask | Adaptive threshold: **Sauvola** (uneven lighting) or **Niblack** (noisy). |
| # of glyphs per image | Connected components + morphological cleanup; drop tiny CCs; split fat CCs by vertical-projection valleys. Aggregate → length distribution. |
| Glyph height/width | Bounding boxes of surviving CCs. |
| Stroke thickness | Distance transform on mask; median ridge value. |
| Character overlap | (sum of CC widths − image width) / image width. Positive = overlapping. |
| Rotation range | PCA on each CC's pixel coordinates → principal angle. |
| Sinusoidal warp | Least-squares fit y = A·sin(ωx + φ) to CC centroids. |
| Line/curve clutter | Remove CCs; run Hough transform on residual; count + measure lines. |
| Noise type | High-freq FFT energy → salt-and-pepper; low-freq → gradient noise; near-zero → clean. |
| Font family guess | For each CC crop: render ~200 candidate fonts (system fonts + Google Fonts) at matching height for the top 10 common English letters; score by IoU / shape context. Cluster CCs → per-position glyph shape prototypes → match. No labels needed (clustering shapes, not naming them). |

**Failure modes:**
- Heavy overlap breaks CC segmentation → fall back to vertical-projection cut with expected-length prior from the aggregate.
- Multi-block dark backgrounds break Otsu → HSV-based masking with color-cluster majority.
- Colored glyphs on colored bg → cluster in Lab-space; foreground = whichever cluster has smaller area.

**Tools:** OpenCV, `scipy.optimize`, Pillow (font rendering).

**Done when:** `scheme_profile.json` numbers pass eyeball sanity — colors match the scheme, average glyph count matches, font family plausible.

---

### Stage 3 — Procedural labeled generator

**Purpose:** the actual labeled-data factory. Labels correct by construction.

**Input:** `scheme_profile.json` + target alphabet (e.g. `[a-zA-Z0-9]` or scheme-specific).
**Output:** unlimited pairs `(image.png, label.txt)`.

**Pipeline:**
1. Sample a random string. Length ← distribution from Stage 2. Chars ← alphabet.
2. Sample render params from profile distributions: font (from top-k candidates), color, size, per-char rotation range, global sine warp (A, ω), background choice.
3. Render each char to its own alpha layer with Pillow (`ImageDraw.text` + TTF).
4. Compose glyph strip: apply per-char rotation, then global sine warp.
5. Composite over sampled background (flat / gradient / noise / texture per profile).
6. Add distractor lines/dots per profile.
7. Optional JPEG re-encode + noise to match the scheme's compression artifacts.
8. Save `(image, label)`.

**Tools:** Pillow, NumPy, OpenCV — all hand-rolled, **no captcha-library wrappers** (constraint 5). Raw PHP-GD `imagettftext` is an allowed alternate renderer backend.

**Failure modes:**
- Renderer param distributions off → solver in Stage 6 undertrains. That's Stage 5's job to fix.
- Font pool doesn't contain a close-enough match → Stage 3 output looks "clearly different" at eye test. Add more fonts (Google Fonts, DaFont scrape) or accept that Stage 5 will close the gap.

**Done when:** 100 rendered samples look "obviously the same scheme" at a glance next to 100 real samples.

---

### Stage 4 — Self-supervised critic (small, from scratch)

**Purpose:** build a perceptual feature extractor that captures scheme "look" — needed by Stage 5.

**Input:** unlabeled real folder + (optionally) synth from Stage 3.
**Output:** a feature extractor `f(img) → R^d`, checkpoint file.

**What to train:**
- Small backbone: **ResNet-18** or **ViT-tiny** (~5M params).
- Objective: **SimCLR** or **BYOL** on the unlabeled real folder. No labels, no VLM. Trains in ~30 minutes on an A4000.
- Augmentations: crop, color jitter (mild — don't destroy the scheme's color signature), Gaussian blur, JPEG compression. Skip augmentations that change glyph structure (heavy rotation, cutout on glyphs).

**Tools:** PyTorch + `lightly` or `solo-learn` (both pip installs, both self-contained, no VLM).

**Failure modes:**
- Overfits to trivial features (JPEG level, mean color) → domain gap won't close in Stage 5. Mitigation: stronger color augmentation during pretraining so structure dominates.
- Too small a real pool (< 500 images) → representation is noisy. Mitigation: include synth in the SSL pool with a small weight; SSL is label-free so this is fair.

**Done when:** t-SNE of Stage-4 features on real images shows a compact cluster; features change smoothly with distortion parameter sweeps on synth.

---

### Stage 5 — Unsupervised domain refinement

**Purpose:** close the residual gap between procedural synth and real, without losing labels.

**Input:** synth (Stage 3) + real (unlabeled) + critic (Stage 4).
**Output:** either a tuned generator (5A) or a refiner network (5B). Both preserve labels.

#### Option 5A — Feature-matching parameter tuner (default)
- Treat Stage 3's params (color hex, wave amplitude, noise density, font-mix weights…) as knobs.
- Objective: minimize **MMD** or **CORAL** between Stage-4 features of `synth_batch` and `real_batch`. Both batches are unlabeled — this is legitimate under R1.
- Optimizer: **CMA-ES** for the discrete/categorical knobs (font choice, background type); Adam for the continuous ones where you have autograd hooks (color, warp amplitude).
- Labels stay 100% intact because you never touch the output pixels — you tune the generator, then re-generate the whole synth pool.

**Tools:** PyTorch, `cma` (pip), custom MMD/CORAL loss.

#### Option 5B — CUT refiner network (add only if 5A plateaus)
- Train **CUT** (Contrastive Unpaired Translation, Park et al.) — synth → refined-synth, with a discriminator seeing real.
- Content-preservation guardrail: edge-map IoU or Stage-4-feature cosine between input and output must stay ≥ threshold. Without this, the refiner will happily change glyphs, destroying labels.
- Sanity every N steps: feed synth → refiner → CRNN-with-known-label; if char accuracy drops > 2%, raise the content-preservation weight.

**Tools:** PyTorch, CUT reference impl (BSD, self-contained).

**Recommendation:** ship 5A first. Only add 5B if Stage 6 accuracy plateaus and the residual gap is clearly *appearance* (compression, blend), not *structure*.

**Failure modes (5B):**
- Labels drift silently. Mitigation: the CRNN sanity check above, plus a hold-out synth-only accuracy test after each refiner epoch.

**Done when:** t-SNE of Stage-4 features shows real and refined-synth overlapping; unrefined-synth is a separate cluster.

---

### Stage 6 — Solver training + evaluation

**Purpose:** the objective measurement.

**Input:** refined synth from Stage 3 (+ Stage 5) with labels.
**Output:** trained CRNN/PARSeq + accuracy number.

- Train **CRNN** (baseline) or **PARSeq** (stronger) on ~50-100k refined synth.
- Standard augmentations at training time: mild affine, color jitter, JPEG compression.
- Evaluate on:
  - **~200 hand-labeled real** (label these yourself, once, only for eval — never for training).
  - Optionally, the labeled versions of A1 datasets as an independent generalization test.

Report word accuracy + character accuracy. Feed the confusion matrix back to Stage 2/3 (e.g. many 6/8 confusions → increase glyph diversity for those chars, tune stroke width).

**Tools:** PyTorch, an open-source PARSeq or CRNN impl.

**Done when:** char accuracy on the 200-image hand-labeled eval clears whatever bar you set (30% word / 80% char is a common target for hard schemes; 90%+ for easy ones).

---

## Part C — Build order (six milestones)

Each milestone is one PR-sized increment.

| # | Milestone | Definition of done | Time est. |
|---|-----------|--------------------|-----------|
| **M1** | Data collection (Part A) | ~2000 images from A2 (a hand-rolled proxy scheme). HF captcha set staged as independent test folder. | ~1 day |
| **M2** | Stages 1+2 → `scheme_profile.json` | Print the profile; eyeball vs real folder. Colors, length dist, font family all plausible. | 2-3 days |
| **M3** | Stage 3 procedural gen | 100 rendered samples look "obviously the same scheme" as real. | 2-3 days |
| **M4** | Stage 6 baseline solver on plain Stage-3 synth | CRNN trained on 50k synth; eval on 200 hand-labeled real. Expect 30-60% char acc depending on scheme. | 1-2 days |
| **M5** | Stage 4 SSL critic + Stage 5A feature-matching tuning | Retune Stage-3 params via MMD/CORAL. Re-run M4. Expect +10-20 pts char acc. | 3-4 days |
| **M6** | Stage 5B CUT refiner (only if M5 plateaus) | Unpaired refiner with content-preservation. Re-run M4. | ~1 week |

**Realistic timeline:** ~3 weeks for a working end-to-end system on an easy scheme; +1-2 weeks for a hard one.

---

## Part D — How to know it's working without full labels

The no-label constraint means you can't compute training-time accuracy on real. Three eye-test loops replace it:

1. **Human eye test at M3.** Flip through 100 synth vs 100 real side-by-side. If you can trivially tell them apart in a specific way (e.g. "real has JPEG noise, synth doesn't"), that's a Stage 3 knob to fix.
2. **t-SNE at M5.** Plot Stage-4 features of real, unrefined synth, and refined synth in 2D. Refined-synth should overlap real; unrefined shouldn't.
3. **Hand-labeled eval at M4/M5.** 200 real captchas labeled by you, once, only for scoring. Standard practice — nothing in the training path saw a label on real data, so R1 is not violated.

---

## Part E — Directory layout (proposed)

```
dumb generater/
├── PLAN.md                      (this file)
├── requirements.txt
├── data/
│   ├── real_unlabeled/          (Part A input pool)
│   ├── real_eval_labeled/       (200 hand-labeled, eval only)
│   ├── synth_v1/                (M3 output)
│   └── synth_refined/           (M5 output)
├── configs/
│   └── scheme_profile.json      (M2 output)
├── src/
│   ├── stage1_ingest.py
│   ├── stage2_profile.py
│   ├── stage3_generate.py
│   ├── stage4_ssl.py            (SimCLR/BYOL trainer)
│   ├── stage5a_tune.py          (CMA-ES + MMD)
│   ├── stage5b_cut.py           (optional, CUT refiner)
│   └── stage6_solver.py         (CRNN/PARSeq train + eval)
├── fonts/                       (system fonts + Google Fonts + DaFont scrapes)
└── ckpts/
    ├── ssl_critic.pt
    ├── refiner.pt (optional)
    └── solver.pt
```

---

## Part F — What to explicitly not do

- **Do not use a VLM anywhere.** Not for pseudo-labeling, not as a critic, not for font ID. That's the whole point.
- **Do not use pretrained multimodal encoders** (CLIP, SigLIP, BLIP). These are trained with language supervision — they count as VLMs.
- **Do not label real captchas for training.** Only the 200-image eval set gets human labels.
- **Do not skip the human eye test at M3.** If synth doesn't look right to you, no amount of Stage 5 will save it.
- **Do not train Stage 5B without the CRNN sanity check.** CUT/CycleGAN destroys labels silently.
- **Do not overfit the SSL critic to trivial features.** Use strong color augmentation during pretraining.
- **Do not use any off-the-shelf captcha library** (constraint 5) — not as an A2 proxy generator, not wrapped in Stage 3, not vendored from elsewhere in the repo. Build the renderer from general graphics primitives so the "source-free" claim is real.

---

## Part G — Open questions to resolve before starting

1. **Which target scheme first?** Pick either (a) a hand-rolled proxy scheme for fast iteration, or (b) a real scheme you already have unlabeled images for.
2. **Alphabet?** `[a-z0-9]` / `[A-Z0-9]` / `[a-zA-Z0-9]` / scheme-specific. Depends on what Stage 2's CC segmentation surfaces.
3. **How many hand-labeled eval images?** 200 is the default; go higher (500) if variance is unacceptable.
4. **Backbone for Stage 4:** ResNet-18 (default) or ViT-tiny (better representations, slower). Start ResNet-18.
5. **Backbone for Stage 6:** CRNN (default, fast, weak) or PARSeq (slower, stronger). Start CRNN, upgrade to PARSeq at M6.
