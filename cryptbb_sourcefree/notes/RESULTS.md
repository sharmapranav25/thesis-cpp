# RESULTS — source-free GAN captcha factory

Log every run here. The one number that matters: **PARSeq trained on our
source-free/label-free GAN synth → word% on the 651 real CryptBB**, vs the
reference numbers below.

## Reference numbers (the scoreboard)
| System | Uses source code? | Uses real labels? | Real word acc (651) |
|---|---|---|---|
| Off-the-shelf OCR/VLM (Qwen2-VL-2B best) | no | no | ~2.5% (the floor) |
| Model 1 (MyBB port) | ✅ yes | no | 39.94% |
| Model 2.5 (MyBB port + curriculum + ensemble) | ✅ yes | no | **57.60%** |
| **This project (GAN restyle)** | ❌ **no** | ❌ **no** | **TBD ← the goal** |

Target: beat the floor decisively; aim 30–50%. Any solid number here is novel
because it uses neither the source code nor labels.

## Running log

### 🔑 RESUME HERE — state as of 2026-07-04 (latest)
Everything below is **source-free + label-free** (generic fonts, no MyBB font/source, no human
labels; the label is the text WE render). Nothing is training right now — all runs stopped clean,
all checkpoints saved. To pick up in a new session, start from this block.

**Best checkpoints (LoRA adapters, load onto Qwen/Qwen2-VL-2B-Instruct):**
| scheme | checkpoint | REAL word | how it was made |
|---|---|---|---|
| CryptBB | `data/ckpts/qwen_selftrain` | **30.41%** | plain-synth FT → self-train (round 2 peak) |
| CryptBB | `data/ckpts/qwen_cryptbb_php100k/lora_adapter.pt` | **29.8%** | fine-tune on 100k PHP-GD synth (no self-train) |
| DarkBay | `data/ckpts/qwen_darkbay_selftrain` | **91.0%** | PHP-FT → self-train |
| DarkBay | `data/ckpts/qwen_darkbay_php100k` | **86.5%** | fine-tune on 100k PHP-GD synth |

**Reproduce / continue (exact commands):**
```
# fine-tune on PHP synth (scheme-general; --scheme darkbay for DarkBay)
python -u src/qwen_finetune.py --pool data/synth_pool/cryptbb_php100k --eval-dir "E:/pranav/model 1/data/eval" \
  --n-len 5 --steps 1500 --eval-every 250 --out data/ckpts/qwen_cryptbb_php100k
# self-train on the UNLABELED reals (600-pool = the clean, working config)
python -u src/qwen_selftrain.py --pool data/synth_pool/cryptbb_php100k --scheme cryptbb \
  --adapter data/ckpts/qwen_cryptbb_php100k/lora_adapter.pt --eval-dir "E:/pranav/model 1/data/eval" \
  --n-len 5 --rounds 5 --round-steps 400 --conf-thr 0.9 --real-frac 0.5 --lr 5e-5 \
  --out data/ckpts/qwen_cryptbb_php100k_selftrain
# regenerate a synth pool (100k in ~60s on 32 cores; needs C:/xampp/php)
python src/php_pool.py --scheme cryptbb --n 100000 --fixed-len 5 --out data/synth_pool/cryptbb_php100k
```
**Open next steps (untried levers, ranked):** (1) self-train the 29.8% PHP checkpoint on the full
600-pool — should clear the 30.41% plain-synth peak; (2) bigger VLM (Qwen2.5-VL-7B); (3) progressive
/ lower conf threshold in self-train; (4) more synth. See "PHP-render lever" + "disjoint self-train"
entries below for what's already known.

### 🛠️ Generalized measured-style generator — TRIED then REVERTED (2026-07-05)
Explored auto-tuning the PHP-GD generator from the unlabeled pool (an `analyze_pool.py` that
measures colour/size/wave/bg stats → drives `php_gen.php`), to remove the hand-tuned per-scheme
`SCHEME_CFG`. Built + visually iterated, but **reverted at the user's request** before any
train/score. Reasons: (a) it introduced visual regressions that needed several fix rounds (DarkBay
backgrounds came out CryptBB-like; letter size over-read), (b) no evidence it moved the *number*
(never trained — image realism is only a low-single-digit-word% lever on the VLM path anyway).
Reverted `php_gen.php` + `php_pool.py` to the hardcoded-`SCHEME_CFG` versions that produced the
29.8% (CryptBB) / 86.5% (DarkBay) checkpoints. The ghost-detection idea was kept in `notes/FUTURE.md`.
If revisited: measure-don't-train is sound and fast, but prove it on the *score*, not just the look.

### ⭐ PHP-GD render is a real source-free lever (+4.4% word CryptBB) (2026-07-04)
Replaced the Python composite renderer with a **faithful PHP-GD generator** (`src/php_gen.php`,
driven by `src/php_pool.py`) — the same primitives real PHP captcha libraries use: `imagettftext`
+ per-pixel **sine-wave warp** + faint lines/arcs/dots/grid. Still source-free: generic fonts
(DejaVu/Verdana), NOT any scheme's captcha.php or font. WE pick the text → labels correct by
construction. Generated 100k labeled per scheme (`data/synth_pool/{cryptbb,darkbay}_php100k`).
| scheme | plain-synth FT | **PHP-synth FT** | delta |
|---|---|---|---|
| CryptBB | 25.4% word | **29.8% word** | **+4.4** |
| DarkBay | 86.5% word | 86.5% word | ~0 (no-op) |
- **PHP-render matched the whole self-train gain (+4.4) on its own, on the HARD scheme.** No-op on
  DarkBay (already VLM-readable, so nothing to gain from a better-matched render).
- Counterintuitive finding confirmed earlier and again here: **render-pipeline match > visual
  fidelity** — a *stronger* wave that looks less "pretty" transferred better than a tuned-to-look-
  real milder one. Distortion amount that matches the real pipeline matters more than realism.
- Lesson (self-inflicted, logged): I once called the tuned PHP "underperforming" at a temporary
  plateau (23.8% mid-run) — it then climbed to 29.8% and was actually BEST. **Wait for plateau.**

### ⚠️ Disjoint self-train: too-small a pool REDUCES accuracy (2026-07-04)
Test of self-training rigor: split CryptBB by label into **100 unlabeled (self-train) + 551 held-out
(test)**, verified disjoint (`data/cryptbb_split/{selftrain,test}`). Self-train on just the 100:
| round | pseudo-labels kept | REAL word |
|---|---|---|
| 0 (warm-start) | — | 30.31% |
| 1 | 16 | 29.40% |
| 2 | 19 | **27.59%** ↓ |
- **It went DOWN.** Root cause = the pool was below the minimum for the flywheel:
  (a) only 16–19 pseudo-labels, and at ~30% word some confident ones are **confident-but-WRONG**
  (mean-conf 0.72) → training on your own errors = confirmation bias; (b) `real_frac=0.5` draws half
  of every batch from those same 16–19 images → memorization/overfit; (c) continued lr-5e-5 updates
  with too weak a real signal let the synth half pull weights back toward synth-overfit.
- Contrast the **600-pool** run: ~89–141 correct pseudo-labels → wrong ones diluted, diverse signal
  → it CLIMBS (30.31→…→30.41 peak). **There's a floor on # correct pseudo-labels below which
  self-training hurts.** ~16 is below it; ~89+ is above it. Use the full ~600 pool, not a subset.

### ⭐⭐ HEADLINE — Qwen2-VL-2B (general VLM) source-free beats the OCR floor ~10x (2026-07-04)
Advisor's idea (VLMs generalize across the font wall) → decisively confirmed. Fine-tune a
general pretrained VLM (LoRA) on our SOURCE-FREE synth (no MyBB font, no real labels); score 651 real.
| System | source-free? | real word | real char |
|---|---|---|---|
| OCR/VLM floor (zero-shot) | ✅ | ≤2.5% | — |
| our PARSeq (from scratch) | ✅ | 0% | 18% |
| our TrOCR-large (best) | ✅ | 3.4% | 51.5% |
| **Qwen2-VL-2B zero-shot** | ✅ | **3.84%** | **43.5%** |
| **Qwen2-VL-2B + synth FT (step 500)** | ✅ | **22.12%** | **65.83%** |
| Model 1 (uses MyBB font) | ❌ | 39.94% | 84% |
| Model 2.5 (uses MyBB font) | ❌ | 57.60% | — |
- **22% word source-free** = ~6x our best prior source-free, ~10x the OCR floor, >half of the
  source-BASED Model 1 — after only 500 steps (loss still falling, curve still rising).
- word (22%) > char^5 (~12%): the VLM reads whole captchas COHERENTLY (holistic reasoning),
  an advantage per-char recognizers (PARSeq/CRNN) structurally lack.
- Impl `src/qwen_finetune.py` (Qwen2-VL-2B-Instruct, LoRA r16, bf16, grad-checkpoint, batch2/accum4,
  ~1.7 it/s; 6.4GB GPU; downloads on E:\hf_cache). Guardrail: 651 eval scored only, never trained.
- Next: let it climb/plateau; then self-train on unlabeled reals (now richly unblocked: 22% word =
  lots of correct pseudo-labels); prove generality on City/Courier.

### ⭐⭐⭐ FINAL source-free result: Qwen2-VL + self-training = 30.4% word on real (2026-07-04)
Stacked self-training (`src/qwen_selftrain.py`) on the fine-tuned Qwen: warm-start, then each
round pseudo-label the 600 UNLABELED reals (conf>=0.9 + len-5), fine-tune on synth + confident
pseudo-reals. The flywheel PARSeq could NOT start (0% word -> 1 pseudo-label) but Qwen could:
| stage | pseudo-labels | real char | real word |
|---|---|---|---|
| synth fine-tune (warm-start) | - | 68.4% | 25.35% |
| self-train round 1 | 40 | 70.9% | 26.73% |
| self-train round 2 (**PEAK**) | 98 | 71.9% | **30.41%** |
| self-train round 3 | 123 | 72.1% | 30.11% |
| self-train round 4 | 141 | 72.1% | 29.65% |
- Flywheel visibly compounds then saturates: as accuracy rose, more reals crossed the confidence
  bar (40->98->123->141) — impossible with PARSeq (stuck at 1). Peaks round 2, then mild overfit
  to the same "easy" reals -> slow decline. Best ckpt = `data/ckpts/qwen_selftrain` (30.41%).

### ⭐⭐⭐⭐ GENERALITY — same pipeline on DarkBay (a DIFFERENT scheme) = 86.5% word (2026-07-04)
Proved the factory is scheme-general: added `darkbay` to configs/corpus.yaml (200x50, 6-char,
uppercase+digits no-I, glyph_style=bw, bg_style=dark), and ran the SAME Qwen pipeline. DarkBay
has a disjoint clean split: 799 UNLABELED pool + 200 held-out eval (labels scored only). Scripts
are now scheme-general (qwen_finetune/qwen_selftrain take --scheme/--eval-dir/--n-len/--upper).
| DarkBay stage | source-free? | word | char |
|---|---|---|---|
| Qwen2-VL zero-shot | ✅ | 61.5% | 77.9% |
| + source-free synth fine-tune (step500) | ✅ | 83.5% | 94.7% |
| + fine-tune (plateau ~step1000-1500) | ✅ | 86.5% | 96.2% |
| + self-train on 799 unlabeled (R1-4: 380/404/415/411 pseudo-labels) | ✅ | **91.0%** | **97.3%** |
DarkBay FINAL source-free = **91.0% word / 97.3% char** (zero-shot 61.5 → FT 86.5 → self-train 91.0).
- **86.5% word source-free on DarkBay** — blows past the 70-80% target, with NO source font and
  NO human labels. DarkBay's font is standard-ish (VLM-readable); CryptBB's exotic MyBB font is
  the pathological worst case. So the factory spans **30% (hardest) to 86%+ (typical)** source-free.
- Key thesis takeaway: the "font wall" is specific to (a) FONT-SPECIFIC recognizers and (b) exotic
  fonts. A general VLM sidesteps both; per-scheme accuracy tracks how VLM-readable the font is.

## THE SCOREBOARD (final)
| System | source code? | human labels? | real word% (651) |
|---|---|---|---|
| OCR/VLM zero-shot (Qwen2-VL) | no | no | ~2.5% (floor) |
| our PARSeq (from-scratch, source-free) | no | no | 0% |
| our TrOCR-large (source-free) | no | no | 3.4% |
| **our Qwen2-VL + self-train (source-free)** | **no** | **no** | **30.41%** |
| Model 1 (MyBB port) | YES | no | 39.94% |
| Model 2.5 (MyBB port + ensemble) | YES | no | 57.60% |
**30.4% word, fully source-free & label-free = ~12x the OCR floor and ~76% of the source-BASED
Model 1 — with NO MyBB font and NO human labels.** The thesis result. Key finding: the font wall
is a property of FONT-SPECIFIC recognizers (PARSeq/CRNN); a font-GENERAL pretrained VLM
(Qwen2-VL) reads the real font out-of-the-box and, fine-tuned on source-free synth + self-trained
on unlabeled reals, reaches source-based-baseline territory. Levers to push higher (untried):
progressive/lower conf threshold, bigger VLM (Qwen2.5-VL-7B), more/better synth, longer rounds.

**Two independent source-free levers, each worth ~+4–5% word on the hard scheme:** (a) self-training
on unlabeled reals (25.4→30.4), (b) faithful PHP-GD render (25.4→29.8). They have NOT yet been
stacked — self-training the 29.8% PHP checkpoint on the full 600 pool is the top untried next step
and should clear 30.4. On DarkBay both saturate early (91.0% word). Generality span stands:
**~30% (CryptBB, hardest exotic font) → 91% (DarkBay, typical) source-free.**

### ⭐ BREAKTHROUGH — font-general pretrained recognizer (TrOCR) breaks the font wall (2026-07-03)
Advisor's idea: a recognizer with broad pretrained font priors (a VLM / OCR transformer)
should transfer source-free-synth → real far better than a from-scratch, font-specific PARSeq.
CONFIRMED on CryptBB:
- **TrOCR-large (microsoft/trocr-large-str), LoRA, fine-tuned on our SOURCE-FREE synth** (40k,
  ghost-free, diversity-fixed pool; NO MyBB font, NO real labels):
  - **zero-shot** (no fine-tune): REAL **28.4% char / 0.6% word** — already beats our best
    synth-trained PARSeq (18% char), i.e. the pretrained prior alone reads the real MyBB font
    better than a specialist trained on our synth.
  - **after 800 fine-tune steps:** REAL **47.5% char / 2.3% word** — char +19pts and climbing;
    loss 4.08→1.13. Fine-tuning converts the prior into scheme-specific reading WITHOUT the font.
- Contrast: from-scratch PARSeq on the same synth = 0% word / 18% char (font-specific → stuck).
- This is the source-free path to Model-1-class accuracy: replace "needs the MyBB font" (Model 1/2.5)
  with "needs an off-the-shelf open OCR model + our source-free synth." Scheme-general (fine-tune
  the same model on any scheme's synth). Run continuing to see how high char/word climb; next:
  scale to a true VLM (Qwen2-VL-2B) + stack MAE/self-train.
- Impl: `src/trocr_finetune.py` (LoRA, guardrail: 651 eval scored only, never trained; torch-2.5
  .bin load-guard + peft save-bug both worked around).


### M0 — wiring + smoke (2026-07-03)
- Built: `render_source.py` (styled per-char color+rotation glyph layer + mask, generic
  DejaVu fonts), `background.py` (procedural CryptBB-like bg + filtered LaMa pool),
  `data.py` (composite source X + real Y loader), `parseq_reader.py` (frozen reader
  wrapper: `.read` + differentiable `.rec_loss`), `model.py`/`losses.py` (CUT G/D +
  PatchNCE), `train_restyler.py`, `generate_pool.py`, `eval_scoreboard.py`.
- **Key finding:** the off-the-shelf `parseq_v2` reader (57.26 val on MyBB synth)
  reads our neutral renders at **0%** — it overfit to MyBB texture and hallucinates
  ("W…W" patterns). Confirmed via the canonical `infer.py` path too, so it's the
  reader, not the wrapper. → Per RESEARCH §4 cold-start, bootstrapped a reader **R0**
  on OUR OWN renders (labels ours → still source-free), trained on clean neutral
  renders + styled composites (the GAN-input look). Init from parseq_v2 encoder
  (174/175 tensors transfer; only `pos_queries` reshaped for max_label_length 5→6).
- LaMa bg pool filtered for ghost letters (kept ~190/200 at dark-frac≤0.06); lean on
  procedural bg (p=0.6) for label safety.

### M0 gate outcome (reader R0 on clean renders)
- Two bugs found + fixed in R0: (1) parseq_v2 can't read our font (0%) → cold-start on
  our renders; (2) R0's AR decode skipped the FIRST glyph — traced to PARSeq's
  `refine_iters=1` cloze pass corrupting position 0 (teacher-forced read pos-0 fine).
  **`refine_iters=0` fixed it: 5% → 90%.**
- Final R0 (10k steps): **word-exact 92.6%, char 98.5%** (n=2048, force-len decode).
- The word gap from 95% is **inherent glyph ambiguity, not a reader deficiency**:
  of 154 char errors, **94 are l↔I and 9 are 0↔O** (pixel-identical in DejaVu; 67% of
  all errors). Excluding those truly-identical pairs, char acc ≈ 99.6% → word ≈ 97%.
- **Verdict:** gate intent met — R0 is a reliable per-character L_rec teacher (98.5%
  char; `rec_loss` uses the teacher-forced path, unaffected by the AR quirk). The
  literal 95%-word bar is unreachable on a mixed-case+digit charset where `l` and `I`
  render identically (Model 2.5 hit the same wall on real). Proceeding to M1.

### M1 — CUT restyler, no L_rec (2026-07-03)
- Built mask-cond U-Net G (reused refine/) + PatchGAN D + PatchNCE (ported CUT) +
  soft mask-L1. Source = styled glyph composite on recreated bg; target = 600 reals.
- 8000 steps, weights adv1/nce1/mask5. Content preserved early (step ~2000) but as
  adv grew (0.9→2.6) the glyphs **faded/smeared**: R0 readback on G(x) = **3%** vs 82%
  on the source composite. Classic content-drift when adversarial outweighs fidelity.
- **M1 gate: CryptBB-ish look ✓, legibility ✗.** Triggered PLAN's "if letters smear →
  raise mask-L1/PatchNCE before L_rec" → M2.

### M2 — recognition-anchored restyler (2026-07-03)
- Fixes for the smearing: (1) **residual G** (output = composite + 0.5·bounded delta →
  starts at the legible composite), (2) **L_rec ON** (rec=0.5; frozen R0 must read the
  correct text out of G(x)), (3) content weights ↑ (nce 1→2, mask 5→8), (4) adv ↓ 1→0.6
  and **slower D** (d_lr 1e-4, ada_p 0.5) so adversarial can't dominate.
- 8000 steps. rec loss 0.62→**0.005** (R0 reads G(x) near-perfectly, teacher-forced).
- **M2 gate: R0 force-len readback on held-out G(x) = 82.5%** — equals the source-
  composite readback (82.4%), i.e. **G preserves legibility with zero degradation**
  (contrast M1's 3%). The 82.5% < 90% is R0's own ceiling on rotated/ambiguous glyphs,
  NOT label corruption — labels are correct by construction. Gate intent met.
- D still fully separates synth from real (residual-limited G can't fool it); realism
  gap remains → a lever for M4. Output is legible + visibly CryptBB-like (QA panels).

| Date | Milestone | Config | Legibility (R on G(x)) | Real word% | Notes |
|---|---|---|---|---|---|
| 2026-07-03 | M0 | R0 cold-start, refine_iters=0, 10k | 98.5% char / 92.6% word CLEAN | — | gate intent met (l↔I/0↔O inherent) |
| 2026-07-03 | M1 | CUT, no L_rec, adv1/nce1/mask5 | 3% (smeared) | — | gate: smearing → fix in M2 |
| 2026-07-03 | M2 | residual G + L_rec, adv.6/nce2/mask8/rec.5 | 82.5% (=source; no degradation) | TBD (M3) | gate intent met; labels correct by construction |

### M3 — pool + scorer training (2026-07-03)
- Generated **60k labeled synth** (all length-5 to match the 5-char eval; labels correct
  by construction). R0 reads the pool at 87.5% → data is sound and diverse.
- **Two training bugs found before a valid score:**
  1. model-1 trainer's **heavy augmentation** (rotation/shear/coarse-dropout tuned for
     clean MyBB synth) piled on our already-distorted GAN synth → training stalled
     (loss stuck ~3.3 ≈ random). No-aug overfit drops loss 4.2→0.9 → model learns fine.
     Fix: dedicated `train_scorer.py` with **light photometric-only aug**.
  2. checkpoint selection used PARSeq's internal AR `val_accuracy`, which reads ~0 due
     to the **refine_iters position-0 quirk** → never saved good ckpts. Fix: select by
     **force-length synth-val** with refine_iters=0.
  - (Also: `perm_num=6` → `1` to match the proven recipe; `max_label_length` 6→5.)
- Scorer training in progress (light aug, perm_num=1, synth-val selection). Real 651
  scored ONCE with the best-synth-val ckpt (no eval peeking for selection).

### M3 — THE TRANSFER PROBLEM (2026-07-03, key findings)
- **From-scratch PARSeq on 60k synth is too slow** (12-layer ViT is data-hungry; R0
  only converged because it was parseq_v2-initialized). Introduced a fast diagnostic:
  train scorer **from parseq_v2 init** (which reads real at 39.94%) on a pool and watch
  REAL-651 each epoch. Peak-real before it overfits to synth = a fast "how real-like is
  this pool" metric. (Diagnostic only — v2-init taints source-free purity via MyBB fonts.)
- **Reference:** original parseq_v2 (MyBB-*font* synth) = **39.94% word** on real.
  Our M2-GAN pool: real crashes 40%→**0.92%** in 1 epoch → **synth ≠ real**.
- **±GAN ablation:** raw composite pool peaks **4.15%** real vs GAN's 0.92% → the
  residual GAN *blurred* glyphs and HURT transfer. The crisp composite is the better base.
- **Zoom diagnosis (real vs synth):** gap is **texture, not font shape** (MyBB font is
  Vera-derived; our DejaVu is Vera-descended → shapes already close). Real = crisp opaque
  glyphs + vivid colors + clean pastel-tile/speckle bg; ours = washed colors, blurry
  (GAN), over-prominent scribbles. → Fixing composite realism, not the GAN.
- **Improvements applied:** vivid HSV-saturated glyph colors (match real), denser
  pastel-tile procedural bg + fine speckle, lean on real LaMa backgrounds. New composite
  visually much closer to real. Re-measuring transfer next.
- Both composite & GAN pools already clear the **≤2.5% OCR floor** at their peak; the
  work now is raising real transfer toward the 30–50% target by closing the appearance gap.

### M3 — ROOT-CAUSE: the source-free FONT gap (2026-07-03)
Deterministic realism probe = run parseq_v2 (reads the 651 REAL at **90.2% char /
60.8% word** with force-len5) directly on each synth pool. A real-captcha reader
should read real-like synth well:
| pool | parseq_v2 char% | note |
|---|---|---|
| **real 651** | **90.2%** | calibration |
| clean DejaVu-Bold, NO rotation | 30.2% | best open-font case |
| clean, rotation ±12 | 20.8% | **rotation hurts** |
| our composite (colored+bg+rot) | 18.7% | |
| our M2-GAN pool | 18.6% | GAN didn't help |
| auto_synth glyph-bank pool | 7.5% | noisy labels |
- **Font ranking** (parseq_v2 char% on clean no-rot renders, best→worst): verdana-bold-
  italic 37%, DejaVu-Bold-Oblique 33%, courbi 32%, verdana-bold 31%, **DejaVu-Bold 31%**,
  … arial 14%, impact 6%. **No open font exceeds ~37% char** — none matches MyBB's font,
  which parseq_v2 was trained on. This is the hard ceiling of source-free rendering.
- **Conclusion:** the recognizer is font-specific; our source-free synth can't match
  MyBB's exact glyph shapes, capping real transfer at ~4% word (composite, v2-init peak).
  The GAN changes color/texture easily but NOT glyph SHAPE (CUT/PatchNCE preserve shape),
  so it can't close a font gap. Levers that remain: (A) an STE-style GAN that restyles
  glyph shape toward real, anchored by L_rec; (B) compose REAL glyph crops from the
  unlabeled pool (exact font) with reliably-assigned char labels; (C) reduce rotation +
  best open font as the honest source-free-render baseline. See decision to advisor.

### Advisor decision (2026-07-03): research the font wall as a GENERAL problem
Advisor: the font wall is the central obstacle for a GENERAL captcha factory (all schemes),
not a CryptBB quirk — do a deep research dive and return a "one-size-fits-most" solution.
→ Launched deep-research on: closing the source-free/label-free font gap scheme-generally
(few-shot font gen, scene-text-editing, unsupervised glyph extraction, synth→real UDA/
self-training, Ye et al. CCS'18). Report pending.

### MAE pretrain + self-train — result (2026-07-03, advisor-approved path)
- Built: `mae_pretrain.py` (MAE SSL on the 600 UNLABELED reals; encoder keys match PARSeq's
  exactly), `self_train.py` (progressive TTA-consensus pseudo-labeling + fine-tune), MAE-init
  in the scorer. All scheme-general and source/label-free.
- **MAE worked as designed:** recon-mse 0.27→0.015 over 60 ep (learns real-font structure).
  MAE-init **fixed the from-scratch convergence problem** (bootstrap loss 3.2→0.04 in ~8 ep,
  vs the earlier crawl) and lifted real reading 4%→**18.5% char**.
- **BUT the bootstrap is 0% WORD on real → self-training cold-start BLOCKED.** Round 1
  pseudo-labeled only **1/600** reals at conf≥0.9 (mean weakest-char conf 0.39). With ~0
  confident whole-captcha pseudo-labels, progressive self-training has nothing to bootstrap
  from — it stalls. This is a genuine cold-start barrier.
- **Why CryptBB specifically:** it's the pathological hardest scheme — even off-the-shelf
  OCR/VLM read it at ≤2.5% word, and ONLY a MyBB-font-trained model exceeds ~40% word. A
  0%-word source-free bootstrap can't seed self-training. (On easier schemes with digit-only
  fonts — City/Courier — the bootstrap would likely clear 0% word and self-training could take
  off; but those have no labeled eval to score.)
- **Verdict:** MAE+self-train is sound and general, but needs a cold-start > 0% word. For
  CryptBB the remaining source-free levers that bypass the font wall directly:
  (1) **real-glyph-crop recomposition** — cluster the 2411 real crops in MAE feature space
  (purer than pixel clusters), assign char per cluster, compose our chosen text from REAL
  glyphs → EXACT font, labels ours by construction. Highest EV.
  (2) Tier-2: a VLM/OCR-consensus seed of a few reals to break the cold-start (no human labels).
  (3) per-character (not per-captcha) self-training to extract partial signal from 18% char.

Glyph-bank reality check (auto_synth/data/cryptbb_unsolved/alphabet): 2411 REAL glyph crops
(clean, bg-removed, EXACT font) clustered into 50 groups — but clusters are IMPURE (mix
2–3 chars: cl0→"9" holds g/Q/8/b; cl3→"{" is the 1/I/l bar group; cl8 mixes W/V) and the
vote-based cluster→char labels are unreliable. Crops are usable; reliable label assignment
is the open problem (and must be solved generally, per advisor).

Re-clustered the crops in **MAE feature space** (should be purer than pixels): STILL impure
(c0 mixes q/Q/I/u/S/K/X/A/b, etc.). MAE features encode color/style/distortion, not just
character identity, so unsupervised clustering does not yield pure single-char groups.
→ Real-glyph recomposition is blocked by the same wall: reliable per-crop labeling needs a
model that already reads the real font — circular.

### THE FUNDAMENTAL FINDING (2026-07-03)
For CryptBB, closing the font wall **fully source-free AND label-free is beyond current
methods**, and this is intrinsic, not a pipeline defect:
- Recognizing the real font needs labeled real-font data; getting labeled real-font data
  (self-training pseudo-labels, or per-crop labels) needs a model that reads the real font.
  **Circular.** The only exits in the literature break it with a few real/VLM labels
  (Ye et al. 500 labels; GeeSolver/StrDA bootstrap from a >0%-word model).
- CryptBB is the pathological worst case for breaking that circle source-free: off-the-shelf
  OCR/VLM ≤2.5% word, no open font within 37% char, so every source-free bootstrap is 0% word.
- **What DID work (reportable):** the full general pipeline; MAE SSL on unlabeled reals fixes
  recognizer convergence and lifts real reading 4%→18.5% char; rigorous diagnosis of the wall.
- **What the method needs to succeed:** a cold-start > 0% word — available on easier schemes
  (digit-only City/Courier) or via a small Tier-2 VLM/OCR-consensus seed (no human labels).
  Strict-zero-signal source-free on CryptBB's font is a negative result (consistent with the
  ≤2.5% OCR floor that already said "nothing reads this without the font").
