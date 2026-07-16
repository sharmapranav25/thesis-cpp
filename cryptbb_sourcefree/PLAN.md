# PLAN — build guide for the Fable agent

Execute the architecture in `RESEARCH.md §7` (recognition-anchored CUT restyler).
This file is the turnkey build plan: what to reuse, exact paths, milestones with
go/no-go gates, and the config skeleton. Read `GOAL.md` (rules) and `RESEARCH.md`
(why) first.

## Environment
- Python: `E:/pranav/model 1/.venv/Scripts/python.exe` (Py 3.9.13, torch 2.5.1+cu121). Reuse it; add deps only if needed (kornia/lpips for losses).
- GPU: single RTX A4000 16 GB. Windows 11. No proprietary APIs.
- sqlite DLL shim in any torch script: `os.add_dll_directory("C:/ProgramData/Anaconda3/Library/bin")`.

## The corpus (all schemes)
Full registry: **`DATASETS.md`** + **`configs/corpus.yaml`**. Develop/score on
**CryptBB**; prove generality by looping the same code over **City** and
**Courier** (swap `real_dir` + canvas). 🚨 **Never** feed `JoeyRetitling`
(= the CryptBB eval set, byte-identical) or `DarkBay` (labeled-only) as inputs.

## Data paths (exact — CryptBB)
| Role | Path | Rule |
|---|---|---|
| **Target domain Y** (unlabeled real, look-at) | `E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023/` (600 imgs, 200×60) | source-free input; OK to look |
| **Final eval only** (labeled) | `E:/pranav/model 1/data/eval/` (651, filename=label) | ❌ NEVER in training/tuning. Scoring only. |
| Extracted backgrounds (optional seed) | `E:/pranav/auto_synth/data/cryptbb_unsolved/bg_lama/` | derived from the 600 unlabeled — fine to use |
| Discovered glyph bank (optional ref) | `E:/pranav/auto_synth/data/cryptbb_unsolved/` | fine to use |
| Output (all generated + ckpts) | `E:/pranav/cryptbb_sourcefree/data/` | write here |

## The generator = 2 jobs (both learned from unlabeled reals; output always labeled)
| Job | What it makes | How | Reuse |
|---|---|---|---|
| **B — recreate background** | scheme canvas (tiles/speckle/lines), NO text | sample the LaMa-extracted real bg pool; add a small bg-only GAN (StyleGAN2-ADA) only if variety is short | `auto_synth/data/cryptbb_unsolved/bg_lama/`; `auto_synth/src/part_c/extract_bgs.py` |
| **G — recreate font/glyph style** | our chosen glyphs, restyled to the scheme's font look (color/distortion/stroke/shadow), still legible | mask-conditioned CUT generator + PatchNCE + **recognition loss** (RESEARCH §7) | `synthetic image generation/src/refine/`; `model 1/src/parseq/` |

Compose G's styled glyphs onto B's background → `(image, label)`. **Label = the
rendered text, always known.** Keeping B (background) separate from G (glyphs)
means the letter-preserving anchors only have to protect G's output, and the GAN
never has to invent a background from scratch — the main failure risk shrinks to
"keep the glyphs legible," which the mask + recognition loss handle.

## Reuse from the repo (do NOT rebuild these)
- **Mask-aware U-Net generator + PatchGAN discriminator:** `synthetic image generation/src/refine/{model.py,data.py,train.py}` — ~70% of G/D. Upgrade: add PatchNCE (CUT) + recognition loss; feed mask as input channel.
- **Recognizer (for L_rec AND final eval):** `model 1/src/parseq/` (+ `third_party/parseq/`). Also `model 1/checkpoints/parseq_v2/` as a ready reader for the OCR-consistency loss (frozen at first).
- **Neutral source renderer:** write a small PIL renderer (random alphanumeric string → 200×60 clean text on plain/near-white canvas + binary text mask). Use a GENERIC font (e.g. DejaVu/Arial) — **NOT** MyBB fonts (source-free). Or reuse `model 1/src/php_synth/generate_mybb_captcha.php --text-only` but swap the font to a neutral one.
- **Eval harness:** `model 1/src/eval/case_decode.py` + `score.py` for the 651-real scoreboard.
- **CUT reference:** port PatchNCE from the official CUT repo (taesungp/contrastive-unpaired-translation) — small, ~1 file.

## Milestones (each has a go/no-go gate — don't advance until it's met)

### M0 — wiring + smoke (~half day)
- Source renderer: produce `(clean_text_img, mask, label)` triples. Eyeball 10.
- Load the 600 real Y images (Dataset). Load frozen PARSeq reader; confirm it reads the clean source renders ~perfectly.
- **Gate:** renderer + real loader + reader all run; reader reads clean renders >95%.

### M1 — CUT restyler, no recognition loss yet (~1 day)
- G (mask-conditioned) + D (PatchGAN, add ADA-style aug) + PatchNCE + soft mask-L1. Train on source→real-style.
- **Gate:** outputs visually read as CryptBB-ish (colored distorted glyphs on tile/speckle bg) AND letters still human-legible in a 4×8 panel. If letters smear → raise mask-L1 / PatchNCE weight before adding L_rec.

### M2 — add recognition-consistency loss (~1 day)
- Add `L_rec = CE(R(G(x)), label)`. R frozen PARSeq at first; then try refreshing R on (G(x),label) every N steps (self-training). Tune λ per RESEARCH §4 (fidelity high, rec moderate; watch for hallucination).
- **Gate:** R reads the *correct* label from held-out G(x) ≥ 90%. (If <90%, labels are unreliable — fix before generating a pool.)

### M3 — generate pool + train + SCORE (~1 day)
- Generate 50–100k `(ŷ, label)` → `data/synth_pool/`.
- Train PARSeq from scratch on it (reuse `model 1` trainer). Eval on the 651.
- **Gate / headline number:** word acc on 651. Beat the OCR floor (≤2.5%) decisively; **target 30–50%** (vs source-based Model 2.5 = 57.60%). Record in `notes/RESULTS.md`.

### M4 — ablations + generality (~1–2 days)
- Ablations: CUT vs CycleGAN · ±L_rec · ±mask cond · bg-sample vs bg-GAN · λ sweep · Ye-style 500-real transfer (Tier-3 upper bound, reported separately).
- Generality: rerun M0–M3 on **City** (`E:/pranav/City (live)-.../`, 2000 imgs, 324×81) changing only the unlabeled folder + canvas size. This is the real "any captcha" proof.

## Suggested folder layout (Fable creates under this dir)
```
cryptbb_sourcefree/
  GOAL.md  RESEARCH.md  PLAN.md  START_HERE.md
  src/
    render_source.py     # random text -> clean img + mask + label
    data.py              # source triples + unlabeled-real dataset
    model.py             # CUT generator (mask-cond) + PatchGAN D
    losses.py            # PatchNCE + mask-L1 + recognition(CTC/CE) + adversarial
    train_restyler.py    # the GAN training loop
    generate_pool.py     # trained G -> labeled synth pool
    eval_scoreboard.py   # wraps model 1 PARSeq train + 651 eval
  configs/restyler.yaml
  data/    (synth_pool/, ckpts/, qa_panels/)
  notes/   (RESULTS.md, running log)
```

## Config skeleton (`configs/restyler.yaml`)
```yaml
real_dir:  "E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023"
canvas:    [200, 60]           # W,H  (City: [324,81])
charset:   "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
len_range: [4, 6]              # CryptBB ~5; render a spread
source_font: "DejaVuSans-Bold" # GENERIC, not MyBB
reader_ckpt: "E:/pranav/model 1/checkpoints/parseq_v2/parseq-epoch=11-val_accuracy=57.260.ckpt"  # frozen R for L_rec (a general reader; NOT trained on eval)
loss_weights: {adv: 1.0, patchnce: 1.0, mask_l1: 5.0, rec: 0.5}
use_ada: true
batch: 16
steps: 30000
out_dir: "E:/pranav/cryptbb_sourcefree/data"
```
Note on `reader_ckpt`: using `model 1`'s PARSeq as the frozen legibility judge is
a pragmatic start (it was itself trained label-free on synth, never on the eval
set). For strict source-free purity in the final writeup, retrain R once on the
clean source renders only. Flag whichever you use in `notes/RESULTS.md`.

## The one number that matters
PARSeq-on-GAN-synth word accuracy on the 651 real captchas — **source-free,
label-free** — vs 57.60%. That is the thesis result.
