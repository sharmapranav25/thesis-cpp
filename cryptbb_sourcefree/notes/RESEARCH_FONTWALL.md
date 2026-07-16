# Beating the font wall — research synthesis (2026-07-03)

Deep-research run (partial; synthesis agent hit a session limit) + verified claims +
our own experiments. Question: how to make source-free/label-free captcha synth transfer
to real, **scheme-generally**.

## The reframe (the key insight)
Stop trying to *render* the target font perfectly — it's impossible source-free (no open
font exceeds ~37% char agreement with MyBB; our probe). Instead: **bootstrap** a recognizer
on imperfect source-free synth, then **adapt it to the real font using the UNLABELED real
pool** via self-supervised pretraining + self-training. This is exactly what the *generic*
captcha solvers and STR domain-adaptation papers do. The synth only needs to be good enough
to bootstrap; adaptation on unlabeled reals closes the font gap. Scheme-general by
construction (swap the unlabeled folder), and label-free (pseudo-labels, not human labels —
allowed per RESEARCH.md Tier-2 fallback).

## Prior art (from the research; ✓ = adversarially verified 3-0)
- **GeeSolver (IEEE S&P/Oakland 2023) — "Generic, Efficient, Effortless Solver w/ Self-
  Supervised Learning".** Directly answers "replace the ~500-real fine-tune with zero human
  labels". Insight: captcha distortions act like image *masks*, so a **masked-autoencoder
  (MAE) self-supervised pretrain on UNLABELED real captchas** learns the real font's features;
  then a small head + light labels solve it. Generic across schemes. → strongest lead.
- **Generic Solver, WWW 2020.** Unsupervised decomposer segments a captcha into character
  blocks, then recognizes — generic, reduces labeled-real need. Prior art for the glyph route.
- **StrDA (2024) ✓** — progressive/stratified self-training for STR: partition unlabeled
  target into subsets of increasing domain gap, self-train sequentially (not one-shot).
  Uses labeled synth + unlabeled real; label-free adaptation. Handles a large initial gap.
- **Ye et al. CCS 2018** (the north star): GAN synthesizer learns per-scheme style from real +
  hand-approximated fonts, base solver on synth, **fine-tune on ~500 real LABELED** (lifts
  e.g. 4.7%→78%). NOT source-free (needs real labels + per-scheme fonts) — the thing we remove.
- **CycleGAN captcha attack (2020)** — corroborates the "pixel GAN restyles texture/color but
  not letterform" wall; still needs real labels. (Matches our CUT result.)
- **Few-shot font generation:** FUNIT (ICCV'19) & **MX-Font (ICCV'21) need NO component
  labels** and generate a font from <10 reference glyphs, no test-time finetune; can train on
  rendered glyph *images* (not TTF). **DA-Font (2025) ✓ transfers glyph SHAPE** (corner-
  consistency + elastic-mesh losses) — the shape property CUT lacks. Caveat: all need
  *identified/segmented* reference glyphs (the labeling problem) and target CJK; Latin-captcha
  use needs validation. A route for the glyph-alphabet, not a full pipeline.
- **Scene-Text-Editing:** MOSTEL (semi-supervised, labeled synth + **unpaired real**, stroke-
  guidance = shape-level edits) and **TextStyleBrush ✓ (one-shot, self-supervised, transfers
  font+geometry+color from ONE example)**. TextStyleBrush has no open weights; heavy.

## Ranked options for a one-size-fits-most factory
| approach | transfer potential | scheme-general | source/label-free | fits A4000 |
|---|---|---|---|---|
| **Self-sup (MAE) pretrain on unlabeled real → self-train w/ synth bootstrap** | **high** | **high** | **yes** (pseudo-labels) | **yes** |
| Progressive self-training (StrDA) on synth-bootstrap | med-high | high | yes | yes |
| Unsupervised glyph extraction → real-glyph alphabet | med-high | medium | borderline (glyph ID) | yes |
| Few-shot font gen (MX-Font/DA-Font) to render target font | med | medium | needs seg. glyphs | yes |
| STE (MOSTEL) restyle glyph shape | med | medium | semi-sup | yes |
| Better source-free render only (best font, less rotation) | low (~a few %) | high | yes | yes |
| CUT/CycleGAN pixel restyle (what we tried) | very low | high | yes | yes |

## Recommendation (implement locally — no cloud agents needed)
**Self-supervised bootstrap + self-training**, in order of build:
1. **Best-effort source-free synth** (crisp composite, best open font, real LaMa bg, reduced
   rotation) → bootstrap recognizer. Already reads real ~40% char.
2. **MAE self-supervised pretrain** the recognizer's ViT encoder on the 600 UNLABELED reals
   (GeeSolver idea) → learns real-font features label-free. Biggest single lever.
3. **Progressive self-training** on the 600 unlabeled reals: TTA/consensus pseudo-label,
   confidence-filter, add to training, iterate (StrDA-style). Closes the font gap w/o labels.
4. Eval on the 651 real ONCE.
All scheme-general (swap the unlabeled folder), label-free, source-free, single-GPU.
**Guardrail:** self-training uses the 600 `unsolved` pool ONLY — never the 651 eval.
