# Paste this to Fable

You are the build agent for a thesis project. Your workspace is
`E:\pranav\cryptbb_sourcefree\`. Read these first, in order: `START_HERE.md`,
`GOAL.md`, `RESEARCH.md`, `DATASETS.md`, `PLAN.md` — they are the full spec,
method survey, data registry, and milestone plan. Follow them exactly.

**Mission.** Build a general, source-free, open-source **captcha data factory**:
given a folder of UNLABELED captchas from a scheme, it produces LABELED synthetic
training data that looks like that scheme. The generator has two jobs —
(1) recreate the **background**, (2) recreate the **font/glyph style** — composed
onto text *we* render, so every output is labeled by construction. Prove it on
CryptBB: train PARSeq on the synth, score on the 651 real captchas, vs the
source-code baseline (Model 2.5 = **57.60%** word) and the OCR floor (≤2.5%).

**Method (fixed — RESEARCH.md §7).** Render our chosen text in a neutral font →
content-preserving GAN (**CUT / PatchNCE, NOT CycleGAN**) restyles the glyphs to
the scheme's font look, anchored by a **text mask + a recognition-consistency
loss** (a recognizer must read the correct text out of the output) → compose onto
a **background sampled from the LaMa-extracted real pool** (or a small bg-only
GAN). Reuse `synthetic image generation/src/refine/` (mask-aware U-Net + PatchGAN)
and `model 1/src/parseq/` (recognizer + eval). Do not rebuild them.

**Non-negotiable rules.**
- We ALWAYS know the label — text is rendered by us, never read off a real captcha.
- Look ONLY at unlabeled pools in `configs/corpus.yaml`. NEVER train on / feed the
  GAN: `model 1/data/eval` (the 651 CryptBB answer key), `JoeyRetitling`
  (byte-identical to that eval set), or `DarkBay` (labeled-only). Eval = final
  scoring only.
- Open-source, local, RTX A4000 16 GB. No CryptBB source, no MyBB fonts, no APIs.
- Do NOT repeat what already failed here: plain CycleGAN (content drift),
  GlyphControl diffusion, naive composite.

**How to work.**
- Execute PLAN.md milestones **M0→M4 in order; each has a go/no-go gate — don't
  advance until it's met.**
- **Verify before scaling:** at M2, prove the recognizer reads the *correct* text
  out of held-out generated images **≥90%** before generating a big pool. A
  mislabeled pool is worse than none — fix labels first.
- Save 4×8 real-vs-synth QA panels to `data/qa_panels/` each stage. Log every run
  (config, legibility %, real word%) in `notes/RESULTS.md`.
- Keep the code **scheme-general**: parameterize by `configs/corpus.yaml` so the
  same pipeline runs on City/Courier by swapping one entry. Prove generality at M4.

**Definition of done.** A row in `notes/RESULTS.md`: "PARSeq trained on our
source-free, label-free GAN synth → X% word on the 651 real CryptBB" with X well
above 2.5% (target 30–50%), plus the same pipeline reproduced on City by changing
one config entry.

**Start now with M0.** At the end of each milestone, show me the QA panel and the
gate result before moving on.
