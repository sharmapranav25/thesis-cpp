# START HERE (for the Fable build agent)

You are building a **general, source-free, GAN-based captcha data factory.**
Read in this order: `GOAL.md` (what & the rules) → `RESEARCH.md` (why & the
method survey) → `DATASETS.md` (the corpus — all schemes + leakage guardrails) →
`PLAN.md` (turnkey build steps, paths, milestones).

⚠ Corpus guardrail: `JoeyRetitling` is byte-identical to the CryptBB **eval** set
(the answer key) and `DarkBay` is labeled-only — **never** use either as a GAN
input. Safe unlabeled pools: CryptBB(600), City(2000), Courier(2510). Details in
`DATASETS.md` / `configs/corpus.yaml`.

## The task in 3 lines
1. We render text we choose (so we know the label) in a plain font.
2. A **content-preserving GAN** (CUT + recognition loss) repaints it to look like
   the target scheme, learned only from that scheme's **unlabeled** images.
3. Train a recognizer on the output; score on real labeled captchas vs **57.60%**.

## The 3 rules you cannot break
- **We always know the label** — text is rendered by us, never read off a real captcha.
- **Look only at the unlabeled pool.** For CryptBB: the 600 `CryptBBCaptchas(unsolved)`. **Never** touch `model 1/data/eval/` except final scoring.
- **Open-source only.** No CryptBB `captcha.php`, no MyBB fonts, no proprietary APIs. A4000 16 GB.

## Don't repeat these (already failed here — see RESEARCH §6)
CycleGAN full translation · GlyphControl diffusion · plain composite paste.
Use **CUT (PatchNCE), not CycleGAN**, plus a **recognition-consistency loss** —
that combination is the whole point.

## Start with M0 in PLAN.md
Wire the source renderer + unlabeled-real loader + frozen PARSeq reader, smoke-test,
then build up M1→M4. Reuse `synthetic image generation/src/refine/` (G/D) and
`model 1/src/parseq/` (reader + eval) — don't rebuild them.

## Definition of done
A row in `notes/RESULTS.md`: "PARSeq trained on our source-free/label-free GAN
synth → X% word on the 651 real CryptBB (vs 57.60% source-based)." Then the same
pipeline reproduced on City by swapping one folder.
