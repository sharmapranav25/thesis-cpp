# Deep research — source-free, general synthetic captcha regeneration (2026-07-06)

20 primary sources (CCS/TOPS, ICCV, NeurIPS, AAAI, IJCV, TPAMI, arXiv), 24/25 claims verified 3-0.
Question: can we regenerate a scheme's captchas from ONLY an unlabeled folder, no per-scheme code,
train on synth, transfer to real?

## Headline
**No published method meets our iron rules** (source-free + label-free + zero hand-tuning +
point-at-folder + 16GB). Every canonical attack is a **procedural-generator + GAN HYBRID** that
needs BOTH (a) ~500 human-LABELED real captchas per scheme and (b) per-scheme manual parameter
configuration. So the gap we're filling — replace the labels with self-training, replace the
hand-tuning with auto-fit — **is the novel contribution**, not a solved problem.

## What the literature confirms (each 3-0 verified)
1. **Canonical solvers are hybrids, not pure GANs** (Ye et al. CCS'18 / TOPS'20; Li et al.
   CycleGAN'20; Info.Sci.'21). A *parametric renderer* makes the base captcha; a GAN only adds
   pixel-level security features. → our procedural-first instinct is correct.
2. **The field's answer to the FONT WALL = font-by-SELECTION, not glyph reconstruction.** They
   supply a **large free-font library** and let the learner pick the best-fitting font per scheme.
   (We currently use only ~4 fonts — this is our biggest concrete miss.)
3. **GAN restyling does NOT cross the font wall** — preserves glyph shape. Confirms our CUT/PatchNCE
   failure. Don't revisit it.
4. **Synth-only transfer is scheme-dependent, mirroring our split**: easy schemes 85-88% from synth
   alone (Amazon 88.4%, Sina 85.0%); hard/exotic-font schemes collapse to 22-36% (MS two-layer
   22.4%, Wikipedia 26.6%) — lifted to 75-88% ONLY by the labeled fine-tune. Our CryptBB ~30% vs
   DarkBay ~91% is exactly this pattern. Self-training is what must replace that labeled fine-tune.
5. **Auto-fitting render parameters from samples is real**: inverse de-rendering (Shimoda ICCV'21)
   parses text/size/font/effects from a raster via a parser + **differentiable-rendering**
   reconstruction. This is the "auto-configure, no hand-tuning" mechanism — but its font is a
   fixed 100-font classifier, so it too hits the exotic-font wall unless the library covers it.

## Font/glyph-shape recovery methods — all have a disqualifying catch (source-free, exotic font)
- **FontDiffuser (AAAI'24), Diff-Font (IJCV'24)**: one-shot font gen, but need the exemplar glyph's
  CHARACTER IDENTITY to be KNOWN (our pool glyphs are unknown-identity) + heavy multi-font supervised
  pretraining. Optional *booster*, not a standalone recoverer.
- **TextStyleBrush (TPAMI'23), TextCtrl (NeurIPS'24)**: per-IMAGE restyling (single source at
  inference), shape-preserving (like the failed CUT), need 200k+ paired supervised pretraining.
- **CLIP font transfer / Differentiable Variable Fonts**: prompt-driven or stay inside a designer's
  axis space — neither infers appearance from a sample folder.
→ None is a drop-in source-free glyph-shape recoverer. Font-by-selection from a big library is the
practical answer; one-shot font-gen only where the library fails.

## Recommended architecture (the report's synthesis — medium confidence, our open question)
**HYBRID, self-configuring:**
1. **Procedural renderer, parameters AUTO-FIT to the unlabeled pool** — font selected from a **LARGE
   free-font library** (Google Fonts etc.), plus warp/noise/spacing/rotation/color measured or
   discriminator/differentiable-fit from the samples. (This is the analyze_pool idea, done right +
   a much bigger font set.)
2. **Font-general recognizer** (Qwen2-VL) trained on the synth.
3. **Self-training / pseudo-labeling / entropy-min on the unlabeled real pool** to REPLACE the ~500
   labeled fine-tune the literature depends on.
4. **One-shot font generation as a FALLBACK glyph booster** only where the font library can't cover
   the scheme's letterforms (the CryptBB case).

## The open question that IS the thesis
Can label-free self-training fully replace the ~500-label fine-tune and close the hard-scheme gap
(22-36% → 70-88%) with zero human labels? No paper shows this end-to-end. **We already have partial
proof**: CryptBB 30.4% and DarkBay 91.0% self-trained, no labels — i.e. we're already doing the
unproven part. The contribution is making the *generator* self-configuring so the whole loop is
point-at-a-folder.

## Concrete, actionable next steps (ranked)
1. **Expand the font library** from ~4 to a large free-font set + font-SELECTION (biggest, cheapest
   lever the literature points to for the font wall). Source-free (generic fonts).
2. **Auto-fit render params from the pool** (the measure-from-unlabeled approach) — validated as the
   right mechanism; the earlier attempt was the right idea, just needs the font-selection + a cleaner
   build. Prove it on the SCORE, not just the look.
3. Keep the VLM + self-training loop (already ahead of the literature's label-free bar).
4. One-shot font-gen (FontDiffuser) only as a later booster for exotic fonts.

Sources: Ye CCS'18 (eprints.whiterose.ac.uk/151526), Ye TOPS'20 (10.1145/3378446), Li CycleGAN
(arxiv 2008.11603), Info.Sci.'21 (S0020025521007301), Shimoda ICCV'21 (de-rendering),
FontDiffuser (arxiv 2312.12142), Diff-Font (arxiv 2212.05895), TextStyleBrush (arxiv 2106.08385),
TextCtrl (NeurIPS'24), SMILE entropy-min DA (arxiv 2202.11949).
