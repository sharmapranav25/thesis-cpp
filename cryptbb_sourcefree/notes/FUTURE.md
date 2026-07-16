# FUTURE WORK — deferred ideas (source-free, label-free, general)

Ideas that are sound but NOT yet implemented. Each must stay inside the iron rules:
we render the text (labels ours by construction), no scheme font, no scheme source,
no human labels, look at the UNLABELED pool only (measure style, never read answers).

---

## 1. Ghost-character detection for real-background reuse  ⚠️ (the reason we don't steal backgrounds yet)

**Problem.** Lever "composite onto LaMa-extracted real backgrounds" is tempting (real texture),
but the LaMa inpaint that erases the real text leaves faint **ghost letters**. If a ghost survives
into a background we then label with OUR word, the image secretly contains a letter we didn't count
→ our own label is WRONG → training on it makes the model worse. (Logs: this attempt HURT.)
So real backgrounds are blocked until we can *reliably* reject ghosted ones.

**Why perfect detection is hard.** A perfect "is this smudge a letter?" detector is itself a reader
for this exact font — the thing the whole project lacks. A faint ghost vs. background texture is
genuinely ambiguous. BUT we only need a binary "letter-shaped stuff here? yes/no", which is far
easier than reading.

**Detectors, worst → best:**
1. **Dark-blob test (current, `background.py:_has_ghost_glyph`)** — largest dark connected component
   in the central band. Crude: misses faint ghosts, false-positives on legit dark bg.
2. **VLM ghost-gate (recommended).** Run Qwen2-VL on the BARE background (before compositing). If it
   emits any alphanumeric above a low confidence → reject. Clean (general VLM, not their font),
   label-free (output is a keep/trash switch, never a label), tests exactly what matters.
3. **Mask-targeted residual check (most accurate, IF masks were saved).** The backgrounds were made
   by inpainting away the real text. If the inpaint MASK still exists, only inspect ink *inside that
   region* ("did any dark structure survive where letters used to be?") — far fewer errors than a
   global search. Check whether the LaMa pipeline saved masks.
4. **Consensus.** blob-test AND VLM AND OCR; reject if ANY flags a letter.

**Key principle — over-reject.** Costs are lopsided: dropping a good background costs ~nothing
(backgrounds are free); keeping a ghost poisons labels. So tune every detector conservative and keep
only the top ~20% you're most sure are clean.

**Cleanest alternative (already the default):** don't detect at all — generate a PROCEDURAL background
whose *statistics* match the reals (see `analyze_pool.py` bg stats). No stealing = no ghost = no
detector needed. Only pursue real-background reuse if the thesis specifically wants real texture.

---

## 2. Bigger reader / stack the two levers (bigger wins than any generator tweak)
Recorded in RESULTS.md but restated: on the font-general VLM path, image-realism tweaks are worth
low single-digit word% on the hard scheme (~0 on easy). The two untried levers that dominate:
- **Stack self-train on the 29.8% PHP checkpoint** (full ~600 pool) → should clear the 30.4% peak.
- **Qwen2.5-VL-7B** instead of 2B.
Do the generator work for the *realism* contribution; do these two to move the *number*.

---

## 3. Font re-ranking — deeper version
`analyze_pool.py` ranks the available generic fonts by aggregate ink-shape distance (stroke width,
aspect) to the reals. Deeper, deferred: learn/interpolate a font from unlabeled glyph exemplars
(few-shot font generation — DG-Font/FontDiffuser). Heavy; only lever that truly touches glyph SHAPE
(the hard wall), but capped by "no open font matches the exotic target" for font-specific readers.
Less critical now that a font-GENERAL VLM sidesteps the wall.
