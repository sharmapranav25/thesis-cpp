# Future work — deferred items

Notes for later. None of these are blocking the current MVP.

## FFG (few-shot font generation): DG-Font / AGIS-Net / FFG-Net

**Why deferred, not skipped:** the parametric renderer's 4 Windows TTF fonts
(Times Bold, Georgia Bold, Cambria Bold, Consolas Bold) happen to be a
reasonable approximation for DarkBay and CryptBB — both use serif-style
glyphs in that family. That's a coincidence of the schemes we have, not a
property of the architecture.

**When it becomes the critical path:** a new scheme whose font shape isn't
in our TTF pool. Symptoms:
- Visible glyph-shape mismatch in real-vs-synth comparisons that the
  refinement net can't fix (because the refiner is text-mask-anchored —
  it preserves whatever glyph the parametric layer drew).
- Recognizer trained on parametric synth fails on real eval primarily on
  per-character error, not per-position error.

**Architecture slot it fills:**
```
[FFG-net]  ──► glyph images  ──┐
                                ├──► [Generator] ──► refinement net ──► output
[real bg pool] ──► bg ──────────┘
```
Replaces the TTF rendering branch inside `render_darkbay_v2._render_char` —
instead of `font.getbbox(ch)` and `mdraw.text(...)`, we'd call
`ffg_net.render(ch, style_refs)` for the per-char glyph image.

**Pick:** DG-Font is the simplest starting point — content-style disentangled,
public PyTorch, well-documented. AGIS-Net is more powerful but heavier.

**Training data per scheme:** 50–200 real char crops. Source: run
`model 1/checkpoints/detector_v2/best.pt` on real captcha samples to extract
crops. No labels needed for FFG training itself (it's style transfer, not
classification).

**Estimated effort:** ~1 day for the first scheme (download repo, adapt
data loading, train, integrate). Per-additional-scheme: ~2h (just retrain
the FFG style adapter on new char crops).

**Trigger:** revisit when (a) DarkBay/CryptBB recognizer numbers are
locked in, and (b) we move to a scheme where the TTF approximation visibly
breaks. Probably City (324×81, layout differs from MyBB family) is the first
test.

## RealBgGenerator: residual bleed-through

**Issue:** the real-bg path masks the original captcha's text region using
the LaMa-vs-real diff, dilates by 4px, then `cv2.INPAINT_TELEA` fills it.
On some samples the mask is incomplete (e.g., when original chars and bg
share similar grey values), so visible char fragments survive into the
final composite alongside our synth text.

**Fix candidates (in cost order):**
1. Bump `ORIG_TEXT_DILATE` to 4–6px. Cheap, may over-mask but eliminates
   most edge bleed.
2. Use the existing detector (`model 1/checkpoints/detector_v2/best.pt`)
   to get tight original-text bboxes, union with the diff mask. Catches
   chars the diff misses.
3. Use **LaMa** for the local inpaint instead of cv2.TELEA. LaMa
   propagates texture across larger gaps. Already imported via
   `simple_lama_inpainting`. Slower per-sample but possibly worth it for
   the t=1 endpoint.
4. Cache the inpainted bg per real-image rather than re-inpainting per
   sample (since our text bboxes mostly overlap the original text bboxes).

## Other deferred items

- **DarkBay unlabeled data sourcing.** The current bg pool is extracted from
  the labeled eval set — eval contamination. To run a clean comparison
  study on DarkBay, we need 500–1000 unlabeled DarkBay captchas not in the
  eval. Until then, all DarkBay refinement-net + FFG training carries a
  contamination footnote.

- **Refinement net trained on a clean unlabeled pool.** Once DarkBay
  unlabeled is available, retrain v4-style refiner on it. Same architecture,
  no eval contamination.

- **Cross-scheme generality refactor.** `render_darkbay_v2.py` hardcodes
  W=200, H=50, N_CHARS=6, CHARSET. Refactor to a `SchemeSpec` dataclass
  (W, H, n_chars, charset) so the same renderer drives any scheme.
  ~45 min. Do this before adding the second scheme, not after.

- **Curriculum schedulers + 4-model trainer.** Out of scope for the
  generator MVP. Belongs to the `model 1/` recognizer-training side. See
  `plan.md` Phase A2/A3.

- **Model 4 teacher mechanism.** Three candidates listed in `plan.md`;
  pick one only after Models 1/2/3 numbers exist.

- **Pixel refinement net trained on full composite vs bg-only.** Current
  v4 trains on full composite with mask-protected text. Alternative is to
  train a bg-only generator (no text in input or output) and composite
  text afterward — cleaner separation. Worth comparing if v4 still has
  text-bg integration artifacts.
