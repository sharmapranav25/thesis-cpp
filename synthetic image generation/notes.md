# Notes — alternatives & ideas to revisit

Things that came up during DarkBay synth work but were deferred. None of these
are scoped now; recorded so they don't get lost.

## Alternative architectures for "model understands captcha text style"

These are options we considered instead of (or in addition to) the DINOv2
parameter auto-tuner. Each goes further than the auto-tuner in the direction
of "let the model do everything" but costs more.

### 1. Few-shot font generation networks (FFG-Net, DG-Font, AGIS-Net)

Given 5-10 examples of a font, generate any character in that style. Replaces
the parametric renderer with a generative network that *learns* glyph style
from real samples.

- **Pro:** no need to express the font as a parameter set; arbitrary fonts.
- **Con:** bigger model, paired-character training setup, more code.
- **When to revisit:** if the parametric DINOv2 tuner hits a ceiling because
  the right styling isn't expressible in our parameter space (e.g., subtle
  glyph distortions specific to the captcha's font that no Windows TTF
  matches).

### 2. Diffusion + LoRA fine-tune on the captcha set

Fine-tune a small LoRA on Stable Diffusion using ~1000 real captchas. At
inference, condition on a text prompt + char layout and generate a captcha-style
image directly.

- **Pro:** very general; same recipe for any new scheme.
- **Con:** GlyphControl / AnyText already failed on captchas in this project
  (per memory: content fidelity issues). LoRA might do better but is unproven
  for a domain this narrow. Setup cost is high.
- **When to revisit:** only if both DINOv2 auto-tuning *and* the pixel-level
  refinement net underperform. Treat as a last resort.

### 3. Pixel-level refinement net (Phase 4 in the original plan)

Small mask-aware U-Net: input `(rough_composite, text_mask)` → output
`realistic_captcha`. Adversarial loss against real captchas + L1 on text-mask
region (preserves content).

- **Pro:** most general — learns at pixel level, captures things the renderer
  can't parameterize (JPEG noise, anti-alias kernel, color bleed at text edges,
  exact paper texture, etc.).
- **Con:** ~half day of work, needs ~500+ real captchas as discriminator's
  "real" set, mode-collapse risk.
- **When to revisit:** the natural next escalation if the DINOv2 auto-tuner
  gets us 60-80% of the way and we want to close the rest. The renderer-tuner
  and the refinement net are complementary, not alternatives — auto-tune
  produces a strong starting composite, refinement net polishes.

## Auto-tuner-specific ideas (deferred)

- **Per-scheme param search budgets:** start with 200 random configs; if loss
  curve hasn't plateaued, extend to 500. CMA-ES would converge faster than
  random search but adds a dependency.
- **Real-image batching for the anchor:** currently the search uses the *mean*
  feature of all real images as the anchor. If real DarkBay has multiple
  visually distinct subtypes, mean-feature blurs them; a multi-anchor distance
  (assign each synth to its nearest real subtype) would be more honest. Only
  worth doing if the simple anchor underfits.
- **Free knobs we didn't add:** per-char color drawn from a learnable palette
  (sampled from real char crops); char overlap fraction (some real DarkBay
  captchas have touching chars); independent box rotation vs char rotation.

## Bg-extraction quality

- LaMa hallucinates ~20% of the time on CryptBB (small bg-pattern blocks at
  former text positions). DarkBay extraction quality not yet QA'd at the same
  level; do a 20-sample review when we get back to it.
- A **tighter mask via saliency thresholding inside each detector bbox** is
  the cheap improvement (less area to inpaint = less hallucination). Worth
  doing if the GAN/composite eval suffers visibly from bg artifacts.

## CryptBB Phase 4 (refinement net) — unfinished

- Full CryptBB pipeline (extract → composite → train) was set up; user took
  over training. If that result lags v2's 49.16%/86.86% by 5-15 pts, the
  refinement net is the next move there too — same architecture as for
  DarkBay, just trained on the CryptBB pair.
