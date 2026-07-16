# GOAL — a general, source-free captcha data factory (GAN-based)

## The vision (one sentence)
Point it at a folder of **unlabeled captchas from ANY scheme** and it produces
**labeled synthetic training data that looks like that scheme** — using only
**open-source models**, **no source code**, **no human labels**.

## The one iron rule
**We always know the answer to every fake we make**, because *we* choose the
text. The recognizer's label comes from what we typed, never from reading a real
captcha. Only the *look* (font style, colors, distortion, background) is learned.
Every past failure in this repo broke this rule; every success kept it.

## Chosen approach — "render, restyle glyphs, compose on recreated background"
```
 we type "aB3xK"  ──►  plain renderer  ──►  clean glyphs + text mask
   (label known)                                    │
                        content-preserving GAN  ─────┤  JOB 2: recreate FONT STYLE
                        (mask + recognition loss)    │  (restyle glyphs, keep them legible)
                                                     ▼
                                          styled glyphs "aB3xK"
                                                     │
 unlabeled reals ──► bg pool / bg-GAN ──────────────┤  JOB 1: recreate BACKGROUND
   (LaMa-extracted, no text)                         │  (tiles/speckle, no letters)
                                                     ▼
                                    compose (glyphs onto background)
                                                     │
                                                     ▼
                         (image that says "aB3xK",  label="aB3xK")  ← labeled data
```
(All the "look" — style + background — is learned only by WATCHING the scheme's
unlabeled images. The text, and therefore the label, is always ours.)
Why this is general: **it never has to read a real captcha.** It only has to
learn the *look* of a scheme from unlabeled images, then repaint text we already
know. Swap the unlabeled folder → new scheme, no new labeling.

## What the generator must do (two jobs — output is ALWAYS labeled)
The "look" of a scheme is two separable things; the generator recreates both:

1. **Recreate the background** — the scheme's canvas: colors, tiles, speckle,
   lines, texture (CryptBB = pastel tile mosaic + white speckle). No text in it.
2. **Recreate the font/glyph style** — how this scheme draws characters: color
   per char, distortion/rotation, stroke weight, shadow, anti-aliasing.

Compose: our chosen glyphs, restyled to (2), placed on (1). **The label is the
text we typed** — known by construction, never read off a real image. So the
factory emits `(image, label)` pairs, ready for training, every time.

Design refinement this implies: **don't make one GAN hallucinate the background
from a blank canvas** — that's where GANs wander. Get the background from the
real pool (LaMa-extracted) or a small background-only GAN (safe: no letters),
and let the main GAN focus its capacity on **job 2, the font style**, which is
the part that must stay legible. Backgrounds have no letters to wreck.

## The whole game is content preservation
A plain CycleGAN was tried here before and **failed** — it smeared/changed the
letters while restyling, so labels stopped matching. Route B only works if the
GAN restyles the *style* while keeping the *letters exactly*. Anchors that force
this (all three, belt-and-suspenders):
1. **Text-mask conditioning + L1** — feed the letter mask; penalize changes to letter pixels.
2. **OCR-consistency loss** — a recognizer must read the SAME text out of the GAN's output (legibility guarantee).
3. **Contrastive (CUT) rather than cycle** — better content preservation than CycleGAN.

## Scoreboard: CryptBB
CryptBB is the one scheme with a labeled eval set (651) AND a known
source-code-based number to beat: **Model 2.5 = 57.60% word**. So we develop and
measure here: train PARSeq on our GAN synth → eval on the 651 real → compare to
57.60%. Clearing, say, 35–50% with **zero source code and zero labels** is a
strong result. Then prove generality on City / Courier (DarkBay is blocked — see
`DATASETS.md`).

## Hard rules (inherited, non-negotiable)
- Look ONLY at unlabeled pools. CryptBB: the 600 `CryptBBCaptchas(unsolved)` (200x60). NEVER `model 1/data/eval/` except final scoring.
- Open-source, local, A4000 16 GB. No proprietary APIs, no CryptBB `captcha.php`, no MyBB fonts.

## Where to look next
- `RESEARCH.md` — the full method survey (what to use, what failed here, why).
- `PLAN.md` — the concrete build plan Fable executes (components, milestones, commands, what to reuse from the repo).
- `notes/` — running notes. `data/` — generated + intermediate data.
