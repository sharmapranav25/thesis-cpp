# Goal

## What this project is

Build an **automated synthetic-captcha generator that supports curriculum
learning**. The generator must:

1. Produce labeled synthetic captchas for any scheme without per-scheme code.
2. Expose a controllable **difficulty / realism axis**: at the low end, easy
   clean samples for a beginner student; at the high end, samples
   indistinguishable from real captchas of that scheme.
3. Be drivable by a curriculum schedule (static or adaptive) and by student
   feedback.

The synth generator is one component. The full study is a 4-model comparison
of how curriculum and feedback affect a PARSeq recognizer's accuracy on real
captchas.

## The 4-model study

| Model | Curriculum | Adaptation | Teacher | Question it answers | Expected acc |
|---|---|---|---|---|---|
| 1 | ❌ | ❌ | ❌ | Does naive mixed-difficulty synth alone train a working recognizer? | 10–30% |
| 2 | ✔ static | ❌ | ❌ | Does a fixed easy→hard schedule beat random? | 20–40% |
| 3 | ✔ | ✔ student feedback | ❌ | Does the generator self-improving from student loss/acc beat static? | 30–60% |
| 4 | ✔ | ✔ | ✔ external | Does an external teacher beat self-supervised adaptation? | TBD |

All four use the same generator and the same student (PARSeq). What differs is
**how training data is selected over time**.

Model 4's teacher mechanism is currently undecided — see `plan.md` for
candidates.

## Three generator capabilities (composed under one difficulty knob)

| Capability | Role | When invoked |
|---|---|---|
| Parametric renderer (TTF + per-char styling + bg sampling) | Cheap; covers easy→medium difficulty | Always |
| Pixel refinement U-Net (text-mask-aware GAN translation) | Closes the gap between parametric output and "indistinguishable from real" | High difficulty |
| Few-shot font generator (DG-Font / AGIS-Net) | Handles schemes whose font isn't in our TTF pool | Only when parametric+refinement still wrong on glyph shape |

The difficulty knob composes these: at `difficulty=0` only the parametric
renderer at its easiest setting is used; at `difficulty=1` the full stack
(parametric + FFG glyphs + refinement) runs to produce real-like output.

## "Highly similar to real when needed" — what it means concretely

When the curriculum or downstream consumer requests `difficulty=1`, the
generator must produce captchas that:
- Pass a visual A/B check against real samples of that scheme.
- Drive a recognizer's accuracy on those samples to a level comparable to
  recognizing actual real samples.

The parametric renderer alone cannot reach this (cosine distance plateaued at
0.189 on DarkBay; per-image-style mismatch is visible). The refinement net is
what closes this gap.

## Constraints carried over from `model 1/`

- **No human labels in training the recognizer.** Synth labels come from the
  renderer.
- **Eval sets are hands-off:** `model 1/data/eval/` (CryptBB, 651), the
  1000-image DarkBay labeled set, any future per-scheme eval set.
- **Per-scheme generality:** new scheme = new config + a folder of unlabeled
  real samples. No new code.

## Validation criteria

- **Curriculum hypothesis:** Models 1/2/3 ranking on a real eval set is
  1 < 2 < 3 in word/char accuracy. If the ranking holds, the curriculum +
  adaptation hypotheses are validated.
- **Realism claim:** at max difficulty, samples pass a visual A/B check *and*
  the recognizer's per-difficulty accuracy curve plateaus near real-eval
  accuracy.
- **Generality:** the same code runs on a second scheme by changing only a
  config file and a real-samples directory.

## Out of scope

- Real-time deployment. This is a research/training system.
- Beating the prior CryptBB baseline (49.16% w / 86.86% c) on absolute terms
  for Model 1 — Model 1 is *expected* to underperform; it's the floor.
