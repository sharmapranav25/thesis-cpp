# Plan

The full plan. For the 6–8h testable cut, see `mvp_plan.md`. For the goal, see
`GOAL.md`. For project state, see `CONTEXT.md`.

## Architecture

```
[real samples]  ──► extract bgs ──► bg pool
[scheme spec]   ──► parametric renderer ──┐
[difficulty t]  ──► difficulty mapper ────┤
                                          ├─► composite ──► refinement net (t→1) ──► synth + label
[t high]    ──► FFG glyphs (when needed) ─┘                       ▲
                                                                  │
                                            curriculum scheduler ─┘
                                                  ▲
                                                  └── student feedback (Models 3, 4)
```

The difficulty knob `t ∈ [0, 1]` is the only public input on the generator.
Internally it (a) interpolates parametric knobs between an `easy_config` and
a `hard_config`, (b) gates refinement-net invocation, (c) gates FFG invocation.

## Phase A — difficulty-controlled parametric generator

**Goal:** one knob `difficulty ∈ [0, 1]` produces correspondingly easy→hard
captchas. Drop-in usable by Models 1/2/3.

| step | description | est |
|---|---|---|
| A1 | Define `easy_config` (no rotation, single font, no boxes, plain bg) and `hard_config` (= the v2 DINOv2 winner per scheme) | 0.25h |
| A2 | Implement `interpolate(easy, hard, t)`: lerp numeric knobs; threshold-sample discrete (font set, box_prob) | 0.5h |
| A3 | Factor renderer over `SchemeSpec(W, H, n_chars, charset)`. Replace `render_darkbay_v2.py` constants. | 0.75h |
| A4 | Factor `compose.py` to accept either a text dir or a callable `text_renderer`. | 0.5h |
| A5 | Smoke test: render 16 samples each at t=0.0/0.5/1.0; visual ranking should be obvious. | 0.25h |
| A6 | Refactor `dinov2_search.py` to take `--scheme-spec`, `--real-dir`, `--bg-dir`, `--out-cfg`. | 0.5h |

**Exit criterion:** generator runs on CryptBB and DarkBay specs unchanged.

## Phase B — pixel refinement U-Net

**Goal:** close the realism gap parametric can't reach. This is what makes
"highly similar to real when needed" actually work.

| step | description | est |
|---|---|---|
| B1 | Mask-aware U-Net architecture (input `(rough_composite, text_mask)`, 4 down + 4 up blocks). | 1h |
| B2 | Loss: adversarial on full image (small PatchGAN discriminator) + L1 on text-mask region. | 1h |
| B3 | Training data: rough composites (from Phase A at t=1) paired with real samples of the scheme. | 0.5h |
| B4 | Refinement-intensity knob: scale residual `(refined - rough) * α(t)` so t=0 skips refinement, t=1 fully applies. | 0.5h |
| B5 | Training loop + checkpoint per scheme. | 2h |
| B6 | Quality sanity: real-vs-refined visual diff + recognizer accuracy bump on Model 3. | 0.5h |

**Hard dependency:** unlabeled real-sample pool per scheme.
- CryptBB: 600 available — Phase B can run.
- DarkBay: 0 unlabeled — **blocked** on data sourcing or relaxing the hands-off
  rule.

**Exit criterion:** Model 3 with refinement > Model 3 without on the same eval.

## Phase C — few-shot font generator

**Goal:** handle schemes whose glyph shapes aren't in our TTF pool.

| step | description | est |
|---|---|---|
| C1 | Pick architecture: DG-Font (content-style disentangled, public PyTorch). | 0.25h |
| C2 | Char-crop dataset: extract 50–200 char crops per scheme using Model 1's detector. | 1h |
| C3 | Fine-tune DG-Font per scheme. | 4h |
| C4 | Plug into renderer: at high difficulty, replace TTF glyph rendering with FFG output before composite. | 1h |
| C5 | Sanity: visual diff of TTF vs FFG glyphs vs real char crops. | 0.5h |

**Trigger:** only run if Phase A+B leaves a glyph-shape gap visible in real-vs-
refined comparison. Skip otherwise.

## Curriculum schedulers (one per model)

| Model | Scheduler |
|---|---|
| 1 | `RandomScheduler` — t drawn uniformly per batch (the "all distortions mixed" baseline) |
| 2 | `StaticScheduler(steps_total)` — `t = step / steps_total`; optional staged plateaus |
| 3 | `AdaptiveScheduler(target_acc)` — track student EMA acc on a small held-out synth-val; t rises when EMA > target+δ, falls when < target-δ; backtracks if acc collapses |
| 4 | `TeacherScheduler(teacher)` — TBD per teacher choice |

## Teacher (Model 4) — open

Pick one before Model 4 work starts:

- **Pretrained recognizer scoring sample utility.** Pretrained = trained on a
  separate scheme or earlier checkpoint. Each candidate batch scored by its
  loss; high-loss high-utility samples favored.
- **Real-vs-synth discriminator.** A small classifier trained to distinguish
  real from synth at each difficulty level. Its confidence becomes a "realism
  is rising" signal.
- **Distribution anchor.** DINOv2 features of unlabeled real samples used as
  a target. The teacher pulls the generator toward producing samples that
  match the anchor in feature space.

Decision deferred until after Models 1/2/3 have numbers.

## 4-model comparison harness

- Same scheme (CryptBB first; DarkBay later).
- Same student architecture, same training budget, same eval set.
- Same total samples seen during training.
- Outputs: per-model word/char on the held-out eval, plus per-model training
  curves (acc vs steps, t vs steps).
- Single CSV `mvp_results/comparison.csv` with one row per model.

## Scheme rollout

| Order | Scheme | Why |
|---|---|---|
| 1 | CryptBB | 600 unlabeled + 49.16% baseline + Model 1 v2 trainer ready |
| 2 | DarkBay | Blocked on unlabeled-pool sourcing for Phase B |
| 3 | City | No blockers; 2000 unlabeled |
| 4 | Courier Market | 2510 unlabeled; may be MyBB-based — try `model 1/`'s approach first |

## Open questions

1. **PARSeq trainer location.** Fork `model 1/src/parseq/train.py` into this
   project, or import via `PYTHONPATH`? Forking simplifies curriculum-loop
   modifications; importing avoids duplication.
2. **Refinement net: per-scheme or shared?** Per-scheme is simpler and almost
   certainly better; shared could amortize training cost across schemes if
   they share visual features.
3. **Backtracking criterion for Model 3:** how to detect "stuck" vs
   "collapsed"? Initial proposal: rolling-window stddev of difficulty < ε ⇒
   stuck (perturb); rolling-window mean acc drop > δ ⇒ collapsed (lower t,
   reset state).
4. **Teacher mechanism for Model 4** (see above).
5. **DarkBay eval contamination.** The current `data/darkbay_bg_lama/` was
   extracted from the labeled eval set itself, and the DINOv2 anchor reads
   the same set. Source unlabeled DarkBay or accept the contamination and
   report it in results.
6. **Online vs offline generation.** Generating per-batch on the fly avoids
   the 100k disk write but couples generator throughput to training
   throughput. Decision: start online; switch to offline only if generator
   becomes the bottleneck.

## Decision points

- After Phase A: does parametric-only Model 3 already beat Model 1 by a usable
  margin? If yes, B/C may not be needed for this scheme.
- After Phase B: does refinement-net Model 3 close the gap to a "realistic"
  acc level (close to what the recognizer would score on real labeled data
  with full supervision)? If yes, stop; C is unnecessary.
- After Phase B for one scheme: does the same code generalize to a second
  scheme without changes?

## Out of scope for this plan document

- Real-time deployment.
- A specific compute target beyond the local A4000.
- Specific PARSeq hyperparameter tuning beyond "use the configs Model 1 v2
  used".
