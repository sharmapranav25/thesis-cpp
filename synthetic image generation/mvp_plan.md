# MVP plan — synth generator with a difficulty knob

**My deliverable for this MVP:** a synthetic-captcha generator that takes a
single difficulty input `t ∈ [0, 1]` and produces correspondingly easy→hard
samples. Nothing else. Trainer integration, schedulers, recognizer training,
eval — all downstream of this and out of scope for the MVP.

See `GOAL.md` for the project goal, `plan.md` for the full multi-phase roadmap.

## Deliverable

A Python module + CLI that, given a difficulty `t` and a count `n`, writes
`n` labeled DarkBay-style synth captchas to disk:

```
"E:/pranav/model 1/.venv/Scripts/python.exe" src/generator/render.py \
    --t 0.7 --n 1000 --out data/mvp_samples_t0.7
```

Output dir contains:
- `0000000.png` … `<n>.png` — RGB 200×50 captchas
- `labels.csv` — `filename,label`
- `bboxes.jsonl` — per-char bboxes (compatible with `model 1/`'s trainer
  conventions)

Plus a one-shot visual smoke artifact: `mvp_results/smoke_t0_05_10.png` — a
grid of 16 samples each at `t=0.0`, `t=0.5`, `t=1.0` for visual ranking
verification.

## Pass criterion

Visual: at `t=0` samples are clean (no rotation, single font, no boxed chars,
plain bg); at `t=1` they look like the v2 DINOv2-tuned output (which already
matches real DarkBay style in bg + boxed-text style); `t=0.5` interpolates
visibly between the two. Samples at all `t` are correctly labeled.

That's it. No accuracy numbers, no curriculum, no models trained.

## Time budget (~2h)

| Step | Description | Est. h |
|---|---|---|
| 1 | `EASY` config (no rot, single font, no boxes, no jitter) | 0.1 |
| 2 | `HARD` config (load `configs/darkbay_renderer.json`) | 0.05 |
| 3 | `interpolate(easy, hard, t) -> RendererConfig` — lerp numerics, threshold-sample discretes | 0.4 |
| 4 | `ParametricGenerator.render(t, n)` — wraps `render_darkbay_v2.render_one` + `compose.py` jitter+composite+border | 0.5 |
| 5 | CLI (`src/generator/render.py`) with `--t`, `--n`, `--out`, `--seed` | 0.25 |
| 6 | Smoke grid: 48 samples → `mvp_results/smoke_t0_05_10.png` | 0.5 |
| 7 | Sanity-check labels match rendered text | 0.1 |

## Out of scope (handed off to the trainer / scheduler / eval — your domain)

- Curriculum schedulers (`Random` / `Static` / `Adaptive`).
- Online generation inside a training loop (the generator is a callable; how
  it's invoked per-batch is the trainer's call).
- PARSeq trainer modifications.
- Eval on the labeled DarkBay set.
- The 4-model comparison.

The generator's API is designed so the trainer can call
`generator.render(t=scheduler.next(), n=batch_size)` per batch when that
phase begins, but building the trainer-side wiring isn't in this MVP.

## Files I'll create

```
src/
  auto_tune/
    difficulty.py          # EASY, HARD, interpolate(t)
  generator/
    parametric.py          # ParametricGenerator class
    render.py              # CLI: render --t --n --out
notebooks/
  05_smoke_difficulty.py   # build smoke_t0_05_10.png
mvp_results/
  smoke_t0_05_10.png       # visual smoke artifact
```

Reuses unchanged: `src/auto_tune/render_darkbay_v2.py` (the per-config
renderer), `src/auto_tune/renderer_config.py` (the dataclass),
`src/composite/compose.py` (the bg sampler + jitter), `data/darkbay_bg_lama/`
(bg pool), `configs/darkbay_renderer.json` (the hard anchor).

## Risks

- **Discrete-knob interpolation** (font set, box probability) is the only
  non-trivial bit. Plan: sort fonts by "complexity" (single mono → multi
  serif), include a font index in the active set when `t` exceeds its
  threshold; same trick for `box_prob`.
- **EASY config produces visually too-trivial samples** — no styling at all
  may look unrealistic. Acceptable for MVP — the trainer will skip very low
  `t` if needed, or Phase A2 of the full plan can tune `EASY` later.

## After this MVP

Hand the generator off. Next steps live in `plan.md`: trainer integration,
schedulers, then Phase B (refinement net) when needed.
