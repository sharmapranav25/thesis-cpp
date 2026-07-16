# MVP API — synthetic captcha generator

Entry points for downstream code (recognizer trainers, eval harnesses,
curriculum schedulers).

## What it produces

200×50 RGB DarkBay-style captchas with labels and per-character bounding
boxes. One difficulty knob `t ∈ [0, 1]`:

- `t=0`: clean, single font (Cambria Bold), no rotation, no boxed chars,
  no jitter, plain bg sampling.
- `t=1`: full HARD config — 4 fonts mixed, rotation up to ±20°, ~60% of
  captchas have white-on-dark boxed style (per-image consistent), full
  position jitter, sampled real-extracted backgrounds.
- Linearly interpolates in between.

## Three generator classes

| Class | What it does | When to use |
|---|---|---|
| `ParametricGenerator` | Pure rules + sampling. LaMa-extracted bg pool (flat). No GPU. ~150 imgs/sec on CPU. | Always usable. Labels guaranteed correct. Default for low-to-mid `t`. |
| `RealBgGenerator` | At low `t`: same as Parametric. At `t >= bg_switch_t` (default 0.5): samples a **real captcha** as the bg substrate, locally inpaints (cv2.TELEA) just where our synth text will go, then composites text on top. | **Recommended for the high-difficulty / "looks like real" end.** Real wallpaper texture preserved. Some residual artifacts where the original-text mask missed. |
| `RefinedGenerator` | `Parametric` + a trained mask-aware U-Net pass on the high-`t` end. | Available, but `RealBgGenerator` produces visibly more realistic bgs without the trained component's failure modes (text drift). Keep for comparison. |

All three classes expose the same two methods (`render_sample(t)`, `render(t, n)`).

## Usage from this directory

```python
import sys
sys.path.insert(0, r"E:\pranav\synthetic image generation")

from src.generator import ParametricGenerator, RealBgGenerator, RefinedGenerator

gen = RealBgGenerator(seed=0)   # recommended for high-realism t=1
# or:  ParametricGenerator(seed=0)   # cheap, LaMa bg pool
# or:  RefinedGenerator(seed=0)      # parametric + trained U-Net

# one sample
img, label, bboxes = gen.render_sample(t=0.7)
# img    : PIL.Image RGB (200, 50)
# label  : str, 6 chars, "PR2GXW"-style
# bboxes : list of [x0, y0, x1, y1] per character

# batch
samples = gen.render(t=0.7, n=64)
# list of (img, label, bboxes) tuples
```

## Usage from another project (e.g. `model 1/`)

Add this directory to `sys.path` once at startup:

```python
import sys
sys.path.insert(0, r"E:\pranav\synthetic image generation")

from src.generator import ParametricGenerator
gen = ParametricGenerator(seed=0)
```

Or set `PYTHONPATH` once for the session:

```powershell
$env:PYTHONPATH = "E:/pranav/synthetic image generation;$env:PYTHONPATH"
```

Then `from src.generator import ...` works from any cwd.

## CLI for offline rendering

```powershell
"E:/pranav/model 1/.venv/Scripts/python.exe" `
    "E:/pranav/synthetic image generation/src/generator/render.py" `
    --t 0.7 --n 100000 --out E:/pranav/some/output/dir
```

Writes `0000000.png` … `<n>.png`, `labels.csv` (`filename,label`), and
`bboxes.jsonl` (`{filename, label, width, height, bboxes, t}`). Compatible
with the directory layout `model 1/`'s trainers expect.

## What "saved as MVP" means

The current state is the deliverable:

| Component | State | Where |
|---|---|---|
| Difficulty axis (EASY ↔ HARD interpolation) | done | `src/auto_tune/difficulty.py` |
| Parametric renderer (TTF + per-image styling) | done | `src/auto_tune/render_darkbay_v2.py`, `src/generator/parametric.py` |
| LaMa-extracted bg pool (1000 DarkBay bgs) | done | `data/darkbay_bg_lama/` |
| Refinement net checkpoint (v4 step 5000 = best) | done | `checkpoints/darkbay_refiner_v4_alpha/best.pt` |
| Public Python API | done | `src/generator/__init__.py` |
| CLI | done | `src/generator/render.py` |
| Smoke artifact (visual proof of difficulty axis) | done | `mvp_results/smoke_t0_05_10.png` |
| Comparison artifact (real / rough / refined) | done | `mvp_results/smoke_v4.png` |

Not in the MVP, but in `future.md` for later:
- FFG (few-shot font generation) — needed for schemes whose font isn't in
  Windows TTF pool.
- Bg-only refiner — alternative to the current full-image refiner; cleaner
  separation of concerns.
- `SchemeSpec` refactor — required to run on schemes other than DarkBay
  without code changes.

## Caveats to flag in any results

1. **Eval contamination on DarkBay.** The bg pool was extracted from the
   1000-image labeled DarkBay set (the only DarkBay we have). Recognizer
   eval on those same 1000 will be optimistic. Either source unlabeled
   DarkBay or report this in results.
2. **Refinement net is partial.** v4 adds *some* bg texture but doesn't
   fully match real DarkBay's wallpaper richness. Use `ParametricGenerator`
   if labels matter more than realism (which is the recognizer-training
   case).
3. **Difficulty knob is linear interpolation.** `t=0.5` is the arithmetic
   midpoint of the EASY and HARD configs; whether that's *perceptually* the
   midpoint of difficulty for a recognizer is untested. The curriculum
   scheduler can warp this if needed.
