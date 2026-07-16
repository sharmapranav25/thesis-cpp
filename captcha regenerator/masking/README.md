# masking

Sibling of `char_extraction/`. Same problem (binary character mask from a CAPTCHA image), different approach: **gate first, mask second**.

Pipeline (the "earlier phases" of the prior denoising project, repurposed for pure mask production):

```
Stage 0: VLM colour probe         -> char_color in {red, blue, green, unknown}
Stage 2: if colour known -> HSV inRange + morphology         (Path A, cheap)
         else            -> DINO ViT-S/8 attention + Canny   (Path B, heavy)
Stage 2.5: coverage check; if low, re-run Stage 2 with looser params
```

Stages 3-5 of the source pipeline (SAM-edge intersection, amplification) are **out of scope** — they belonged to a denoising goal, not a masking goal.

## Quickstart

```powershell
py -3.9 -m venv .venv
.venv\Scripts\pip install -e .[dino,dev]

# Single image
.venv\Scripts\mask-extract --image path\to\captcha.png --type city --out tmp\
```

Stage 0 expects an Ollama server at `http://localhost:11434` with `minicpm-v` pulled. If unreachable, Stage 0 returns `unknown` and the pipeline falls through to Path B.

DINO ViT-S/8 weights load via `torch.hub` on first run.

## Output

```
out/<captcha_id>/
  original.png
  mask.png            # binary, same dims as input
  cleaned.png         # original with non-mask pixels zeroed
  overlay.png         # mask edges drawn on original
  attention.png       # raw DINO attention (Path B only)
  metadata.json
```

`metadata.json`:

```json
{
  "captcha_id": "...",
  "captcha_type": "...",
  "image_size": [W, H],
  "char_color": "blue",
  "path": "A",
  "stage_2_5": {"iterations": 0, "final_coverage": null},
  "method": "masking_v0"
}
```

Per-character bbox/crop splitting is **not** in this pipeline — it's a downstream concern.
