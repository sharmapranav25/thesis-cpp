# captcha regenerator — CONTEXT

> Written during the 2026-06 exploration crawl. This is the **top-level map** of
> the `captcha regenerator/` directory. Each sub-project has its own README /
> research.md / past_context.md — this file ties them together and records what
> was actually built vs what the READMEs claim.

## What this directory is

An earlier, more *engineered* attempt at the same problem `auto_synth/` later
solved more holistically: **Phase 1 of a CAPTCHA solver = take a real CAPTCHA
image and produce a binary character-pixel mask + per-character crops + bboxes.**
Those outputs were meant to feed two downstream reconstruction tracks
(`bg_syth/` = background recreation, `Font_syth/` = font recreation), which were
never started (see below).

It contains **two parallel, independently-packaged attacks on Phase 1** plus
three empty placeholder dirs:

| Sub-dir | What it is | State |
|---|---|---|
| `char_extraction/` | Method-zoo + eval harness for char mask/bbox extraction | Built, evaluated, **all methods OUT_OF_SCOPE** |
| `masking/` | "Gate first, mask second" VLM-routed pipeline | Built, evolved past its README (now has CRAFT per-char stage) |
| `bg_syth/` | (downstream) background reconstruction track | **empty** — never started |
| `Font_syth/` | (downstream) font reconstruction track | **empty** — never started |
| `edgetest1/` | scratch/experiment | **empty** |

Shared constraints (both sub-projects): open-source models only, **label-free at
inference**, hand-labeled eval (50-100/type) drives a PASS verdict but never
enters training. Same 4 schemes as the rest of the thesis: DarkBay, City,
Courier, CryptBB.

## Sub-project 1 — `char_extraction/` (the method zoo + eval)

A clean Python package (`char-extract` CLI). Config-driven dispatch: a
`captcha_type` string resolves `configs/_default.yaml` + `configs/<type>.yaml`,
reads `method:`, instantiates the registered method class. Unseen type → falls
back to the universal baseline (`otsu_cc`).

**Architecture:** `api.extract()` → `router.route()` → a `BaseMethod` subclass
with `segment()` (→ mask) + `split()` (→ per-char bboxes), or an overridden
`run()`. Methods registered via `@register_method`.

**Methods implemented** (`src/char_extraction/methods/`):
- `otsu_cc.py` — Otsu threshold + connected components + geometry filters (area/AR/height/width). Zero deps beyond cv2/scipy. ~50 img/sec. The `_default.yaml` baseline.
- `universal_sam.py` — **spectral-residual saliency CCs + Otsu CCs → bbox-prompted MobileSAM (vit_t) → union mask → re-CC for per-char bboxes.** Has IoU/containment dedup helpers. ~1 img/sec on CPU.
- `swt.py` — Stroke Width Transform variant (eval reports exist: `*_swt.md`).
- `psstrnet.py` + `external/psstrnet.py` — wrapper for the pretrained PSSTRNet scene-text-removal net. **Tested and FAILED** (see below).
- `classical.py` — shared primitives (`to_gray`, `otsu_mask`, `detect_polarity`, morphology) used by the above.
- `postproc/components.py` — CC labeling, bbox extraction, filtering, L2R sorting.

**Eval harness:** `cli/eval.py` scores extracted bboxes vs **Claude-VLM
zero-shot pseudo-GT** (`eval/claude_predictions/`) — there's also a Qwen
baseline (`cli/qwen_baseline.py`, `eval/qwen_predictions/`). Metrics: count
accuracy, mean IoU, mask precision/recall. Per-type-per-method reports in
`eval/reports/{type}_{otsu,sam,swt}.md` + `summary.md`.

**HEADLINE RESULT (the thing to remember): every method is OUT_OF_SCOPE on
every type.** Verdict gate is IoU ≥0.7 + mask P/R ≥0.85; best methods land at
IoU 0.13–0.36. (n=10/type, pseudo-GT bbox precision only ~5-10px so absolute IoU
is a noisy lower bound — but method *ranking* is reliable.) Per-type best:
DarkBay/City/CryptBB count-accuracy is okay (80-90%) but bbox tightness fails
everywhere; Courier needs color-based segmentation (digits aren't always
darkest); SAM marginally best on CryptBB. **Conclusion reached in
`eval/reports/summary.md`: "universal-only is dead", per-type methods unavoidable.**

**`research.md` is the most valuable file here** — a scene-text-removal (STR)
literature survey (PSSTRNet, PERT, FETNet, Tang2021 stroke-synthetic, DiffSTR,
TMIM, OTR, EnsNet, EraseNet). Key recorded negative result (2026-05-06):
**pretrained PSSTRNet (both scut_syn + scut_enstext checkpoints) outputs
"everything is background" (mask.mean ≈ 0.999) on all 4 CAPTCHA types** — the
domain gap between natural-scene text and adversarial CAPTCHAs is *severe, not
partial*. Forward paths listed: Tang2021 synthetic-trained model; fine-tune on
synthetic CAPTCHAs; per-type classical+SAM methods. None chosen before the
project pivoted (to `auto_synth/`, effectively).

## Sub-project 2 — `masking/` (gate-first VLM-routed pipeline)

Sibling created 2026-05-06 (see `past_context.md` = a neutral snapshot of
`char_extraction/` taken at that moment). Same goal, different philosophy:
**ask a VLM what color the chars are, then pick a cheap or heavy mask path.**

Pipeline (`src/masking/`, orchestrated by `api.extract()`):
- **Stage 0** (`stage0.py`) — **Qwen2.5-VL** (note: README/reference say MiniCPM-V via Ollama; the *actual code* uses Qwen2.5-VL via transformers) returns structured JSON: `char_colors`, `noise_color`, `noise_types`, `char_style`. SHA256-cached. Falls back to `unknown` (→ Path B) if torch/transformers unavailable.
- **Stage 2** (`stage2.py`) — two paths chosen by color:
  - **Path A (color known)**: HSV `inRange` against a per-color table + morphology; achromatic colors (black/white/gray) use Otsu instead. `_fill_holes` turns outlined glyphs solid.
  - **Path B (color unknown)**: **DINO ViT-S/8 self-attention** (CLS→patch, keep sharpest-std half of heads, bilinear upsample) `0.7·attn + 0.3·Canny`, threshold at percentile, CC filter.
- **Stage 2.5** (`stage2_5.py`) — coverage validator: finds "hot" attention blobs, checks Stage-2 covered them; if <80% covered, re-runs Path B with progressively looser params (no re-running DINO — reuses `run_path_b_with_overrides`).
- **Stage 3** (`stage3.py`) — **CRAFT text detector + per-char crops.** ⚠ The README says "stages 3-5 out of scope / per-char splitting not in this pipeline" but `api.py` *does* wire Stage 3 (CRAFT polys, char crops, crop masks, kept/dropped stats). **The code outgrew the README.**

`reference_denoising_pipeline.md` documents the *original* 5-stage denoising
pipeline this was forked from (Stages 0-5, ending in SAM∩Canny dual-edge
agreement + 1.8× amplification for a VLM to read). The masking project kept
Stages 0/2/2.5 and re-added a CRAFT-based Stage 3. The unifying idea inherited:
**"two independent signals must agree before a pixel is kept as character."**

## How this relates to the rest of the thesis

- This was the **"reconstruct the captcha by extracting its parts" line of attack** that predates / parallels `auto_synth/`. `auto_synth/` Part A (segment) + Part C (LaMa bg extraction) are the *successors* — same idea (mask chars, extract bg, re-render) but integrated end-to-end and actually producing synth pools. The big lesson carried forward: **pretrained STR/OCR nets don't transfer to adversarial captchas; classical CC + SAM is the workable floor.**
- MobileSAM (`checkpoints/mobile_sam.pt`) and the Qwen2.5-VL char-detection idea both reappear in `auto_synth/src/part_c/`.
- The honest verdict here (every extraction method OUT_OF_SCOPE on hard schemes) is *why* `auto_synth/` leaned on visual-A/B validation and OCR-on-weak-schemes rather than precise per-char masks.

## Status / where it was left

Parked. No method reached PASS. `bg_syth`/`Font_syth`/`edgetest1` empty. The
effort effectively continued under `auto_synth/`. If resumed, `research.md`'s
three paths (synthetic-data fine-tuning of an STR net; per-type classical
methods; Tang2021 model) are the documented next steps.
