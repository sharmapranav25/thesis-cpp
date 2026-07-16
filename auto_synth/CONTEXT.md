# auto_synth — Working Context

> See `GOAL.md` for the project framing.

## Status

**Phase**: v1 complete. See `RESULTS.md` for the writeup, visual panels, and known issues.

**v0**: Parts A (alphabet discovery via CC + DINOv2 + KMeans), B (glyph bank with cluster→char mapping via easyOCR vote), C (LaMa bg extraction) all working end-to-end on 4 schemes (Courier Market, City, CryptBB unsolved, JoeyRetitling).

**v1 fixes** (per user feedback):
- **Bank inference** (`src/part_a/infer_bank.py`): observe a digit ⇒ assume 0-9; observe a letter ⇒ assume A-Z/a-z. Saves `inferred_bank` + `missing_glyphs` to manifest.
- **Length inference** (`src/part_a/infer_length.py`): histogram per-image CC counts (with `min_plausible_length=3` filter to discard segmentation failures). Compose samples lengths from this distribution.
- **Cluster collapse** (`GlyphBank.render_char(ch)` in `src/part_b/glyph_bank.py`): pick char first, cluster second — no over-sampling of stylization-rich chars.
- **Courier underline** (`src/part_c/extract_bgs.py`): long thin horizontal components added to LaMa mask. Partial fix; LaMa still leaves residue on curved underlines.

**Headline**: City synth is **visually nearly indistinguishable from real** at 4-5 chars. CryptBB+Joey are good visual matches. Courier remains weakest because the wavy underline is hard to remove cleanly.

**Not done**: training a recogniser on auto_synth output and eval'ing on labeled set. That's the gold-standard test, deferred to next session.

## What exists in adjacent projects (reuse candidates)

| Capability | Where | Status |
|---|---|---|
| LaMa-based bg extraction (CryptBB) | `synthetic image generation/data/cryptbb_bg_lama/` | 600 PNGs ready |
| LaMa-based bg extraction (DarkBay) | `synthetic image generation/data/darkbay_bg_lama/` | 1000 PNGs ready (eval contaminated) |
| LaMa extraction script | `synthetic image generation/src/extract/extract_bgs_lama.py` | Generic enough to run on Courier Market / City |
| Config-driven parametric renderer | `synthetic image generation/src/auto_tune/render_darkbay_v2.py` | Hardcoded CHARSET — needs Part-A output to be generalised |
| DINOv2 cosine search auto-tuner | `synthetic image generation/src/auto_tune/dinov2_search.py` | Hardcoded REAL_DIR / BG_DIR — needs CLI |
| PARSeq training (recogniser eval) | `model 1/src/parseq/`, `model 2/src/parseq/` | Working; bucket-curriculum support added in M2 v2 |
| CryptBB `case_decode.py` ensemble | `model 2/src/eval/case_decode.py` | Reusable for eval |
| Python venv | `model 1/.venv/` | Python 3.9.13, PyTorch 2.5.1+cu121, DINOv2/SAM dependencies need to be checked |

## What's NOT in adjacent projects (must build)

- **Part A — alphabet discovery** is novel. No prior code. Hardest part of this project.
- **Part B — font style learning** partially exists (parametric renderer with hardcoded CHARSET) but must be made alphabet-agnostic, conditioned on Part A's output.
- **Part C — bg/noise generation** has a working extraction (LaMa) but lacks a generative model — current pipeline samples real-extracted bgs directly. Could be upgraded to a learned generator.

## Datasets on disk

All under `E:/pranav/`. Counts include any non-image files; image counts may be
slightly lower.

| Scheme | Path | Files |
|---|---|---|
| Courier Market | `Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)/` | 2510 |
| City | `City (live)-20260503T045450Z-3-001/City (live)/` | 2000 |
| CryptBB unsolved | `CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023/` | 600 |
| CryptBB labeled (eval) | `model 1/data/eval/` | 651 |
| DarkBay | `DarkBay_Labeled-20260503T045615Z-3-001/DarkBay_Labeled/` | 1000 (labeled, eval-only) |
| JoeyRetitling | `JoeyRetitling_extracted/` | 651 |

## Compute

- Same machine: Windows 11, RTX A4000 (16 GB)
- Python venv: reuse `model 1/.venv/Scripts/python.exe`
- Will need: `segment-anything`, `torch`, `torchvision`, `dinov2` (or `transformers` with DINOv2), maybe `scikit-learn` for clustering

## Active questions / open threads

- **First target scheme**: leaning Courier Market (2510 unlabeled, no baseline pressure, biggest pool). Confirm before diving in.
- **Part A approach**: SAM segmentation + DINOv2 cluster vs pretrained text spotter as weak labeler. Trade-offs in `GOAL.md`. Tentative choice: SAM + DINO clustering, validated by re-rendering and measuring recogniser accuracy.
- **Part B approach**: per-character template averaging (cheapest) vs per-char generative model vs few-shot font generation (DG-Font / AGIS-Net). Start cheap, escalate only if needed.
- **Part C approach**: reuse LaMa extraction + sample at composition time (fastest). Generative model only if needed.

## File layout (planned)

```
auto_synth/
  GOAL.md             # vision, validation criteria
  CONTEXT.md          # this file — working state
  src/
    part_a/           # alphabet discovery
    part_b/           # font style
    part_c/           # bg/noise
    compose/          # ensemble + synth captcha generator
    eval/             # train recogniser on synth, eval on real
  data/
    courier_market/   # mirror or symlinks to parent dir's images
    cryptbb/
    ...
  notebooks/          # exploratory work, image dumps for visual checks
  configs/            # per-scheme JSON configs (alphabet, layout, etc.)
```

## What to do next (in order)

1. Survey the datasets visually — sample 10 images per scheme, check style consistency, character set, image dimensions. Choose first target.
2. **Part A v0** — wire up SAM + DINO + clustering on the chosen scheme's unlabeled images. Sanity-check output: do the cluster counts and centroids look like a plausible alphabet?
3. **Part C v0** — run existing LaMa extraction on the chosen scheme.
4. **Part B v0** — collect crops per character cluster, simple averaging or augmented sampling.
5. **Compose** — render synth captchas, manually visual-check vs real.
6. **Validate** — train a small PARSeq on synth, eval on the labeled set if available; otherwise visual A/B.

---

# Appendix — Per-file code reference (added during 2026-06 exploration crawl)

This appendix is a file-by-file map of `src/` so an agent (or future-you) can
navigate the code without re-reading it. The conceptual story is above and in
`RESULTS.md`; this is the "what each file actually does" index.

## Part A — alphabet / length / glyph discovery

There are **three discovery methods**, all emitting the same
`alphabet_manifest.json` + `alphabet/cluster_NNN/` layout so `GlyphBank`/compose
work unchanged downstream:

| Method | Entry | How it works | Best for |
|---|---|---|---|
| **CC + DINOv2 + KMeans** (primary) | `part_a/run.py` | segment → embed → cluster → (separate) OCR-map | hard schemes (CryptBB, Joey) — strong glyph crops |
| **OCR-direct** (v2) | `part_a/ocr_discover.py` | easyOCR reads whole captcha; alphabet/length/crops fall out of predictions+bboxes | weak schemes (City, Courier) — correct alphabet+length |
| **VLM** | `part_a/vlm_alphabet.py` | aggregates Qwen2.5-VL detections from `extract_bgs_vlm.py`'s jsonl | experimental |

- `part_a/segment.py` — CC segmentation. `crop_characters()` tries multiple Otsu
  binarizations (both polarities + morph-opening to break underlines), scores
  each candidate by how close its component count is to `expected_chars=(3,8)`,
  returns **mask-applied crops** (non-char pixels forced white so DINOv2 sees
  shape not bg). Filters: min_area 60, min_height_frac 0.30 (rejects underline
  fragments), max_width_frac 0.40. `_binary_candidates/_components/_filter_components`
  are reused by Part C.
- `part_a/embed.py` — `DinoEmbedder`: facebook/dinov2-small, CLS token, L2-normalized, D=384, batch 64.
- `part_a/cluster.py` — KMeans `sweep_k` (silhouette pick) + `cluster_with_k` + `filter_alphabet` (≥0.5% freq floor culls noise clusters).
- `part_a/map_clusters.py` — separate OCR-on-centroids step: easyOCR votes a char per cluster, writes `cluster_to_char`. `--allowlist '0123456789'` forces digit schemes. **Known miss:** digit "6" never recovered for Courier/City (mapped 9/10 digits).
- `part_a/infer_bank.py` — `expand_bank()`: saw a digit ⇒ assume full 0-9; saw upper ⇒ A-Z; lower ⇒ a-z. Flags `missing_glyphs` (in-bank chars with no cluster).
- `part_a/infer_length.py` — histograms per-image CC counts, `min_plausible_length=3` filter (segmentation merges/under-counts, never use 1-2 char images). Returns weighted length modes for compose sampling.
- `part_a/ocr_discover.py` — `_interp_char_bbox()` linearly slices easyOCR's word quad into per-char bboxes. **Caveat confirmed in logs:** on CryptBB/Joey easyOCR hallucinates 90+ junk chars (punctuation from noise) and chaotic 1-9 length spread (mean conf 0.44-0.48). Only trustworthy when `mean_ocr_conf > 0.5` (City/Courier).

## Part B — glyph bank

- `part_b/glyph_bank.py` — `GlyphBank` loads `cluster_NNN/` crops, serves
  `render_char(ch)` which is the **leakage fix**: pick char first, then a random
  cluster among that char's variants (`char_to_clusters`), so stylization-rich
  chars aren't over-sampled. `render(cid)` applies gentle jitter (color/size±15%/rot±6°).
  `to_alpha()` turns near-white→transparent for compositing. `only_mapped=True`
  drops noise clusters with no OCR-confirmed char. **No generative model** — it's
  a sampler over real extracted crops.

## Part C — background extraction (FOUR variants, escalating power)

All share: re-segment chars → build mask → LaMa inpaint. They differ in how the
mask is found. Only the simplest (`extract_bgs.py`, CC-based) feeds the final
synth pool (`bg_lama/`); the others were experiments to fix Courier's surviving
underline and are saved in parallel `bg_*` dirs for A/B (`make_bg_compare.py`).

| File | Mask source | Output dir | Notes |
|---|---|---|---|
| `extract_bgs.py` | CC chars + `_detect_underlines` (long-thin horizontals) | `bg_lama/` | **The one used.** 200 bgs/scheme, dilate 4px. |
| `extract_bgs_sam.py` | CC bbox → MobileSAM box-prompt mask | `bg_lama_sam/` | SAM captures anti-aliased edges/halos CC misses. Ran 200/scheme. |
| `extract_bgs_vlm.py` | Qwen2.5-VL-3B detects chars → SAM masks | `bg_lama_vlm/` | Also dumps `vlm_predictions.jsonl` (char identities → feeds `vlm_alphabet.py`). |
| `extract_bgs_grounded.py` | GroundingDINO text-prompt → SAM, **iterative** (re-clean until no boxes) | `bg_grounded/`, `bg_grounded_vlmprompt/` | Grounded-SAM. `--vlm-prompt` gets the prompt from Qwen-7B; `--vlm-oracle` uses Qwen-7B as clean/dirty early-exit. |

- `part_c/grounded_dino.py` — `GroundingDinoDetector` wrapper (IDEA-Research/grounding-dino-tiny).
- `part_c/vlm_qwen.py` — Qwen2.5-VL-**7B** helpers: `generate_scheme_prompt()` (describe chars for GroundingDINO) + `is_clean()` oracle. Lazy-loaded, `unload()` frees ~14GB.
- `setup.md` documents the Part C design rationale ("interactive object removal" = segmentation + inpainting; Grounded-SAM; where a VLM helps). **Conclusion reached there:** Option 1 (iterative grounded loop with early-exit oracle) is the recommended path; bigger SAM doesn't help because the residue is LaMa *hallucination*, not a mask problem.

## Compose + drivers + viz

- `compose.py` — `compose_one()`: sample bg (or grey fallback), sample chars uniformly from `observed_chars`, render glyphs to ~70% canvas height, lay out horizontally with even slots + y-jitter ±3px, squeeze if overflow. Labels = real chars. Naive layout (no wave/overlap).
- Drivers (per-scheme configs baked in): `run_all.py` (Part A + sample compose, force_k 18/20/50/50), `run_part_c.py` (LaMa bg, 200/scheme), `run_part_c_sam.py`, `run_part_c_vlm.py`, `run_ocr_discovery.py`, `run_mapping.py`, `run_v1.py` (length+bank infer + recompose), `run_final_compose.py` (2000-sample pools into `synth_pool/`), `run_vlm_alphabet.py`.
- Viz: `make_compare_panels.py` (real/CC-synth/OCR-synth rows), `make_method_compare.py`, `make_bg_compare.py` + `_4row.py` (compare the 4 Part C variants). Outputs in `notebooks/*_compare.png`.

## Per-scheme canvas + alphabet (baked into drivers)

| Scheme | Canvas | chars/cap | force_k | Final alphabet (OCR-mapped) |
|---|---|---|---|---|
| courier_market | 200×60 | 4 | 18 | `012345789` (digits, missing 6) |
| city | 324×81 | 5 | 20 | `012345789` (digits, missing 6) |
| cryptbb_unsolved | 200×60 | 5 | 50 | `0123456789CGJMTVWXbdfhnpx{` (26, one misread `{`) |
| joey_retitling | 200×60 | 5 | 50 | `01245689ACEIJMNPRUVWYdnp` (24) |

## The one thing still NOT done (gold-standard test)

Train a PARSeq recognizer on a `synth_pool/` and eval on a held-out labeled set
(CryptBB `model 1/data/eval/` is the natural target — head-to-head vs Model 2.5's
57.60%). Until that runs, "the synth is good" rests only on visual A/B panels.
Everything needed exists (`synth_pool/` pools of 2000 each — bump `n` to 10k+;
PARSeq trainer in `model 1`/`model 2`).
