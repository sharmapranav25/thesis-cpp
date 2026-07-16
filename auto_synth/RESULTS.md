# auto_synth — overnight v0 + v1 fixes

**Date**: 2026-05-05 (overnight)
**Scope delivered**: Parts A, B, C all working end-to-end on 4 captcha schemes,
with real-character labels via OCR-on-centroids and real backgrounds via LaMa
inpainting. v1 fixes applied for the four issues you flagged:
1. **Bank inference** — observe a digit ⇒ assume 0-9; observe an upper letter ⇒ assume A-Z, etc.
2. **Length inference** — histogram CC counts per image, take modal length(s) and sample from them at compose time.
3. **Cluster collapse / leakage fix** — `GlyphBank.render_char(ch)` picks a char first then a random cluster within that char's variants, so chars with more stylization variants don't get over-sampled.
4. **Courier underline** — Part C now also masks "long thin horizontal" components so LaMa wipes underlines/strikethroughs.

## What was built

| Component | Path | Status |
|---|---|---|
| **Part A** segmentation | `src/part_a/segment.py` | ✓ CC on Otsu-thresholded image, multi-candidate (Otsu both polarities + morph opening), filter by size/aspect/min-height. Crops are background-masked so DINOv2 focuses on shape. |
| **Part A** embedding | `src/part_a/embed.py` | ✓ DINOv2-small CLS-token features (D=384). |
| **Part A** clustering | `src/part_a/cluster.py` | ✓ KMeans with k-sweep + silhouette pick, `--force-k` override. |
| **Part A** char mapping | `src/part_a/map_clusters.py` | ✓ easyOCR vote across cluster crops → cluster_id ↔ character. |
| **Part A** driver | `src/part_a/run.py` | ✓ scheme dir → alphabet manifest + per-cluster crops + centroid grid PNG + embeddings.npz |
| **Part B** glyph bank | `src/part_b/glyph_bank.py` | ✓ Loads per-cluster crops, exposes `render(cluster_id, seed)` with light color/size/rotation jitter + RGBA-with-alpha for compositing. Reads `cluster_to_char` mapping. |
| **Part C** bg extraction | `src/part_c/extract_bgs.py` | ✓ Re-segments characters per image, builds dilated mask, LaMa inpaints over it → clean bg. |
| **Compose** | `src/compose.py` | ✓ Picks N cluster IDs, renders glyphs from bank, composites onto sampled bg from `bg_lama/`, outputs synth.png + labels.csv with real chars. |
| Drivers | `src/run_all.py`, `src/run_part_c.py`, `src/run_mapping.py`, `src/run_final_compose.py` | ✓ End-to-end per-scheme runs. |

## v2 — OCR-based discovery (parallel path)

**Insight**: instead of CC-segment → DINOv2-cluster → OCR-on-centroids, we can
use a pretrained OCR (easyOCR) directly on the captcha images. The OCR returns
both the predicted text *and* the bounding box; from those we get for free:
- Char bank: union of chars across all predictions
- Length distribution: histogram of `len(predicted_text)` per image
- Per-char glyph crops: linearly interpolate the word bbox into per-char bboxes

**Built**: `src/part_a/ocr_discover.py` + `src/run_ocr_discovery.py` runs OCR
on each scheme's images and outputs an `alphabet_manifest.json` in the same
format as the CC pipeline, so downstream `GlyphBank` + `compose` work
unchanged.

**Output paths**: `data/<scheme>_ocr/` (separate from `data/<scheme>/` so the
two methods can be compared side-by-side in `notebooks/<scheme>_method_compare.png`).

**Caveat**: captchas are designed to defeat OCR. For weak schemes (City,
Courier, Joey), easyOCR will read them mostly correctly. For heavily
distorted CryptBB unsolved, OCR will fail or be unreliable — fall back to
CC+clustering for those.

### v2 results per scheme

| Scheme | Method | Alphabet found | Length modes | Mean OCR conf | Glyph quality |
|---|---|---|---|---|---|
| Courier Market | OCR | **all 10 digits 0-9** ✓ | 4: 65%, 5: 30%, 3: 6% — **modal 4 (correct)** | 0.698 | weak crops (small) |
| Courier Market | CC | 9 digits (missing 6) | 3: 49%, 4: 37%, 5: 14% (under-counts) | n/a | better crops, but bg residue |
| City | OCR | **all 10 digits 0-9** ✓ | 5: 44%, 6: 42%, 7: 14% — **modal 5 (correct)** | 0.726 | weak crops |
| City | CC | 9 digits (missing 6) | 3: 46%, 4: 37%, 5: 17% (under-counts) | n/a | strong crops + great visuals |
| CryptBB unsolved | OCR | (chars unstable; OCR conf low) | 1-9 spread, modal 7 (chaotic) | 0.477 | small/wrong crops |
| CryptBB unsolved | CC | 50 alphanum clusters | 4: 44%, 5: 33%, 3: 23% | n/a | strong crops |
| JoeyRetitling | OCR | (chars unstable) | 1-9 spread (chaotic) | 0.437 | weak |
| JoeyRetitling | CC | 50 alphanum clusters | 5: 41%, 4: 38%, 3: 20% | n/a | strong crops |

### Headline: hybrid approach is the v3 fix

- **Alphabet + length**: use OCR (better for weak schemes, gives correct alphabet AND length distribution).
- **Glyph crops**: use CC-based segmentation (better foreground extraction; OCR's bbox-interpolation gives crops that are slightly too small / sometimes mis-aligned).
- Per-scheme: pick OCR if `mean_ocr_conf > 0.5` else fall back to CC-based for both. CryptBB / Joey fall back; City / Courier use OCR.

### Method-comparison panels (3 rows: real / CC-synth / OCR-synth)

- `notebooks/courier_market_method_compare.png`
- `notebooks/city_method_compare.png` — **OCR-synth has correct chars but smaller; CC-synth still visually closer**
- `notebooks/cryptbb_unsolved_method_compare.png` — CC clearly wins
- `notebooks/joey_retitling_method_compare.png` — CC clearly wins

## v1 fix outcomes

| Issue | What v1 did | Effect | Residual problem |
|---|---|---|---|
| Bank inference | Detected char types in OCR mapping; expanded to full set | Courier+City alphabets now correctly assumed 0-9 (10 chars, 1 missing). CryptBB+Joey assumed full alphanumeric (~62 chars, ~37-38 missing). | Missing-glyph chars (e.g. digit "6") not rendered. Need TTF fallback or manual labeling pass. CryptBB has 1 OCR misread "{" still in mapping — clean by editing `cluster_to_char` in manifest. |
| Length inference | Histogram CC counts per image with `min_plausible_length=3` filter, sample modes at compose | Variable-length captchas now consistently 3-5 chars per scheme, matching real | Modes still slightly under-count in heavy-noise schemes (CryptBB true=5 but mode=4 with 33% at 5). Fine for v1; could hand-override per scheme. |
| Cluster collapse / leakage | `GlyphBank.render_char(ch)` picks char first, cluster second | Chars with multiple stylization clusters no longer over-sampled in synth. | None — fix is clean. |
| Courier underline | Long thin horizontal components added to LaMa mask | Mask now includes underline regions | LaMa still leaves residue on long-curving underlines (LaMa is better at small holes than thin paths). v2 fix: render a synthetic underline on the synth output to match real, instead of trying to clean the real one. |

## Per-scheme results

| Scheme | Source images | Crops | k | Cluster freq filter | Inferred alphabet | OCR-mapped chars |
|---|---|---|---|---|---|---|
| **Courier Market** | 2510 | 9090 | 18 | 18/18 above 0.5% | digits-only (forced) | `012345789` (9 unique) |
| **City** | 2000 | 5755 | 20 | 20/20 above 0.5% | digits-only (forced) | `012345789` (9 unique) |
| **CryptBB unsolved** | 600 | 2411 | 50 | 50/50 above 0.5% | unrestricted | `0123456789CGJMTVWXbdfhnpx{` (26 — incl. one OCR misread) |
| **JoeyRetitling** | 651 | 2670 | 50 | 50/50 above 0.5% | unrestricted | `01245689ACEIJMNPRUVWYdnp` (24) |

## Visual outputs (open these to verify quality)

### Centroid grids — discovered alphabets
- `data/courier_market/centroid_grid.png` — clean digits 5,9,4,3,2,1,6 + 5 noise clusters (wavy underline fragments)
- `data/city/centroid_grid.png` — **clean digits 0-9 across most cells**, very clear
- `data/cryptbb_unsolved/centroid_grid.png` — wide alphanumeric, multiple stylization variants per char (e.g. multiple "A" or "b" clusters)
- `data/joey_retitling/centroid_grid.png` — wide alphanumeric, similar stylization variants

### Synth samples (with real LaMa-extracted bgs)
- `notebooks/courier_market_synth_with_bg/` — 24 samples, partially right
- `notebooks/city_synth_with_bg/` — 24 samples, **closest to real**
- `notebooks/cryptbb_unsolved_synth_with_bg/` — 24 samples, colorful chars + cross-hatch bg
- `notebooks/joey_retitling_synth_with_bg/` — 24 samples, colorful chars + grid bg

### Final synth pools (2000 per scheme)
- `data/<scheme>/synth_pool/` — 2000 synth captchas + `labels.csv` with real-char labels (post-compose, post-mapping). `GlyphBank(only_mapped=True)` → only renders clusters that have an OCR-confirmed character, dropping noise clusters.

### Real vs synth comparison panels — quick visual A/B
- `notebooks/courier_market_compare.png` — **weakest match.** Synth digits look correct but Courier's wavy underline survives Part C's LaMa masking, so synth bgs have residual underline fragments.
- `notebooks/city_compare.png` — **strongest match.** Synth digits are nearly indistinguishable from real City captchas — same black-on-grey style, similar crossing lines, similar layout.
- `notebooks/cryptbb_unsolved_compare.png` — good match. Real has colorful chars on noise bg; synth has the same pattern with slightly cleaner glyph edges.
- `notebooks/joey_retitling_compare.png` — good match. Real has colored chars on grid bg; synth replicates the style.

## Things to check first

1. **`data/<scheme>/centroid_grid.png`** — does the grid show real characters or noise? If most cells are real chars, segmentation+clustering worked. Open all 4 grids.
2. **`notebooks/<scheme>_synth_with_bg/synth_*.png`** — visual A/B against real samples in source dirs. Synth should look stylistically like real to a casual eye.
3. **`data/<scheme>/alphabet_manifest.json`** — `cluster_to_char` field for inferred alphabet, `cluster_counts` for per-cluster frequency.

## Known v0 issues / paths to v1

| Issue | Symptom | v1 fix |
|---|---|---|
| **Cluster→char mapping misses chars** | Courier+City: only 9 of 10 digits recovered (missing "6"). CryptBB OCR misread one cluster as "{". | Higher OCR vote count; manual labeling pass on centroid grid as ground truth; allowlist-aware OCR (force charset). |
| **Multiple clusters per char** | CryptBB has multiple "A" or "b" clusters from stylization variation. | Either accept (treat as font variants), or post-process: cluster centroids by visual similarity again to merge. |
| **Some clusters are noise** | Courier has ~5 underline-fragment clusters. | Tighter min-height-frac filter; or filter clusters by silhouette/density after OCR mapping (a "?" OCR result = drop). |
| **Compose layout is naive** | Glyphs equally spaced, slight y-jitter only. Real captchas often have wave patterns, char overlap, etc. | Per-scheme layout config inferred from real-image char positions. |
| **No quality validation against held-out labels** | We don't yet train a recogniser on auto_synth output. | Generate larger pool (~10k), train PARSeq, eval on JoeyRetitling labeled set as smoke test. |

## What's NOT in this v0

- **No training of a recogniser on auto_synth output yet.** That's the gold-standard test of whether the synth is good enough. Requires: bigger synth pool (~10-50k), PARSeq training, real-eval comparison.
- **No A/B against hand-tuned synth** (CryptBB Model 2.5 used hand-tuned PHP renderer + M1.5-bucket curriculum hitting 57.60% real word). The fair head-to-head: train PARSeq on auto_synth's CryptBB output, eval on `model 1/data/eval/`, compare to 57.60%.
- **DarkBay scheme skipped** (eval contamination — see `synthetic image generation/CONTEXT.md`).

## Iteration plan when you wake up

In rough priority order:
1. **Eyeball the 4 centroid grids.** If quality is clearly good for a scheme, that scheme's pipeline is shippable. Move to step 4 for it.
2. **For schemes with bad centroids** (likely Courier — too many underline fragments): tighten min_height_frac in `segment.py`, re-run Part A, re-OCR.
3. **For chars missing from OCR mapping** (digit "6" in Courier+City): bump easyOCR `n_per_cluster` and lower its confidence floor; or do a quick manual labeling pass on the centroid grid → set `cluster_to_char` directly in JSON.
4. **Smoke-test with PARSeq.** Generate 10k+ synth via `src/run_final_compose.py` (bump `n` to 10000), train PARSeq for 5 epochs, eval on JoeyRetitling labeled set. If real-word > 20% the pipeline is on the right track.
5. **Compare to Model 2.5 baseline** for CryptBB scheme specifically. Train PARSeq on auto_synth's `cryptbb_unsolved` output, eval on `model 1/data/eval/` (the M2.5 eval set is the unsolved set's labeled twin). Target: get within 20pt of 39.94% (M2.5's PARSeq-alone real word).

## Compute notes

- All Parts A, B, C, mapping ran in ~30 minutes total on RTX A4000.
- Part A batched DINOv2 at batch_size=64, peak ~3 GB VRAM.
- LaMa runs at ~1.5 sec/image on GPU; capped at 200 bgs/scheme.
- easyOCR loaded English language pack (~80 MB) once.
