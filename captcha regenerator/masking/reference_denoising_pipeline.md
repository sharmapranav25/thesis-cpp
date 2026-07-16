
# Reference — denoising pipeline (Stages 0–5)

Source: prior denoising-only project, shared by the user on 2026-05-06. Recorded here as **reference**, not as a chosen approach. The user noted: *"this is from and only denoising approach but it might be useful here."* No commitment yet on whether/how it applies to `masking/`.

## Stage 0 — Colour detection (`stage0.py`)

- **Goal:** Identify the character colour so Stage 2 can pick the cheaper path.
- **Tool:** MiniCPM-V (local VLM via Ollama).
- **Method:** Chain of yes/no probes — *Can you see characters?* → *Are they all one colour?* → *Red family?* → *Blue family?* → *Green family?* Stops at the first "yes."
- **Output:** `char_color ∈ {red, blue, green, unknown}`. No image changes.

## Stage 2 — Character isolation (`stage2.py`)

First-pass mask separating characters from background. Two paths, picked from Stage 0:

### Path A — colour known (HSV)

- **Tool:** OpenCV `cvtColor` + `inRange` against a per-colour table (e.g. blue: H 75–130, S ≥ 50, V ≥ 50).
- Morphological close + open to clean.
- Fast and exact when colour is reliable.

### Path B — colour unknown (DINO)

1. Pad to multiple of 8, normalise, push through DINO ViT-S/8 (self-supervised ViT, no fine-tuning).
2. Extract `[CLS] → patch` attention from the last block: shape `[heads, n_h, n_w]`.
3. Filter heads by std — keep the sharpest half (averaging all heads dilutes the signal).
4. Bilinear upsample to pixel resolution → `attn_up`.
5. Compute Canny edges, dilate with 9×9 ellipse → `edges_d`.
6. Combine: `score = 0.7 · attn + 0.3 · edges`.
7. Threshold at the 60th percentile (or Otsu).
8. Filter connected components: drop < 60 px (noise) and > 20 % of image (background bled in).

**Output:** cleaned image (foreground pixels kept, background zeroed) + raw attention map for Stage 2.5.

## Stage 2.5 — Coverage validator (`stage2_5.py`)

- **Goal:** Catch the case where DINO saw a character but the threshold cut it.
- **Method:** Find "hot" blobs in the attention map (≥ 0.55, ≥ 50 px). For each blob, check whether Stage 2 kept any pixels there. If < 80 % of blobs are covered, re-run Stage 2 with progressively looser params:

  | iteration | `noise_pctile` | `max_area_pct` |
  |-----------|----------------|----------------|
  | 1         | 50             | 0.30           |
  | 2         | 40             | 0.40           |
  | 3         | 30             | 0.50           |

- Stops as soon as coverage hits 80 % — or accepts the best result if all retries fail.
- Path A skips this stage (no attention map exists).

## Stage 3 — Dual edge detection (`stage3.py`)

- **Goal:** Compute structural edge maps from two independent images so Stage 4 can intersect them.
- **Tools:** SAM ViT-B + Canny.
- **Method:**
  - Run SAM `AutomaticMaskGenerator` (IoU ≥ 0.80, stability ≥ 0.85, min area 100). For every segment, draw its boundary at thickness 2 → normalise to `[0, 1]`.
  - Run Canny (50, 150) → normalise.
  - Combine: `edge = 0.8 · SAM + 0.2 · Canny`.
- Run **twice per variant:**
  - on the Stage 2 cleaned image → `edge_map_clean`
  - on the raw original image → `edge_map_original`

## Stage 4 — Common-edge intersection (`stage4.py`)

- **Goal:** Keep only edges that exist in both sources — those are real strokes.
- **Method:**

  ```
  intersection = edge_clean * edge_original   # element-wise multiply, [0, 1]
  mask         = intersection >= 0.25
  mask         = dilate(mask, 5x5 ellipse, 1 iter)   # fill stroke interior
  ```

- Edges from Stage 2 cleaning artefacts vanish (the original doesn't have them).
- Background texture vanishes (the cleaned image doesn't have it).
- The result is applied to the variant image to keep only the traced characters.

## Stage 5 — Amplification on original (`stage5.py`)

- **Goal:** Produce the final image the VLM will read.
- **Method:**
  1. Recompute the same intersection mask, dilate harder (4 px × 2) so it covers full stroke width.
  2. On the untouched original:
     - Character pixels → multiply by 1.8× (capped at 255).
     - Everything else → pure white.
- **Key property:** Character appearance always comes from the original — none of the cleaning artefacts from Stage 2 leak into the final image. The mask only decides *where* the characters are; the original decides *how* they look.

## Unifying idea (as stated in the source)

> Two independent signals must agree before a pixel is kept as character.

## Notes on relevance to `masking/`

- This was a **denoising** pipeline — final output is an amplified image for a VLM to read, not a binary character mask + per-character crops.
- Several intermediate artefacts in Stages 2–4 *are* binary masks, so the masking ideas (HSV path, DINO attention path, dual-edge agreement) may transfer.
- Whether and how this informs the `masking/` work is **not yet decided** by the user.
