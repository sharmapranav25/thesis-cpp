# Setup — Interactive Object Removal (Part C background extraction)

A two-step pipeline with two well-known technical names.

## 1. Selecting the thing — interactive / salient object segmentation

Apple's specific implementation is called **"Lift Subject from Background"** (iOS 16+). Closest open research:

- **SAM** (Segment Anything Model, Meta 2023) — click/tap a point or draw a box, get a clean mask. Closest modern equivalent to Apple's tap-to-select.
- **U²-Net** — older, lighter, runs on-device. What Apple's was likely built on initially.
- **MaskCut / GroundingDINO + SAM** — text-prompted variants.

## 2. Removing it (filling the hole) — image inpainting

Modern models:

- **LaMa** (the same one we used for Part C) — fast, specialized.
- **MAT** (Mask-Aware Transformer).
- **Stable Diffusion Inpainting** — more flexible, much heavier.

The user-facing feature is sometimes called **"object removal"** or **"content-aware fill"** (Photoshop's old term). Google's "Magic Eraser" and Samsung's "Object Eraser" are productized versions.

The combination — **interactive segmentation + inpainting** — is the standard 2-stage pipeline in research literature, sometimes called **"interactive object removal"**.

## Grounded SAM (language-guided segmentation)

The text-prompted variant is called **language-guided segmentation** or **Grounded SAM** (the most popular implementation). It composes:

```
text prompt ("characters", "letters", "digits")
       │
       ▼
   GroundingDINO  →  bboxes for everything matching the prompt
       │
       ▼
      SAM  →  refined masks
       │
       ▼
     LaMa  →  clean bg
```

This is much smarter than the current "CC finds chars" approach because:

- **CC fails on anti-aliased edges, color halos, sub-pixel artifacts** — Grounded DINO doesn't.
- **CC includes noise blobs** — Grounded DINO understands "this is not a character".

## VLM-assisted enhancements

Two natural slots for a VLM in the pipeline. Use **Qwen2.5-VL-7B-Instruct** (open source, local, ~14 GB fp16) — same family as v3's Qwen-3B detector, no API key, no per-call cost. Both slots keep the VLM out of the inner GroundingDINO/SAM loop so latency stays bounded.

### Scheme-level prompt generation (one-shot, high leverage)

Generic prompts (`"characters, letters, digits"`) underutilize GroundingDINO. Instead:

1. Sample K crops from the unlabeled folder.
2. Hand them to Qwen with: *describe the character appearance precisely (color, weight, case, style, decorative noise)*.
3. Use that description as the GroundingDINO prompt.

One Qwen call per scheme — negligible cost. GroundingDINO sharpens significantly on tight descriptions like *"thin black serif uppercase letters overlaid on red curved noise lines"* vs. category nouns. After prompt-gen, Qwen can be unloaded if the oracle isn't used (frees ~14 GB).

### Early-exit oracle (per-image, sparse)

The Option 1 loop needs a stopping criterion: *"is the background clean?"*. Qwen is well-suited:

- Cheap heuristic (no GroundingDINO box above threshold) flags **candidate clean**.
- Qwen is invoked only on candidates: *"any character-like glyphs in this image?"*.
- Cache by perceptual hash to avoid re-asking on near-duplicates (TODO).

This keeps Qwen out of the inner loop while catching residue that thresholds miss. Memory note: oracle requires Qwen resident alongside GroundingDINO + MobileSAM + LaMa — fits an A4000 16 GB but with little headroom; drop to bnb int4 if needed.

### Skip

Per-bbox VLM filtering inside SAM. GroundingDINO's confidence + box threshold get most of the way for free.

## Three options analysis

Three options in increasing power/cost. **Recommendation: Option 1 with built-in early-exit** (option 1 + option 3 lite — best of both).

**Option 1 (loop)** — cheap, direct, and converges. Each iteration:
- Re-segment the current bg (SAM + box prompt).
- If no char-like components remain → early exit, this bg is clean.

**Option 2 (bigger SAM)** — the residue we're seeing isn't mostly a *mask* problem. It's that LaMa hallucinates char-shaped textures when the mask is wrong AND/OR the surrounding bg has patterns. Bigger SAM gives a slightly better mask, but doesn't address LaMa's hallucination. Marginal gain for 2.5 GB + 5× slower.

**Option 3 (discard)** — works but wasteful. We'd need to detect "dirtiness" anyway, which means running SAM/CC again on the output. Once we run the detector, why not also clean it? Discarding is the worst case of cleaning.

## Future direction

Option C is the future direction if we want one model for alphabet + length + bg extraction. But that's a bigger architectural shift better done after we've validated the current pipeline downstream.
