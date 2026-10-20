# Optional Phase 3: frozen DINO parameter adjustment 01

User explicitly dropped learned character count. Active default returns to Phase1 CV+frozen style features -> Phase2 fonts. This optional trial fixes five generic characters, not a predicted family length. Count model never loaded. No automation restart, no model training, no runtime VLM, no target-family generator/font assets.

## Method and controls

Frozen cached official Apache-2.0 DINO ViT-S/8, source/weights SHA256 and source snapshots recorded. Cached upstream commit unknown. Input224x72/ImageNet normalization. CLS embedding, mean-head CLS-to-patch attention, last normalized patch embeddings. Attention weights pool patch embeddings; spatial attention is pooled to3x10 occupancy. Family mean-descriptor objective = CLS cosine distance +0.5 attention-pooled embedding cosine distance +0.25 spatial attention L1. This original engineering objective is not a published validated CAPTCHA perceptual metric, DINO training objective, or full distribution distance. No head selected based on real results. An ablation selects with CLS alone from the identical candidate pool.

Fifteen predeclared candidates: unchanged baseline plus low/high settings for text height, spacing, glyph width, straight-line count/width, dot count and curve count. One control at a time; no combined optimization. Existing font identity/effect locked. Color palettes/gradient locked, exact baseline glyph RGB and per-kind operation colors reused, same synthetic text/seed and count. Geometry and coverage may change the resulting pixel histogram; color lock does not mean identical image pixels. Renderer caps/rounding may make a parameter ineffective. Nondefault glyph-width settings also use the renderer's existing alternate width cap; this is not a pure unconstrained width scaling experiment.

Search first4 existingfit references per family, report next4; Pymlex only2fit+2report.30fit and30report images; within-run pixel-hash separation asserted. All have prior development exposure; existing fonts/variants were selected earlier with broader exposed pools, so report images are NOT pristine held-out validation. No historical reserved evaluation accessed. Fresh CV each reference, reused pixel-hash-checked learned feature inference; fixed existingPhase2 fonts. Real images influence renderer parameter selection only, never classifier/DINO training. Real text labels never used. Candidate selection written before development scoring. Development results did not change selection.

## Results

| Family | Selected control | DINO objective reduction | Separate CV distance |
|---|---|---:|---|
| cryptbb | spacing_gain_1.2 | 1.89% | worse |
| city | curve_count_gain_0.5 | 4.83% | worse |
| courier | spacing_gain_0.8 | 4.02% | worse |
| darkbay | glyph_width_gain_1.2 | 0.04% | better |
| outlined_text_captcha | spacing_gain_0.8 | 0.52% | better |
| roncuevas | curve_count_gain_1.5 | 0.93% | better |
| pymlex | text_height_0.8 | 0.30% | worse |
| sloth | dot_count_gain_1.5 | 7.32% | better |

Mean DINO objective reduction: 2.55%. CV diagnostic (Canny edge fraction, grayscale standard deviation, horizontal intensity difference) improves4/8 and worsens4/8. It is not an independent learned perceptual metric or an established gate. The DINO objective is also the selection objective; improvement on it cannot establish visual correctness. Attention changes the selected control only for Courier (narrower spacing rather than increased lines). Seven other selections match CLS-only. Courier combined objective is lower for attention selection, but this does not prove general attention benefit.

Rendered30 final development images plus30 matched baselines, with same five characters per pair. Glyph identity/count/font/color assertions and renderer per-glyph visibility checks pass. Exact baseline color-locked replay asserted for60fit+report samples. One fitcandidate rejected: DarkBay increased lines creates an operation type with no baseline color to reuse. Initial assertion stopped the job after3 completed families; resumed with explicit invalid-candidate rejection, preserving completed results and failed script snapshot. No color constraint relaxed. No final fallback required. status seconds refers to resumed segment, not total trial wall time.

## Modest assistant visual review

Both full boards inspected. Changes are small: CryptBB spreads letters; Courier/outlined compress spacing; City reduces generic curves; DarkBay width change subtle; Roncuevas adds curves; Pymlex shrinks already mismatched pixel-like type; Sloth adds noise while retaining the incorrect hollow style. These do not resolve background/contrast, font, outline, or fixed-length mismatch. Increased Sloth dots score best but are not convincing visual improvement. No visual success gate or default promotion. Preserve Phase1+fonts baseline. Optional DINO is a useful experiment, not a proven distribution-matching solution.

Artifacts: bank_1.png and bank_2.png (reference/before/after), per-family panels,30after PNGs, before PNGs, results.json with recipes/evaluation splits, frozen per-family selection/trials, protocol.json, failures.json, source/license snapshots. Color locks constrain all candidates. Length module is disabled, with historical count artifacts retained.
