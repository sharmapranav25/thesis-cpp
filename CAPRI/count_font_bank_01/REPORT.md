# Phase 1 count + Phase 2 fonts: all-family preview 01

User requested testing through Phase2 across the existing bank. Completed32 final renders,4 per8sources, plus32 Phase1 renders. No retraining or optional refinement/DINO. Preserved historical experiments and vendor files. Automation remains paused.

Sequence: Phase1 fresh CV with pixel-hash-matched cached frozen learned feature predictions, plus fresh frozen countCNN inference; Phase2 fresh frozen font inference. Count aggregated over212 existing fit/reference images; font inference over up to16 per family. These are exposed development references, not independent validation. No real labels used. Existing selected rendering variants retained. Generic uppercase/digit generation alphabet is a fixed choice, NOT an inferred character type; it explains letters in numeric-looking families.

Length policy declared in script before rendering: mean calibrated synthetic-model posterior per family; four deterministic stratified quantiles(.125,.375,.625,.875). This explores model uncertainty and is NOT a proven family length distribution, calibrated real margin, or guaranteed +/-1 interval. Supported lengths2–8 only. No manual family-specific count correction. Font and count checkpoint hashes verified unchanged. Original three learned feature models hash-checked for original5; cached pixel-matched inference reused across all8. Fresh CV and font inference performed for every displayed source.

| Family | Four generated lengths |
|---|---|
| cryptbb | 4, 5, 5, 7 |
| city | 5, 5, 5, 5 |
| courier | 6, 6, 8, 8 |
| darkbay | 5, 6, 6, 7 |
| outlined_text_captcha | 6, 6, 6, 8 |
| roncuevas | 4, 6, 7, 7 |
| pymlex | 4, 4, 4, 4 |
| sloth | 6, 6, 7, 8 |

Validation:32/32 requested text lengths equal renderer glyph records; all glyph contributions>=0.55 and built-in bounds checks pass. For each paired Phase1/Phase2 image, background-operation records match exactly. These check rendering integrity, not style or count correctness.

Assistant visual review of both full comparison boards: City/Pymlex counts plausible; Courier and Roncuevas visibly overcount; outlined usually overcounted too. CryptBB/Sloth length variation may reflect model errors rather than real variation. DarkBay retains broad palette/block resemblance but count variations remain unverified. Font matching still selects pixel-like fonts for outlined/Pymlex; outlined effect remains incorrect and Sloth hollow styling weak. Some foreground contrast and background/noise mismatches persist. No quality gate pass, real accuracy claim or promotion as a reliable count system. This is a requested integration preview.

Artifacts: all_families_final.png, bank_1.png, bank_2.png, per-family comparison images, individual Phase1/final PNGs and masks, results.json with text/recipes, count_inference.json with212 probabilities/pixel hashes, protocol.json with checkpoint hashes, validation.json. Source entrypoint preserved in run with count model dependency. No new paper adopted; existing count/font/feature methods only.
