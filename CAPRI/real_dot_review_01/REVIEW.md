# Dot scatter: real-image assistant visual review

2026-09-28. User requested real-data inference and assistant VLM review. No VLM is part of the executable pipeline. Thirty previously exposed fit/development examples: six from each Tier1 family, chosen deterministically by path hash without consulting scores. Reserved validation records were not used. This is a small qualitative audit, not an unbiased real-domain accuracy estimate.

Frozen checkpoint: completed screen_5x3_pop, copied and SHA256-recorded before inference. Final view search/confirmation still running. RGB images resized to128x64 with bilinear interpolation, then classifier's own morphology pop applied. Presence threshold0.5 is provisional; no real calibration or fitting. Scores are not validated real-domain probabilities.

Assistant inspected five original-image contact sheets without scores before reading predictions. Target definition is compact non-text round scatter (synthetic radius1..3), not arbitrary grain or textured strokes.

| Family | Predicted present / 6 | Visual finding |
|---|---:|---|
| City | 2 | IDs5,6 appear false positive: bands and glyphs, no visible round scatter. Other4 correctly absent. |
| Darkbay | 0 | All6 appear correctly absent: background rectangles/bands and glyphs. |
| Outlined text | 0 | All6 appear correctly absent: outline glyphs, gradient, interference stroke. |
| Courier | 2 | IDs2,5 are likely false positive: rough/textured strokes rather than distinct scattered disks. Other4 absent. This distinction is less certain than City. |
| CryptBB | 3 | IDs1,3,6 positive. Fine speckles mixed with grids/curves/blocks are ambiguous under the narrow disk-scatter definition. All6 excluded from binary visual accuracy claims. |

On18 clear negative examples (City/Darkbay/outlined), two false positives at the provisional threshold. Including the six less certain Courier negatives would give four false positives among24. These are small, selected development cohorts and assistant judgments, not expert ground truth. No unambiguous positive cohort was established, so real recall is unmeasured. Do not interpret the remaining negatives as proof of robust detection.

Conclusion: encouraging synthetic selection AP did not establish real reliability. City errors show transfer failure even on obvious negatives; Courier suggests texture confusion. Resizing may affect compact-mark scale, but this was not causally tested. Keep model experimental. Next work should add synthetic hard negatives resembling bands, rough strokes and glyph fragments, plus varied resolution; independently audit positive real examples. Do not fit on these real images or lower the acceptance gates. These reviewed examples are development evidence for future revisions.

Artifacts: predictions.json retains each source path, score and decision; protocol.json records checkpoint hash and preprocessing; *_blind.png retain reviewed sheets. No external paper used in this audit.
