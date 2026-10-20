# Full-family two-stage stroke inference

Completed 2026-09-28 at user's explicit request despite failed synthetic diagnostic gate. All6909 paths from the full dot manifest scored, no decoding failures. Frozen original detector and shape checkpoints; no real fitting, no threshold tuning, no runtime VLM. CPU inference. Per-image JSONL, model hashes, source hash and protocol retained. Whole-dataset predictions include evaluation-directory exposure already authorized for dot inference; these are not untouched evaluations for future development.

| Family | Images | Absent | Straight | Curved |
|---|---:|---:|---:|---:|
| CryptBB |600|74|206|320|
| City |2000|2000|0|0|
| Courier |2510|19|68|2423|
| Darkbay |999|1|7|991|
| Outlined text |800|0|340|460|

These are predicted distributions, not accuracy. The model outputs one shape for the entire mask; it does not separate multiple strokes or represent both straight and curved interference within one image.

Assistant visually reviewed six evenly spaced manifest examples per family, with predicted masks alongside resized inputs. Review is modest qualitative evidence, not exhaustive annotation of6909 images:

- CryptBB: masks generally isolate glyph fragments/blobs and fail to follow the fine interference strokes.
- City: six inspected masks are empty despite visible broad pale bands. This is a thin-stroke training scope mismatch if bands are expected as interference lines; no claim that all2000 negatives are correct.
- Courier: masks include rough glyph/stroke fragments. Some interference is captured, but shape output cannot confidently be attributed to the intended stroke.
- Darkbay: backgrounds and letter regions are selected as interference. All six reviewed masks show substantial contamination; curved predictions on991/999 are not evidence of a curved-interference family.
- Outlined text: masks select large portions of dark gradient background rather than isolating the obvious black interference stroke. Apparent family-level presence success is misleading.

Conclusion: the authorized full-dataset exploratory test is complete. The two-stage model does not yet reliably transfer to these families. Review suggests brightness/palette and text-versus-stroke separation are major failures, consistent with training on bright plain backgrounds; causal ablations remain unperformed. Preserve failed gates and keep production unchanged. The next development trial should expand synthetic background/polarity hard negatives and verify masks before interpreting shape predictions. No new paper used.

Script: ../full_real_strokes.py. Evidence: summary.json, protocol.json, manifest.json, per-family JSONL, *_masks.png. Old checkpoints preserved. Do not label these full-dataset outputs as ground truth or use their agreement as an accuracy gate.
