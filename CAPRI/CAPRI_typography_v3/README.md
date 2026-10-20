# Phase 2 broad typography revision

Phase2 now teaches recognition across varied character appearances on a family's measured background. It no longer uses the top-three font classifier ranks as its generation pool. This is a new experimental release; CAPRI_v2 and existing100k datasets remain intact.

## Implemented

- 35 independently sourced, licence-checked font files:12existing plus23new at pinned Google Fonts revision. Per-file font and OFL hashes in fonts.json. One requested source had no eligible TTF in the chosen directory and was excluded; see font_acquisition.json.
- Uniform shuffled font decks rather than confidence weighting. Actual accepted glyph/image exposure is recorded because retries and single-font words can cause finite-sample imbalance.
- Default50% mixed-font images and50% single-font images in expectation. Mixed examples contain at least two font identities. No target font assets.
- Per-glyph height0.80–1.18, width0.75–1.25, angle offsets up to15degrees, vertical jitter up to6% canvas height, occasional boldening and hollow effects. About20% gentler samples. All ranges are engineering choices, not validated optimal settings.
- Background rendering code retained. No learned font ranking, palette changes or background-parameter optimisation added. Contrast fallbacks remain part of the inherited renderer.
- Per-glyph visibility and clipping checks, text-hash partitions, replay checks, per-character font/geometry records, labels.csv and preview.
- Optional Meta-Sim refinement is off in this new typography experiment: old adapters were fitted under the old font policy. The chosen frozen Phase1 profile is used directly. This makes a clear Phase2 ablation and does not remove the old release's refinement option.

## Run

Requires the adjacent frozen CAPRI_v2 release (or --base PATH), its existing Python dependencies, and this directory including fonts. It does not train models or download anything at generation time.

```powershell
& 'E:/pranav/model 1/.venv/Scripts/python.exe' generate.py --family darkbay --count 210 --seed 732011 --partition train --out NEW_OUTPUT
```

Optional --lengths 2 3 4 5 6 7 8 and --mixed-fraction 0.5. An explicit length setting is not a verified inferred family distribution. Default lengths remain2–8 until the count research passes validation. Source profiles already contain CV AND frozen joint feature inference; no new real-image access is needed to render them. New profile extraction remains available in frozen CAPRI_v2.

See LENGTH_RESEARCH.md for the recommended character-instance/count experiment and the required training-data repair. This renderer still uses count-dependent fitting cells, so its output alone must not become the next count-model benchmark.

## Limitations

Font coverage and glyph visibility do not establish readable labels or OCR transfer. Some outline/thin samples remain difficult; assistant preview review is qualitative. Balanced broad fonts can help generic recognition but might also increase domain mismatch. Test broad typography and length-aware generation independently before combining their effects. Do not overwrite the in-progress recogniser or existing corpora. No recognition training is started by this update.

## Research attribution

Jaderberg, Simonyan, Vedaldi and Zisserman, *Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition* (2014), [paper](https://arxiv.org/abs/1406.2227), [full text](https://arxiv.org/html/1406.2227v4): synthetic text with diverse fonts and appearance variation motivates the expanded font bank. We do not reproduce its engine, scale, word vocabulary or reported accuracy. Per-character font mixing here is a project-specific augmentation; no claim that the cited paper validates this exact policy. The paper is motivation and attribution, not proof of DarkBay transfer.
