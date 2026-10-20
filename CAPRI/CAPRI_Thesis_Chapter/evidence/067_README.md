# CAPRI v2 — similar-font variety + Meta-Sim-inspired refinement

Default generation now samples one font per image from the top3 frozen font-model scores, weighted by those scores, and applies the independently fitted family refinement adapter when available. This is a new generation policy, not classifier retraining. Phase1 remains CV+frozen joint inference; Phase2 retains font adjustment/effects while varying font identity. Text lengths2–8 and numeric/alphabetic/alphanumeric buckets remain balanced by21-cell cycle.

## Use

```powershell
python capri.py verify
python capri.py generate --family cryptbb --kind conformer --count 210 --seed 731 --partition train --out output/cryptbb --recipes --masks
```

Defaults: --font-mode similar --font-top-k 3 --refinement auto. Existing five hybrid profiles have matching adapters: cryptbb, city, courier, darkbay, outlined_text_captcha. Single-font baseline: --font-mode single --refinement off. Font-only ablation: --refinement off. Require refinement explicitly: --refinement on (fails if no matching adapter). All flags and actual font IDs/families/weights/refinement gains/fallbacks are recorded. One font is chosen per image, never arbitrary per-character mixing. Fonts outside the selected pool are never sampled, including diverse mode.

## New families

```powershell
python capri.py profile --references references.json --name new_family --kind conformer --variant 3 --device cpu --out output/new_profile
python capri.py generate --profile output/new_profile/profile.json --kind conformer --count 210 --seed 42 --partition train --out output/new_images --recipes
```

references.json: {"partition":"style_fit","images":["path/to/image.png", ...]}. This runs CV AND frozen inference. Font variety works immediately. Refinement does NOT transfer adapters from another family: auto prints off for an unmatched profile; on errors. A changed/custom profile also fails exact adapter binding. Gridded generation supports font variety but has no fitted adapter. Fitting new adapters is a separate research workflow, not performed silently by generate.

## What refinement changes

Linecount/width,dotcount,curvecount/amplitude,measured blockcount. It renders the chosen font baseline first, then locks its exact glyph RGB/geometry while applying learned recipe gains. Invalid visibility candidates revert to that baseline with an explicit fallback record. Palettes, gradients, fonts, text and layout are not learned by this adapter. No DINO/VLM is loaded for generation; DINO was used only during adapter fitting. Runtime is CPU-capable.

## Evidence and limits

Similar-font selection is a top3 probability-ranking heuristic, not verified font similarity or guaranteed OCR benefit. Probabilities are not target-domain calibrated. Meta-Sim-inspired12step adapters were fitted using the SINGLE-font baseline; applying them with varied fonts is a new combination. Smoke validation establishes operation and invariants, not real OCR gains. DINO and independentCV improved4/5families in the earlier pilot, CryptBB had mixed results. Do not claim full Meta-Sim reproduction, exact target matching, or a production quality gate. All existing real families have historical development exposure. Read docs/METASIM_PLAN.md and docs/METASIM_RESULTS.md.

## Handoff

Read NEXT_SESSION.md. Frozen seed731 models/base/font assets and licenses are included. Original package preserved. requirements.txt lists tested dependencies, not bundled Python. Legacy reproduction sources retain original machine paths and are reference-only. Base integrity manifest remains intact; outer manifest covers the new snapshot. Outputs must be written to fresh directories. Do not change this package in place; create a new version for further experiments.
