# CAPRI — CAPTCHA Analysis and Procedural Rendering Inference

Research snapshot: seed731, CV-preserving fusion, 2026-09-28. Experimental, NOT a production promotion or proven real-domain improvement. Both architecture weights are frozen. This package never updates from ongoing training automatically.

## Start here

Use Python with requirements.txt (existing tested environment on original host: E:/pranav/model 1/.venv/Scripts/python.exe). Dependencies are not bundled. From this directory:

```powershell
python capri.py verify
python capri.py generate --family cryptbb --kind conformer --count 21 --seed 731 --partition train --out output/cryptbb_smoke --recipes --masks
python capri.py generate --family darkbay --kind gridded --count 210 --seed 732 --partition train --out output/darkbay_grid --recipes
```

Families: cryptbb, city, courier, darkbay, outlined_text_captcha. Output folders must not exist. Generated labels are known chosen text, never recognized reference text. Counts cycle evenly over 21 cells: lengths2–8 × numeric/alphabetic/alphanumeric. Use multiples of21 for exact balance. No real length or alphabet inference. Font handling is existing Phase2, not new joint-model learning.

## New family: CV AND frozen inference

Create references.json with {"partition":"style_fit","images":["absolute/path/image.png", ...]}. Relative paths resolve against that manifest. Do not include reserved evaluation images or labels.

```powershell
python capri.py profile --references references.json --name new_family --kind conformer --variant 3 --device cpu --out output/new_profile
python capri.py generate --profile output/new_profile/profile.json --kind conformer --count 210 --seed 42 --partition train --out output/new_images --recipes
python capri.py audit --corpora output/new_images
```

Profile runs CV, legacy frozen block inference, frozen font inference, then CAPRI joint inference and fusion. CUDA optional. No VLM, training, downloads or target generator. Cached profiles generate without original real files. Fresh profiling needs user-supplied real images. Old absolute reference paths in cached provenance are historical metadata, not generation dependencies.

## What is merged

CV retains palettes, gradients, measured blocks, line/dot density, layout and font settings. V2 curve presence/count/width are applied; only hybrid uses predicted curve amplitude because only it beat median baseline in every seed731 test stratum. Learned negatives cannot erase measured blocks/texture. Grid and blur additions are deferred without CV corroboration. Other predictions are retained for inspection, not silently applied. This is explicit mapping fusion, not retraining or distillation. Shared rules for all families.

Independent line/curve/dot/glyph random streams prevent count-driven downstream random drift. Readability fallback may still change glyph colours; none changed in the matched fusion trial. Background geometry remains approximate and curve placement generic.

## Evidence and limitations

Seed731: both models passed7/15 feature gates across four synthetic test strata. Grid6/9 and hybrid7/9 numeric controls beat training medians. Each inferred6909 already-exposed real images. Fusion trial rendered420 images total. These are synthetic gates, not measured real accuracy. CryptBB density improves but blocks/texture remain approximate; Darkbay loses some blocks; CV introduces unwanted dots. No overall real-quality gate pass. Pure V2 and fusion are separate experiments.

Real results were viewed before mapping fusion; therefore this is development data. Remaining training seeds must retain their precommitted protocol; never tune on locked tests and call them unseen. Do not claim VLM-free contextual inference, character count inference, Meta-Sim, GAN refinement, or PARSeq training are implemented here.

## Package map

- capri.py: portable verify/profile/generate/audit CLI.
- checkpoints/: seed731 weights, calibration thresholds and actual synthetic evaluation.
- profiles/: fused cached family profiles for each architecture.
- model_code/: original architecture and numeric decoding.
- renderer.py: frozen fused renderer.
- base/: exact Phase1/2 release, assets, fonts/licenses and its own integrity manifest.
- galleries/: real/oldV2/measured/fused comparisons, not new unseen tests.
- docs/: research attribution, plan and validation record.
- reproduction/: historical training/evaluation sources; these retain original E: paths and are reference-only, NOT portable training entrypoints.
- INTEGRITY.json: immutable package file hashes; outputs are intentionally excluded.

No dataset images or synthetic training corpus are bundled. Galleries contain historical reference examples; source provenance/rights remain in base/PROVENANCE.json and original research records. Retain third-party font licenses; this bundle does not relicense third-party assets. Review dataset redistribution rights before public release.

Next session: verify, inspect galleries/limits, generate a21-image smoke batch, then use an explicit style_fit/evaluation split for new data. Do not overwrite this snapshot or assume current running jobs finished. Original training monitor paths are in NEXT_SESSION.md.
