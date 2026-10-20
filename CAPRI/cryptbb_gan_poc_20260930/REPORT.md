# CryptBB frozen-hybrid feedback proof of concept

**Run:** 30 September 2026. **Purpose:** check whether the existing frozen hybrid can detect a real-to-rendered feature gap before building a GAN. No GAN was trained or evaluated.

## Method

The input was the existing `feature_joint_mapping_01/run_conformer/real_generation/cryptbb` matched set: 21 historically exposed CryptBB reference images, 21 renders with median controls, and 21 renders with learned controls. Each render pair had the same hybrid-predicted feature flags, reference, generated text, font and seed. The controls changed between arms. This is the historical Phase 1 renderer, not a fresh Phase 2 broad-font generation run.

`evaluate.py` verified the frozen seed-731 checkpoint SHA-256 against the saved experiment protocol. It resized all 63 images to RGB 192×96 with the original bilinear preprocessing and ran the `Joint("conformer")` checkpoint in evaluation mode on CPU. It used the original seed-731 synthetic-calibration thresholds and restricted the headline comparison to the seven features that passed the all-seed/all-stratum synthetic gates. The full machine-readable scores are in `results.json`.

For each group, prevalence is the number of images classified as containing the feature divided by 21. Requested prevalence comes from the hybrid's **own real-image flags**, so real and requested prevalence are identical by construction. Requested-versus-rendered agreement is a self-consistency measure, not real-label accuracy.

## Results

| Feature | Requested/real | Median render | Learned-control render |
|---|---:|---:|---:|
| Curved interference | 0/21 | 3/21 | 3/21 |
| Dot scatter | 8/21 | 3/21 | 4/21 |
| Periodic grid | 18/21 | 8/21 | 9/21 |
| Background texture | 21/21 | 16/21 | 15/21 |
| Background gradient | 6/21 | 4/21 | 4/21 |
| Background blocks | 18/21 | 2/21 | 2/21 |
| Blur | 15/21 | 21/21 | 21/21 |

Mean absolute prevalence gap across the seven features was **0.320** for median controls and **0.313** for learned controls. Mean absolute *probability-score* gap was **0.255** for median and **0.263** for learned. Thus one summary moved slightly toward real, the other slightly away. With 21 images, these are diagnostics, not a robust quality ranking.

The largest discrepancy is background blocks: the frozen model requests them on 18 real references but recognizes them on only two outputs in either arm. Grid and texture also have substantial gaps. The saved six-row comparison sheet visually shows coarse, noisy multicolour blocks and dense structure in several real examples, while most corresponding synthetic examples have fine grids or flat pastel backgrounds. This qualitative check supports a rendering/style mismatch; it is not an independent image-wide feature annotation.

## Interpretation

The **feedback signal is testable now**: frozen real predictions, rendered images and a second inference pass expose a measurable difference. Changing only the learned numeric controls in this saved comparison does not close the dominant CryptBB background-block gap. The next minimal intervention should change the block-rendering and background-layout controls, with exact glyph/label preservation and an independent visual or CV check. A GAN's added value would require a separate third arm and a held-out evaluation.

This experiment does **not** establish that the real-image flags are correct. The same hybrid generated the requests and graded the renders, so agreement can hide systematic errors and disagreement can reflect synthetic-domain bias. There are no audited real feature labels, no real transcription labels, no GAN output and no downstream recognizer evaluation in this proof of concept. These 21 references were already exposed during development; this is a diagnostic, not a pristine test.

## Reproduction

From the thesis workspace:

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B .\cryptbb_gan_poc_20260930\evaluate.py
```

The script reads the frozen E: artifacts and writes only `results.json` in this folder. The source paired gallery is `E:/pranav/Final sem/sourcefree_synth/feature_joint_mapping_01/run_conformer/real_generation/cryptbb/comparison.png`.
