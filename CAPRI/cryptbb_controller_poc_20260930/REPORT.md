# CryptBB block-controller training proof of concept

**Date:** 30 September 2026. **Result:** the small trainable controller did **not** outperform a fixed-setting control. Explicitly drawing additional blocks improved the frozen hybrid's self-consistency on this exposed CryptBB sample, but it did not establish realistic rendering or recognition benefit.

## Question and design

Can a small controller use the frozen CNN/Transformer hybrid's feature scores to choose better background-block settings for CryptBB?

The experiment kept the frozen seed-731 Conformer-style hybrid and historical Phase 1 procedural renderer. It added explicit block count and shared block width controls. A small MLP received the hybrid's 15 feature probabilities plus nine simple image colour/edge statistics and predicted two numbers: block count (0–12) and width fraction. Its training labels were known settings used to make controlled synthetic renders. No real CAPTCHA text labels were read.

The synthetic set contained 168 images: 21 source styles × four block counts (0, 4, 8, 12) × two widths (0.06, 0.12). Source styles 0–14 were training (120 images), 15–17 validation (24), and 18–20 locked test (24). All variants of a source style stayed in one partition. The controller used a two-hidden-layer MLP, AdamW, 160-epoch maximum and the lowest validation MSE checkpoint. The best checkpoint occurred at epoch 2. The test compared it with constant predictions equal to the training-set means.

For the real diagnostic, the trained controller inferred settings from 21 historically exposed CryptBB references. Using each existing median-render recipe, the code generated a controller render with predicted blocks. A separate **fixed-setting arm** drew eight blocks of width 0.09 for every example. Both arms used the same block colours, height rule, source text, font and render seed. Existing median renders were the baseline. The original renderer's separate block RNG kept its other drawing operations comparable, though altered backgrounds can still change text-colour fallback.

## Measured result

| Synthetic locked test, 24 images | Controller | Constant baseline |
|---|---:|---:|
| Block-count mean absolute error | 3.98 blocks | 4.00 blocks |
| Block-width-fraction mean absolute error | 0.0292 | 0.0300 |

Those differences are negligible. On the 21 real references, the controller predicted only **8 or 9 blocks** (13 and 8 images respectively), showing near-constant output rather than meaningful adaptation to each reference.

| Frozen-hybrid CryptBB diagnostic, 21 images per arm | Real references | Existing median renders | Fixed eight blocks | Trained controller |
|---|---:|---:|---:|---:|
| Background blocks detected | 18/21 | 2/21 | 9/21 | 9/21 |
| Mean score gap across seven passing feature heads | — | 0.255 | 0.173 | 0.165 |

The trained arm's 0.008 lower multifeature score gap than the fixed arm is small and does not rescue the failed synthetic parameter test. Its lowest recorded individual glyph visibility was 0.851, above the renderer's 0.55 minimum. The image sheet shows that both added-block arms contain more visible rectangles, but the real images still have denser, noisier, more varied backgrounds. The comparison does not show faithful CryptBB style reproduction.

## Interpretation and limits

This is a **negative result for the trained controller as implemented**. The improvement in block detection came from giving the renderer explicit block controls and drawing more blocks, because a simple constant setting achieved the same 9/21 result. The original mapping's block-presence flag alone often left the inherited profile's block evidence unsupported, so it did not ensure drawing blocks at all.

The frozen hybrid both contributed input scores and judged the new renders. Its CryptBB feature predictions have no independent real-feature labels here. The 21 real examples are historically exposed development images. Synthetic test styles came from the same small family of saved recipes. No GAN, final Phase 2 broad-font renderer, new source family or downstream OCR reader was evaluated. The result does not establish whether a larger controller, better spatial inputs, real adaptation objective or richer renderer would work.

The next meaningful small test is a **renderer-capability experiment**: vary block shape, colour, layering, density and placement while holding text/font/seed fixed; score with independent visual/CV checks as well as the frozen hybrid. Only train a controller after those settings can actually reproduce the desired CryptBB appearance. A larger PHP bank or GAN is not needed to explain this pilot's negative learning result.

## Files and reproduction

- `train_test.py`: deterministic generation, training, testing and paired rendering.
- `results.json`: metrics, per-reference controller choices and limitations.
- `controller.pt`: small fitted controller weights and training standardisation.
- `comparison.png`: six real/baseline/fixed/controller rows. All 21 fixed and controller renders are also saved.

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B .\cryptbb_controller_poc_20260930\train_test.py
```

The script reads frozen assets under `E:\pranav\Final sem\sourcefree_synth` and writes outputs only to this thesis-workspace directory.
