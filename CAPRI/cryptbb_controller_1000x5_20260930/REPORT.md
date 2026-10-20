# CryptBB controller: batch 1,000 × five epochs

**Run date:** 30 September 2026. This is the requested larger proof of concept. It is separate from the earlier 120-image pilot and does not alter the frozen CAPRI package. **Result:** the controller trained successfully, but the evidence does not show that its learned per-image choices outperform a simple fixed setting.

## What was trained

A small controller MLP received 15 scores and 128 pooled local features from CAPRI's frozen seed-731 CNN/Transformer hybrid, plus seven simple colour statistics. It predicted two renderer settings: number of background blocks (0–12) and shared block width fraction. The renderer supplied known synthetic settings as training targets. The hybrid checkpoint remained frozen.

The corpus contains **1,000 training**, **200 validation** and **200 test** synthetic images, rendered from CryptBB's 96 saved style profiles. Entire source styles were assigned to disjoint groups: 70 train, 13 validation and 13 test. Four rejected render attempts were retried within the same partition; 1,400 valid images were produced. Training used **batch size 1,000 for five epochs**, exactly one optimizer update per epoch. The best checkpoint was selected by synthetic validation loss at epoch 3. The validation loss changed from 0.196 after epoch 1 to 0.187 at epoch 3 and 0.188 after epoch 5.

On 200 held-out synthetic images:

| Quantity | Trained controller | Constant training-set mean |
|---|---:|---:|
| Block-count mean absolute error | 4.079 blocks | 4.139 blocks |
| Block-width-fraction error, positive cases | 0.02893 | 0.03000 |

The numerical improvement is small. Five full-batch epochs are only **five weight updates**. This run does not show reliable recovery of block count or width across held-out styles.

## CryptBB render check

The trained checkpoint was applied to 21 historically exposed real CryptBB references. It chose **4 blocks for 3 images, 5 for 14, and 6 for 4**; widths were approximately 0.12–0.14. Three render arms used the same saved source recipes, text, fonts and seeds: original median-control render, a prespecified fixed eight-block render, and the trained controller render. A post-hoc fourth arm used **five blocks of width 0.13**, selected after observing the controller's outputs. It is diagnostic, not an unbiased comparison.

| Arm | Images in which frozen hybrid detects blocks | Mean gap in seven accepted hybrid feature scores |
|---|---:|---:|
| Real references | 18/21 | — |
| Original median render | 2/21 | 0.255 |
| Fixed eight blocks, width 0.09 | 9/21 | 0.173 |
| Trained controller | 16/21 | 0.132 |
| **Post-hoc** fixed five wide blocks | **17/21** | **0.134** |

The trained controller closes much of the frozen hybrid's block-detection gap, but the post-hoc fixed setting matches or slightly exceeds it on that metric. This suggests that **wider, visible blocks** explain most of the gain. The image sheet confirms more rectangles, while target backgrounds still look denser and more varied. One shown render also changes a glyph's displayed colour through the renderer's contrast fallback when its background changes; the chosen text label stays the same. Minimum recorded glyph visibility across controller renders was 0.877. No OCR read-back was performed.

## Conclusion and limits

The expanded run is a valid training-and-test experiment, but **not evidence that a learned controller is necessary or better than a simple rule**. The synthetic held-out gain is marginal, and a comparable fixed width/count produces the same real-image diagnostic gain. The first useful implementation work is richer renderer controls for block colour, layering, texture and placement, followed by an independent test of whether those controls reproduce CryptBB's appearance. A larger controller could be tried after that capability exists.

The same frozen hybrid supplied controller inputs and scored the real/synthetic images; these are self-consistency measures, not verified real feature accuracy. Real images were already exposed during development. Only block count and width were learned; block colours and placement policy were fixed. No GAN, final Phase 2 broad-font generation or downstream recognizer accuracy was tested. The five-wide-block control was chosen after inspecting model predictions and must not be treated as a preregistered baseline.

## Artifacts

- `train_test.py`: synthetic generation, full-batch training, held-out testing and real-image rendering.
- `results.json`: exact split indices, all epoch losses, metrics and per-image controller choices.
- `controller.pt`: best synthetic-validation checkpoint and input standardisation.
- `posthoc_control.py`, `posthoc_ablation.json`: separate fixed-five-wide-block diagnostic.
- `comparison.png`: six real/original/fixed-eight/trained-controller rows. All 21 controller renders and post-hoc fixed renders are saved in this directory.

Reproduce from the thesis workspace with the recorded Python environment:

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B .\cryptbb_controller_1000x5_20260930\train_test.py
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B .\cryptbb_controller_1000x5_20260930\posthoc_control.py
```
