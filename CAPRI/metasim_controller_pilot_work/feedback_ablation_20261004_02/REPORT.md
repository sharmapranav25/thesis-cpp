# City controller feedback comparison

Both requested searches are complete: 100% DINO appearance feedback and 100% image-statistics feedback. Each used the same starting recipes, seed, 15 renderer controls, 1,536 search renders and readability rules as the earlier 60/40 experiment. The two new searches and shared evaluation took 143.5 seconds (2.39 minutes), excluding setup and report generation. Neural weights stayed frozen.

On 512 fresh City images, DINO-only reduced its DINO discrepancy by **27.47% versus the simple hybrid start**, while statistics discrepancy increased **27.95%**. Statistics-only reduced statistics discrepancy by **5.42% versus simple**. The previous 60/40 controller still has the lowest common score and independent appearance audit among these five settings in this run. This is a comparison of individual searches at a small fixed budget, not proof of the best feedback weights generally.

| Controller settings | DINO MMD² ↓ | Statistics MMD² ↓ | Common 60/40 score ↓ | Independent audit ↓ | Strict readability |
|---|---:|---:|---:|---:|---:|
| Simple hybrid start | 0.38317 | 0.52074 | 0.43820 | 2.8579 | 85.4% |
| Specialist start | 0.37100 | 0.53075 | 0.43490 | 2.8601 | 76.0% |
| 60% DINO + 40% stats | 0.35347 | 0.48121 | 0.40457 | 2.2911 | 77.1% |
| 100% DINO | 0.27792 | 0.66629 | 0.43327 | 2.4157 | 88.5% |
| 100% stats | 0.35802 | 0.49251 | 0.41182 | 2.3980 | 77.1% |

Compare values within each column. The common score applies the original 60/40 weighting to every result for reference; it was not used to select either new controller. The previous controller's numbers differ from the first report because this evaluation uses a different set of real images. All groups contain 96 unique images, 96 unique strings and 96 complete rendered texts, and all pass basic renderer usability. Strict readability is a synthetic glyph guard, not OCR accuracy.

![Feedback comparison](metrics.png)

The DINO controller passes strict readability on 85/96 images, compared with 82/96 for the simple start and 74/96 for both the 60/40 and statistics-only controllers. Its selected controls replace the initial font choices with **Reddit Mono**, reduce nominal font size to 0.802 of its initial value, add vertical jitter of 0.187 and flatten band slope to 0.200 of its initial value. Foreground and background colours are blended about 54% toward gray, and 23% of effects are forced to gray palettes. Statistics-only also selects Reddit Mono, but leaves band slope unchanged and applies little colour adjustment.

In the first 12 consecutive examples, DINO-only gives more consistent digit shapes and visibly staggered vertical placement, which looks closer to the displayed City references than the earlier range of ornate fonts. This is a visual judgment of these examples. It also produces gray backgrounds where City has white space, and retains some colour. At the feedback resize, nearly white pixels occupy 0.1% of DINO renders versus 59.0% in the real evaluation images. This brightness mismatch accompanies the worse statistics score; it does not establish which individual statistic caused the increase.

| Synthetic font measurement | Simple start | Specialist start | 60/40 controller | 100% DINO | 100% stats |
|---|---:|---:|---:|---:|---:|
| Mean image median weight (0–10) | 4.302 | 4.448 | 4.000 | 5.646 | 5.448 |
| Mean image median width (0–10) | 5.677 | 6.062 | 3.281 | 4.406 | 4.115 |
| Mean image median x height (0–10) | 5.594 | 5.323 | 5.323 | 5.000 | 5.000 |
| Mean image median slant (0–10) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Both new controllers change the glyph font keys in 90/96 paired images. The original 60/40 controller kept those keys unchanged. This makes DINO-only worth examining for typography, but neither DINO distance nor these synthetic font measurements establish real font accuracy. We have no labelled real font errors or downstream recognition scores from this trial.

![City references and controlled renders](comparison.jpg)

The fitting and selection subsets remain the original 256 and 128 City images. Both new policies use fresh independent population searches with identical first-round candidates, then choose among their own top three fitting candidates and neutral on the selection subset. The winner must stay within five percentage points of neutral usability and strict readability. The DINO winner ranked second on fitting; the statistics winner ranked third, with another candidate failing readability. Their saved choices were fixed before evaluation. The fresh 512-image evaluation subset is disjoint by pixel hash and source path from all 896 images used in the earlier pilot. The specialist start and all adjusted variants preserve the same generated strings and seeds; the simple baseline reuses its original outputs.

For DINO-only, the appearance objective is DINO kernel MMD²; for statistics-only, it is statistics kernel MMD². Each retains the original penalties of 0.8 times the unusable fraction and 0.15 times the strict-readability failure fraction. Both feature channels are computed for cross-metric reporting, but the unused channel has exactly zero weight in fitting and selection. Thus the statistics-only runtime includes DINO diagnostics and is not a CPU-only speed benchmark. Statistics normalisation and all kernel bandwidths are fixed from the original fitting subset. The independent grayscale and edge audit contributes to neither search.

With 200 paired bootstrap resamples, the DINO-only change relative to the specialist start has a 95% interval of -0.11167 to -0.07862 for DINO MMD², and 0.09919 to 0.17143 for statistics MMD². Its common score interval spans zero (-0.01872 to 0.01575). Statistics-only improves statistics MMD² with an interval of -0.06699 to -0.01043. These intervals reflect sampling variation for fixed controllers and one search seed; they do not measure repeatability across searches.

The original checkpoints, renderer files, previous selected settings and previous results are unchanged. The new policies are saved as experiment artifacts and were not installed as production defaults. All 480 saved PNG hashes and complete glyph labels were checked. The evidence is [results.json](results.json), [verification.json](verification.json), [font_details.json](font_details.json), [pixel_statistics.json](pixel_statistics.json), [selected controls](selected_all_controls.json) and [protocol.json](protocol.json). Reproduce both searches with `launch/feedback_ablation.py --parent <original pilot directory> --out <new output directory>` using the existing project Python environment.
