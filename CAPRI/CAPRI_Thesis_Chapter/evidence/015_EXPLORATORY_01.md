# Synthetic-only exploratory training 01

**OBSERVED:** Training completed in 1,716.9 seconds (28.6 minutes), including descriptor extraction for 84,907 crops. User explicitly authorized exploratory synthetic training while the original full-corpus and audited five-dataset gates remain incomplete. Real images and annotations were not used. Historical corpus records remain unchanged and do not become accepted training data by this experiment.

[Results and configurations](../../runs/feature_bank_exploratory_01/results.json) · [Run protocol](../../runs/feature_bank_exploratory_01/protocol.json) · [Completion record](../../runs/feature_bank_exploratory_01/status.json) · [Feature-pop comparison](../../runs/feature_bank_feature_pop_01/comparison.png).

## What was trained

One binary Logistic Regression and one Extra Trees family per feature. Each feature used a seeded sample of 6,000 positive and 6,000 negative training crops; uncertain labels were excluded. Related base groups remain in their original partitions, and retained crop hashes are unique. Distinct crops are not independent observations: overlapping crops and variants share base groups.

Image-only descriptors comprise native color/contrast/connectivity measurements and orientation-pooled HOG on a proportionally padded 256x96 canvas. HOG blocks touching padding are excluded. The enhanced variant adds CLAHE, morphological, threshold and feature-pop measurements. Masks, recipes and generator IDs never enter the descriptor API. This is a compact exploratory representation, not the full planned spatial/multiscale pipeline.

LR tested C=0.1,1,10 with training-only standardization, balanced weights and liblinear. Extra Trees tested 256 trees, depths 16/32 and leaf sizes 2/5, with four workers. LR selected the image-view variant; trees used that same selected variant. All configuration decisions used synthetic selection AP. Sigmoid calibration and decision thresholds used the separate calibration partition. Models, membership lists, descriptors, predictions and source snapshots are saved in the run directory.

## Feature-pop selection comparison

| Feature | Best ordinary LR AP | Best enhanced LR AP |
|---|---:|---:|
| Straight interference | .8241 | .8755 |
| Curved interference | .8523 | .9010 |
| Dot scatter | .8807 | .9514 |

**OBSERVED:** The combined enhanced descriptor package improves selection AP in this experiment. This does not isolate the contribution of the feature-pop image from the other added views. Feature pop also highlights letters; it is not a feature segmentation mask. Thirty-two unit tests passed before completion, covering image transforms, padding exclusion, descriptor determinism and existing label/split/audit guards.

## Reserved synthetic test results

LR was selected for all three features under the predeclared rule favoring it when within .01 AP of Extra Trees. AP is average precision, a ranking metric, not percent accuracy. F1 below uses a fixed calibrated probability cutoff of .5, separate from abstaining decisions.

| Selected model | Test AP | Test F1 at .5 | Confident decision coverage |
|---|---:|---:|---:|
| Straight-line LR | .8832 | .7974 | 94.2% |
| Curve LR | .8800 | .7657 | 0% |
| Dot LR | .9455 | .8729 | 0% |

**FAILED / INCOMPLETE:** Curve and dot models have no accepted separated calibration thresholds under the implemented rule and therefore return uncertainty for every abstaining decision. Their ranking scores do not make them deployable confident classifiers. The current threshold rule independently picks widest positive/negative sets and rejects overlapping cutoffs; absence of an accepted pair does not prove that no feasible pair exists. Review this algorithm using calibration data only before making confidence claims. Calibration precision is estimated on the same calibration samples used for sigmoid fitting and is exploratory.

The fixed-.5 test macro F1 is approximately .812, below the plan's .85 exploratory target. No full-plan gate passes. Withheld-font and withheld-combination results are saved separately in results.json. The withheld straight-plus-dot combination has no positive curve examples; sklearn warned about undefined positive-class ranking support. Treat its curve AP/F1 as **not applicable/insufficient positive evidence**, not a meaningful zero or pass.

**UNRESOLVED:** Real evaluation and user annotation audit remain pending; nine feature types and withheld-renderer evaluation are unsupported. Group-bootstrap uncertainty intervals, clean-only false-positive analysis, full spatial aggregation and feature-to-function matching are not completed by this run. No real-CAPTCHA accuracy or successful reconstruction is claimed.

Next: audit the calibration threshold procedure, inspect errors and report uncertainty by base group without tuning on reserved test results. Preserve this first run as recorded. Publish the reviewed experimental artifacts into the canonical project with preimage/hash checks.
