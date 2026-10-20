# Synthetic feature-negative revision01

Separate synthetic-only LR models trained; originalmodelhashes unchanged.512 new training dot-only negatives,128 each selection/calibration/test, disjointseedranges. No realimages used. ThreeC candidates, negativeweight4. Selection gate: ordinary AP decline no more than.02; minimize negative FP among passing candidates. Newnegative test accessed afterselection. Ordinarytest is historical reusedreporting, not pristine.

| Feature | Selection gate | Ordinary test AP old/new | Dot-only FP at .5 old/new |
|---|---|---|---|
| Straight | Pass | .8832 / .8709 | 71.09% / .78% |
| Curve | Fail | .8800 / .8063 | 97.66% / .78% |

Straight C1 passes selectiongate with AP.8609 vsbaseline.8755. No curve candidatepasses; diagnosticfallback C.01 saved but REJECTED forpromotion. Do not confuse savedcheckpoint withacceptedmodel. Newcalibrator fit onlysyntheticcalibration. Bothbundles have thresholdsNone and are not integrated into runtime.

Next: determine straight abstentionthresholds from syntheticcalibration only and run exploratory frozenCV+inference; investigate a less harmful curve training balance on selectiondata, retain samegate and use freshnegative reportingtest forfuturerun. Never select using these testresults. Neither outcome proves realstyle improvement. No newpaper adopted; original local augmentation. Training script snapshot retained.
