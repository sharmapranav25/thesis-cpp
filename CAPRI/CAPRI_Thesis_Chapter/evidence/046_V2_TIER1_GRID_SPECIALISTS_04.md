# V2 Tier1: grid specialist experiment04

Eight grid-specific classifiers share a CNN feature extractor: 1x1,2x1,3x1,1x2,2x2,3x2,5x3,10x3 (columns x rows). Each grid head is shared across its own blocks. This is eight heads, not 63 independent neural networks. All heads train jointly from synthetic data. Crop presence targets use visible native-mask support >=3 pixels inside the crop. Tiny local fragments can be ambiguous (straight versus curved particularly); labels describe their originating generator feature, not guaranteed local identifiability.

Each crop's feature probabilities, encoded appearance and coordinates enter a learned pooling layer within its grid. Eight resulting summaries enter learned grid pooling to form global context for presence and full-resolution localization. Grid weights are learned attention, not calibrated confidence or a causal explanation. No family ID is supplied. A complete renderer recipe/style-story interface is not yet integrated.

Matched comparison: one shared local classifier versus eight local classifiers, SAME new pooling, local auxiliary supervision, data, optimizer, 1800 steps and checkpoint-selection rule. Existing03 multiview checkpoint provides a further baseline. Shared-versus-specialist is the direct architectural comparison; original03 also differs in fusion and auxiliary loss. Model parameter counts differ; one training seed is exploratory evidence only.

2400 synthetic training images/six fonts and420 selection images/Anton reused from03. Masks loaded at saved8-bit precision, identical for both04 models (03 trained float compositing masks). Local BCE class weights estimated on training only, capped20; mean each grid separately before averaging eight losses, preventing30-cell grid from dominating solely by cell count. Global loss plus unit-weight auxiliary local loss; select minimum global selection loss, excluding auxiliary. Fixed0.5 thresholds, no failed03 calibration carried forward. All heads verified to receive nonzero gradients; clean-target and global-grid consistency checks pass.

Fresh evaluation: 420 images each at seeds51m(same distribution),52m(heldout fonts only),53m(Bezier curve recipe only). No evaluation-driven checkpoint selection. Shared generator remains a limitation; new seeds are not independent-generator validation. Synthetic-only training; real data inference after freezing only,20previously exposed fit examples across five Tier1 families. Existing CV profile data retained beside learned predictions; CV not refitted in this architectural experiment. Tier1 retains nine font families; no Tier2/3 expansion, PARSeq training or Phase3 revival. Historical/default pipeline preserved. No automatic promotion or real-data gatepass.

## Results

| Model | Fresh same macro F1 | Font shift macro F1 | Curve shift macro F1 |
|---|---:|---:|---:|
| original03 | 0.831 | 0.817 | 0.807 |
| shared_local_head | 0.815 | 0.819 | 0.825 |
| grid_specialists | 0.822 | 0.814 | 0.813 |

| Feature | Shared local F1 | Specialist F1 | Shared local FPR | Specialist FPR |
|---|---:|---:|---:|---:|
| line | 0.659 | 0.641 | 0.502 | 0.433 |
| curve | 0.677 | 0.722 | 0.548 | 0.452 |
| dot | 0.858 | 0.900 | 0.028 | 0.053 |
| rectangle | 0.891 | 0.853 | 0.086 | 0.025 |
| grain | 0.991 | 0.994 | 0.004 | 0.000 |

Architecture audit verifies eight distinct parameter sets, gradient isolation for a selected head,63crop targets,all-clean negatives and bottom-right positional labels. Training additionally checks each head receives a nonzero gradient. Frozen real inference records local probabilities and learned grid weights beside the existing CV profile for each reference; no generated output changed.

## Interpretation and decision

Specialists versus matched shared local head: fresh same-distribution macroF1 .822 vs .815, font-shift .814 vs .819, recipe-shift .813 vs .825. Original03 scores .831/.817/.807. Separate heads help some dot/curve results but do not consistently improve overall behavior. Same-distribution specialist line/curve false-positive rates remain .433/.452. One run cannot establish statistical superiority; no promotion. This experiment implements the requested architecture and tests the hypothesis, not proof that specialization solves the remaining weaknesses. Next hypothesis should address ambiguous crop labels (e.g. a locally straight fragment of a curve), loss weighting and synthetic diversity before scaling classifiers blindly.

Feature order in full JSON: line,curve,dot,rectangle,grain. Read per-feature precision,recall,FPR,native-maskIoU and each grid classifier scores in results.json. Do not infer real CAPTCHA accuracy from these synthetic metrics.
