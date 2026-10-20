# Labeled-brick pilot 03

**OBSERVED:** Built a small synthetic-only label-validation demonstration: 8 independent base configurations, all 8 combinations of three effects per base, 64 images, and 30 overlapping bricks per image (1,920 crops). No real images, real annotation files, target generator source, or target fonts were used. No classifier was trained.

[Local image/mask/label examples](../../runs/feature_bank_labeled_bricks_03/label_examples.png) · [All parents, page 1](../../runs/feature_bank_labeled_bricks_03/parents_1.png) · [Page 2](../../runs/feature_bank_labeled_bricks_03/parents_2.png)

## Implementation and provenance

**READ / OBSERVED:** The unchanged existing Python `render_once` makes clean text on fixed generic backgrounds using two previously audited OFL fonts from the downloaded Gregwar resource. New, explicitly instrumented Pillow operations draw straight lines, sine curves and dots. These overlays are new Python bank candidates; they are **not** Gregwar/Securimage PHP effects or a cross-renderer generalization result. Their [source](../../feature_bank/labeled_bricks.py) and exact per-base recipes are saved. The old Python renderer and imported repository are untouched.

**OBSERVED:** Each base records its actual operation geometry/colors once. All variants and leave-one-effect-out renders compose those same operations without calling the RNG again. Saved masks distinguish requested operation pixels from visible final contribution. A pixel contributes only if removing that effect changes an RGB channel by at least 20. All 64 parent images replay exactly and pass the existing base-text checks plus a final minimum visible-text contribution of 0.55 per glyph. Those checks are not an OCR/readability evaluation.

## Local labeling policy

**PROPOSED engineering settings, frozen before run 03:** [local-evidence-v2](../../runs/feature_bank_labeled_bricks_03/protocol.json).

- A feature elsewhere in the parent does not make a crop positive. No operation pixels in the crop gives a synthetic absent label for that added effect; intrinsic glyph strokes are hard negatives.
- Faint, hidden or substantially occluded effects are uncertain. Require at least 8 changed pixels and 65% retained contribution before considering a positive.
- Lines need at least an 8-pixel visible span. Curves also need continuous local arc context with at least 1.5-pixel deviation from a fitted straight axis. A globally curved stroke that looks locally straight yields uncertainty; it does not create an unquestioned curve positive.
- Dot scatter requires at least two complete, visibly distinct connected marks with at least 70% contribution each. Duplicate/overlapping draws cannot count as multiple marks.
- Ambiguous curve fragments also prevent confident straight-line negatives; ambiguous tiny strokes prevent confident dot negatives.
- The other nine vocabulary labels are **unknown**, never encoded as negative. All samples remain `training_eligible=false` pending the label and corpus gates.

Masks are supervision/provenance artifacts, **not permitted classifier inputs**. Future models must use image-derived descriptors only, with labels and generator IDs kept out of descriptor computation. No training loader or classifier was implemented in this step, so that future boundary is a required check, not a completed claim.

## Counts and checks

**OBSERVED:** Raw crop counts (overlapping/repeated crops are not independent support):

| Feature | Present | Absent | Uncertain |
|---|---:|---:|---:|
| Straight interference | 430 | 1,276 | 214 |
| Curved interference | 312 | 1,340 | 268 |
| Dot scatter | 644 | 868 | 408 |

There are **1,139 unique pixel crops and 781 duplicate occurrences**, with zero contradictory present/absent labels among identical crops. Base groups are retained across every combination. All examples are currently pilot-only; no train/test split or accuracy score is claimed. A later corpus must split complete base groups first, remove cross-partition duplicate crops, and count sufficient independent training support. These 1,920 rows must not be called 1,920 independent examples.

Nineteen tests passed, including occlusion, invisible effects, absence of effects outside a crop, recorded replay, a single-dot abstention, duplicate-dot counting, region coverage, group leakage guards and user-audit gates. [Tests](../../tests/test_labeled_bricks.py) · [Counts](../../runs/feature_bank_labeled_bricks_03/status.json) · [Duplicate audit](../../runs/feature_bank_labeled_bricks_03/duplicate_audit.json) · [Brick labels/evidence](../../runs/feature_bank_labeled_bricks_03/bricks.json).

## Failures and revisions retained

**OBSERVED:** [Pilot 01](../../runs/feature_bank_labeled_bricks_01/status.json) assigned a fixed color to each effect and coupled font choice with background polarity. That could provide shortcuts, so it was rejected as a training design before any training. [Pilot 02](../../runs/feature_bank_labeled_bricks_02/status.json) gives all effects the same color-sampling distribution and shows each font in both polarities. Run 03 additionally counts distinct connected dot components, reducing dot-positive rows from 660 to 644. The image pixels in runs 02 and 03 are identical; only the local dot-counting rule changed. Earlier runs and source snapshots are retained.

## Review and limits

**OBSERVED by assistant:** Opened all 64 parent renders in each of the first two batches and the deterministic nine local feature/state examples. Run 03 has identical parent pixels to run 02, verified separately; its label examples were opened. Visible masks track overlay contribution in the reviewed examples; truncated/near-straight curves and single dots illustrate why uncertain is necessary. This is sampled crop review, **not** manual verification of all 1,920 local judgments. The panels show masks, not model predictions.

**UNRESOLVED:** Contribution masks do not establish human-identifiable feature truth in every ambiguous crop. Marks touching glyphs, fragmented line visibility and antialiased contours need further inspection and stronger fixtures. Backgrounds, dimensions, effect strengths and typography are still narrow; a shared sampling distribution alone cannot prove shortcut-free learning. No statistical generalization claim follows from eight base groups. PHP mask instrumentation, the remaining nine features, withheld-renderer testing and real-image auditing remain incomplete.

**Gate: partial.** This demonstrates the label-generation mechanism, not a passed full-corpus or classifier gate. No feature reaches the plan's required 1,000 known positive and 1,000 known negative training examples. The next step is to review ambiguous local labels, broaden and balance verified synthetic coverage, then construct group-disjoint training/selection/calibration/test partitions. Do not tune that process on real evaluation labels.

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B scripts/make_labeled_brick_pilot.py --out runs/feature_bank_labeled_bricks_NEW
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
```
