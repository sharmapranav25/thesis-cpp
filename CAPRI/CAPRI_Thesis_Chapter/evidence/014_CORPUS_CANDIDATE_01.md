# Local-label repair and expanded candidate corpus

**OBSERVED, 2026-09-28 UTC:** Built 5,600 synthetic parent images and 168,000 overlapping native-pixel bricks, using the requested **10 across by 3 down** grid. After quality exclusions and exact deduplication, 119,292 unique crops remain. This is a **candidate corpus, not an accepted training corpus**. No classifier was trained, no real evaluation images or annotations were consumed, and no accuracy is claimed.

[Candidate gallery](../../runs/feature_bank_corpus_candidate_01/gallery.png) · [Status](../../runs/feature_bank_corpus_candidate_01/status.json) · [Protocol](../../runs/feature_bank_corpus_candidate_01/protocol.json) · [Integrity audit](../../runs/feature_bank_corpus_candidate_01/integrity_audit.json) · [Stricter local-label examples](../../runs/feature_bank_labeled_bricks_04/label_examples.png)

## Corrected local labels

**OBSERVED:** The previous rule could count a dot touching a glyph as an isolated dot, or let disconnected visible fragments borrow curvature from hidden recorded geometry. [Policy v3](../../feature_bank/labeled_bricks.py) now requires dot components and a one-pixel surrounding margin to avoid text/other-effect context. Missing background context produces uncertainty. Lines require a substantial connected visible component; curve fitting uses only surviving centerline pixels in that component. Low contrast, hidden effects and short fragments remain uncertain.

The same 64 pilot parents were regenerated as [run 04](../../runs/feature_bank_labeled_bricks_04/status.json). All pass the text-visibility check. Earlier runs remain intact. Raw counts below include overlapping and duplicate crops; these are label-generation results, not classifier predictions.

| Feature | Present | Absent | Uncertain |
|---|---:|---:|---:|
| Straight interference | 382 | 1,241 | 297 |
| Curved interference | 242 | 1,340 | 338 |
| Dot scatter | 415 | 832 | 673 |

**OBSERVED:** All 25 current unit tests pass, including added touching-dot, isolated-dot, missing-context, disconnected-line, hidden-curve-geometry and faint-contribution fixtures. [Tests](../../tests/test_labeled_bricks.py). These fixtures test specific failure modes, not universal semantic correctness. Antialiased overlay rendering has not been implemented or validated; the underlying text renderer produces antialiased glyphs.

## Broader synthetic coverage

**OBSERVED:** [Builder](../../scripts/build_brick_corpus.py) uses the unchanged existing Python text renderer and explicitly instrumented Pillow overlays. It does not claim PHP-library effect coverage. Native dimensions are 160x64, 200x80, 240x96 and 300x100. Text height/width, alphanumeric strings, colors, both polarities and faint effects vary. Every effect uses the same color-sampling distribution. Clean examples include `Il1`, `OO0Q` and `ijIl` as glyph-like hard negatives.

Two audited OFL fonts supply the main corpus. The third, Pixelify Sans, appears only in the withheld-font partition. All fonts and license files were hash-checked. Actual operation coordinates, colors and order are saved once per base and reused for all variants and counterfactual renders. Each parent saves final contribution masks; crops are stored as exact coordinates into saved parents, avoiding redundant image files.

**OBSERVED:** There are 1,000 main base groups, assigned **before rendering**: 600 train, 150 selection, 100 calibration, 150 test. Five variants per base request 20% clean, 40% single-effect and 40% combination images. Two training combination images fail the final per-glyph 0.55 contribution threshold and are retained as rejected artifacts. Accepted training composition is therefore 600 clean / 1,200 single / 1,198 combination, explicitly not an exact 20/40/40 ratio. No silent replacements or retries were made.

Another 100 groups supply 500 withheld-font parents. Another 100 groups supply 100 parents with the exact straight-line-plus-dots combination. That exact enabled set is absent from the main partitions; triples containing those two effects still exist there. This measures a withheld **exact combination**, not entirely unseen feature co-occurrence. No withheld-renderer partition is available yet.

## Duplicate exclusions and support

**OBSERVED:** Exclusions comprise 48,607 repeated crops within a partition, all 41 occurrences of exact crops appearing across partitions, and 60 crops from the two rejected parents. There are zero contradictory known labels among identical eligible crop pixels. All variants of a base retain one partition. Unsupported vocabulary features remain `unknown` rather than becoming negatives.

The following counts are after exact deduplication. A parent may contain both positive and negative local crops, so the two parent columns overlap. Variants sharing a base are dependent; distinct parent counts must not be described as independent statistical observations.

| Feature | Positive crops | Negative crops | Parents with positive crops | Parents with negative crops | Base groups with positive crops |
|---|---:|---:|---:|---:|---:|
| Straight interference | 12,161 | 35,132 | 1,175 | 1,949 | 545 |
| Curved interference | 9,634 | 37,131 | 1,403 | 2,039 | 537 |
| Dot scatter | 9,325 | 25,282 | 1,090 | 1,626 | 514 |

[Full support by split, including uncertainty](../../runs/feature_bank_corpus_candidate_01/support.json) · [Duplicate exclusions](../../runs/feature_bank_corpus_candidate_01/duplicate_audit.json) · [Quality failures](../../runs/feature_bank_corpus_candidate_01/rejections.json).

The `distinct_parent_support_check` is a count check only. It does not establish the plan's full verified-training-data gate. All rows still have `training_eligible=false`.

## Review, gates and next work

**OBSERVED integrity audit:** Independently verified all 5,600 parent hashes and 168,000 crop hashes; confirmed unique retained crops, base-group partition boundaries and reserved font/combination exclusions. Sixty deterministically sampled parents reproduce their images, masks and local labels exactly. A 63-bit DCT hash screen found 483 cross-partition candidate pairs; none met the subsequent normalized RGB mean-absolute-difference threshold of 4. These fixed heuristic checks do not prove absence of all perceptual duplicates. [Audit implementation](../../scripts/audit_brick_corpus.py).

**PROPOSED following user feedback:** After classification, estimate feature-specific density and orientation with uncertainty. [Parameter design and validation gates](FEATURE_PARAMETERS.md). These measurements are not implemented or evaluated in this run.

**OBSERVED by assistant:** Opened the nine pilot feature/state examples and the 48-parent candidate gallery. The gallery includes visible/faint effects, hard-negative lettering, multiple sizes/polarities and the withheld font. This is disclosed assistant visual review, not manual validation of all crops or an independent human annotation audit. The deterministic gallery's training row happens to select clean variants; use the full manifests rather than this gallery to assess composition.

**UNRESOLVED:** Human-identifiable local truth is not guaranteed by causal masks. Overlay antialiasing, broader line/curve geometry, PHP instrumentation, the remaining nine features, and verified capability registration remain incomplete. A heuristic near-duplicate screen is not proof against every visual shortcut. Two quality exclusions disturb exact class-mixture proportions. The real 200-image user annotation audit remains pending and must never be silently promoted from drafts.

**Gate: partial; training blocked by corpus acceptance, not by lack of a classifier implementation.** Next: resolve any audit flags and review local-label failures; validate/register the eligible capabilities and fill missing feature/renderer coverage or explicitly scope a partial subsystem; freeze an accepted corpus. Then evaluate the CV baseline across all five datasets before proceeding to Logistic Regression and then Extra Trees. Do not tune generation or thresholds on real evaluation labels. No full-CAPTCHA recreation is claimed.

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B scripts/build_brick_corpus.py --out runs/feature_bank_corpus_candidate_NEW
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B scripts/audit_brick_corpus.py runs/feature_bank_corpus_candidate_NEW
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
```

Saved source snapshots reproduce this experiment. Earlier experiments and the imported repository are preserved. Before publication, a metadata key was clarified from `independent_parent_support_gate` to `distinct_parent_support_check`; no image, label, split or measured count changed.
