# Feature-bank pilot 01 — resource and adapter gate

Date: 2026-09-28 UTC. **PARTIAL — supervised-corpus gate not passed.**

**OBSERVED:** 64 initial cases for each of four PHP backends and the existing Python renderer: 320 images. Every image was decoded and all ten contact sheets opened for assistant review. Each initial successful render was independently repeated. No synthetic training corpus or classifiers were trained.

| Backend | Rendered | Exact seed replays | Edge / bbox flags |
|---|---:|---:|---:|
| gregwar | 64/64 | 64/64 | 4 |
| securimage | 64/64 | 64/64 | 0 |
| mobicms | 64/64 | 0/64 | 0 |
| simple | 64/64 | 64/64 | 4 |
| python | 64/64 | 64/64 | 0 |

**OBSERVED:** All five invalid-parameter tests passed and all six annotation-gate unit tests passed. No numerical classifier metrics exist. Initial run elapsed time: 108.07 seconds, including repeat renders and source verification, excluding visual review and later repairs.

**OBSERVED:** 164 downloaded source files and 50 original/vendor paths retained their recorded hashes. The original repository and historical experiments were not edited.

## Visual findings

### gregwar

OBSERVED: straight strokes and deformation are visible. Cases 048–051 reach the text edge; reject for supervised use pending clipping repair. Whole-image warp also bends interference lines.

[Cases 000–031](../../runs/feature_bank_pilot_01/gregwar_gallery_1.png) · [Cases 032–063](../../runs/feature_bank_pilot_01/gregwar_gallery_2.png) · [Per-case records](../../runs/feature_bank_pilot_01/gregwar/results.json)

### securimage

OBSERVED: curves and fine scattered marks are visible. Some curve segments are nearly straight. Enabling noise changes text position despite the same seed; these four-column rows are not counterfactual pairs.

[Cases 000–031](../../runs/feature_bank_pilot_01/securimage_gallery_1.png) · [Cases 032–063](../../runs/feature_bank_pilot_01/securimage_gallery_2.png) · [Per-case records](../../runs/feature_bank_pilot_01/securimage/results.json)

### mobicms

OBSERVED: varied fonts, colors and rotations. All initial seed-only repeats differ. Native decorative fonts were not used; case-tuning disabled. Capture/replay repair is a separate run.

[Cases 000–031](../../runs/feature_bank_pilot_01/mobicms_gallery_1.png) · [Cases 032–063](../../runs/feature_bank_pilot_01/mobicms_gallery_2.png) · [Per-case records](../../runs/feature_bank_pilot_01/mobicms/results.json)

### simple

OBSERVED: shadow and rotation are visible. Some text/shadows approach or cross an edge; e.g. 022–023. Corner checks flag four cases but are not a complete clipping detector. No bundled texture backgrounds were used because per-asset licenses remain unresolved.

[Cases 000–031](../../runs/feature_bank_pilot_01/simple_gallery_1.png) · [Cases 032–063](../../runs/feature_bank_pilot_01/simple_gallery_2.png) · [Per-case records](../../runs/feature_bank_pilot_01/simple/results.json)

### python

OBSERVED: thin lines, sparse dots, grid and block combinations are visible. Several marks are faint; nominal requested operations must not automatically become positive visible labels. Glyph shapes are stretched by the existing renderer. Internal geometric checks are not readability proof.

[Cases 000–031](../../runs/feature_bank_pilot_01/python_gallery_1.png) · [Cases 032–063](../../runs/feature_bank_pilot_01/python_gallery_2.png) · [Per-case records](../../runs/feature_bank_pilot_01/python/results.json)

## MobiCMS repair 02

**OBSERVED:** The separately instrumented adapter records native random_int draws and validates their bounds/order when replayed. 64/64 outputs reproduce exactly in the repair run. The one-import change and limitation are preserved in [instrumentation.json](../../runs/feature_bank_mobicms_replay_02/instrumentation.json) and [diff](../../runs/feature_bank_mobicms_replay_02/instrumentation.diff). This resolves the replay issue for the instrumented variant; it does not pass clipping or final visible-label checks.

## Licenses and dependency scope

**READ:** Gregwar and MobiCMS code notices are MIT; simple-php-captcha is MIT. Securimage composer metadata says BSD-3-Clause while its LICENSE.txt contains two redistribution conditions; preserve both and resolve the discrepancy before redistribution. Gregwar supplies per-font OFL notices. The three pilot fonts passed ASCII alphanumeric cmap checks and their notices were copied alongside the MobiCMS font subset. This does not prove visual distinguishability of every glyph.

**READ:** Securimage supplies a separate freeware font notice. MobiCMS baby_blocks.txt explicitly says its license is unknown; other decorative-font rights need per-asset review. simple-php-captcha attributes its textures and font but does not establish their individual licenses in the readme. Those unresolved assets were not used in pilots.

**OBSERVED:** [pilot_runtime.lock.json](../../runs/feature_bank_pilot_01/pilot_runtime.lock.json) freezes the direct-load execution closure and runtime; it is not a Composer dependency lock. Gregwar declares symfony/finder but the exercised code did not load it. Full dependency resolution, including Laminas, remains pending. Laminas has not been piloted or used in training.

## Gate and next work

**PROPOSED:** Finish per-glyph clipping rejection, captured-randomness effect toggles and final visible-effect masks; isolate additional background/text operations; resolve dependency and asset gaps; pilot the withheld renderer. Rerun into new directories. Only then build the capped corpus. Do not silently discard these failures or mark the entire stage passed.

**PENDING:** [User annotation audit](ANNOTATION_PROTOCOL.md). The draft labels remain evaluation-only and cannot authorize scored real-image claims. Classifier ordering remains CV rules → logistic regression → Extra Trees → selection/matching, with five-dataset review gates.

## Related records

- [Acquisition](RESOURCE_ACQUISITION_01.md)
- [Adapter design and limitations](ADAPTERS.md)
- [Annotation protocol](ANNOTATION_PROTOCOL.md)
- [Mapping coverage and gaps](MAPPING_COVERAGE.md)
- [Machine-readable results](../../runs/feature_bank_pilot_01/status.json)
