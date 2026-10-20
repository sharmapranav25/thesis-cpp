# Feature annotations v1

**PROPOSED and frozen for this run:** use the twelve definitions in [feature_definitions.json](../../runs/feature_bank_01/feature_definitions.json), with present / absent / uncertain labels. No transcription. Ambiguous causes remain uncertain. A decorated font does not automatically establish added noise, and glyph rotation alone does not establish nonlinear deformation.

**OBSERVED:** [prepare.py](../../feature_bank/prepare.py) verified all 6,708 historical unique image records and reused their existing partitions and exclusions. The [manifest](../../runs/feature_bank_01/manifest.json) freezes 40 validation members per dataset by SHA-256 ranking using seed `feature-bank-annotations-v1-20260927`. Original source paths are retained for provenance only; opaque IDs name exported review images. External dataset paths were explicitly relocated from historical staging paths and checked against original file and decoded-pixel hashes.

**OBSERVED:** The assistant opened all ten contact sheets before any classifier implementation. The [draft](../../runs/feature_bank_01/draft_annotations.json) contains 2,400 judgments and per-image caveats, with status UNAUDITED. Contact-sheet review can miss faint details; uncertain judgments are deliberately retained. The serialization script encodes this review, not a detector, and must never supply training or calibration labels.

Open [annotation_review.html](../../runs/feature_bank_01/annotation_review.html), enter your reviewer name, inspect each image, change any labels, and mark each image reviewed. Export is enabled only after all 200 images are reviewed. Progress is saved in browser local storage when available. Keep the exported JSON separately; the original draft remains unchanged.

Freeze the completed audit into a **new** directory:

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' -B -m feature_bank audit-freeze --selection runs/feature_bank_01 --audit '<path-to-user-export.json>' --out runs/feature_bank_audit_01
```

The importer requires matching manifest identity, a named user reviewer, exactly the selected IDs, all twelve valid labels, and an explicit reviewed flag per image. It retains disagreements and hashes the draft, manifest, and user export. This verifies record consistency; it cannot independently prove who operated the browser.

**PENDING:** User audit and frozen real evaluation labels. No real precision, recall, F1, calibration or accuracy is claimed. Evaluation annotations must never enter training, calibration, configuration selection, or renderer tuning. These datasets were previously inspected and are not pristine final tests.
