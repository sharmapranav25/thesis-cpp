# Executed release validation

-336generated samples:42per8families,2per each of21cells. Labels/length/type, renderer glyph counts and individual visibility>=.55 checked. Per-sample retries retained in test evidence.
-Byte-equivalent RGB replay of first historical Phase1+Phase2 render for all8families with the exact original text/font/seed/variant. This verifies packaging preserved baseline behavior.
-Fresh CV and frozen three-class inference on2Cityfitting references matched cached image-level feature decisions. Font inference path ran onCPU. This does not establish classifier/font accuracy.
-Copied package to a different directory, generated21validation samples in diverse mode from an unrelated working directory without old source scripts/assets in its module path.
-Generated6,300text draws across allcells/partitions: label sets disjoint across train/validation/test. Stable text split is not a substitute for pixel audit.
-Audited63train/validation image manifests,CSV labels,image hashes and cross-partition text/pixels. Passed.
-Negative leakage test: deliberately reused style-fit image in an evaluation manifest; audit rejected it as required.
-Deterministic repeat:42Cityimages reproduced identical pixel hashes with same seed and settings.
-No PARSeq optimizer steps, no model retraining, no100kbulk generation. No real style/recognition gate claimed.

Evidence source: sibling phase12_validation_01. Installed release includes validation_summary.json. The test runner uses current local staging paths and is retained in release provenance/test evidence rather than a mandatory runtime dependency.
