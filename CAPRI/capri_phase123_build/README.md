# CAPRI Phase 1-2-3 unified package

This release connects the three official CAPRI phases behind one `capri.py` command while preserving the frozen Phase 1 and Phase 2 snapshots.

1. **Phase 1** measures an unlabeled real-family fit pool with CV and frozen feature inference, producing a reusable visual profile.
2. **Phase 2** uses that profile with 35 broad, licence-recorded fonts and varied typography to generate labelled synthetic data over lengths 2–8 and broad character groups.
3. **Phase 3** trains CRNN, SVTR-style CTC and PARSeq readers using synthetic images and labels only. It selects/calibrates on synthetic partitions, freezes checkpoints, infers only on the real fit pool, keeps validation predictions separate, records disagreement and uncertainty, and generates a separate targeted synthetic corpus from a reviewed scheme.

The historical optional image-refinement experiments remain inside the frozen Phase 1 snapshot for provenance. They are not Phase 3 and are disabled in broad and targeted generation.

## Commands

```powershell
python capri.py verify

# Phase 1
python capri.py profile --references references.json --name my_family --kind conformer --variant 3 --device cpu --out runs/my_profile

# Phase 2
python capri.py generate-broad --family city --count 100000 --seed 731 --partition train --lengths 2 3 4 5 6 7 8 --out data/city_broad

# A fresh Phase 1 profile connects directly to Phase 2:
python capri.py generate-broad --profile runs/my_profile/profile.json --count 100000 --seed 731 --partition train --out data/my_family_broad

# Phase 3 end-to-end reader stage
python capri.py phase3-run --family city --corpus data/city_broad --real-split splits/city.json --run runs/city_phase3

# After reviewing runs/city_phase3/inferred_scheme.json
python capri.py generate-targeted --scheme runs/city_phase3/inferred_scheme.json --count 100000 --seed 732 --out data/city_targeted
```

`phase3-prepare` and `phase3-train` expose the two halves of `phase3-run` for resumable workflows. Every output must use a new directory.

## Boundaries

- Real transcription labels are never used for reader training or scheme inference. Paths remain locators and are never model input.
- The split manifest must contain nonempty `fit` and `validation` lists and `labels_read: false`. Only fit predictions enter the scheme.
- Synthetic labels drive checkpoint selection, calibration and reported synthetic accuracy. A predicted real histogram is not real accuracy.
- The renderer still has count-dependent layout. Layout stress results are reported; the package does not claim shortcut-resistant counting.
- Broad character groups remain digits, lowercase and uppercase. Missing predictions do not prove that a character is forbidden.
- The final production recognizer is outside this package's Phase 3 endpoint; Phase 3 ends with an audited targeted dataset.

## Package map

- `phase1/`: frozen CAPRI v2 visual profiling snapshot and cached profiles.
- `phase2/`: frozen broad-font typography renderer, fonts and licences.
- `phase3/reader/`: CRNN, SVTR-style CTC, PARSeq, auxiliary heads, training, inference and aggregation.
- `phase3/upstream/`: pinned open-source reader implementations and licences.
- `phase3/prepare.py`: data partition, hash and real-boundary verification.
- `phase3/targeted.py`: scheme-conditioned broad-font final generation and audit.
- `PROVENANCE.json`: source releases, commits, hashes and known limitations.

The package includes code and frozen Phase 1 assets. It does not bundle generated 100,000-image corpora or reader checkpoints; those remain output artifacts with their own hashes and provenance.
