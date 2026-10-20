# Simple CNN on the existing DarkBay synthetic corpus

User-requested direct baseline following the failed instance detector. This experiment uses the already-generated100,000DarkBay CAPRI_v2 images, not the separate procedural count corpus and not the newer35-font smoke samples. No new synthetic data is generated.

## Training

Known synthetic labels supply only their string lengths. A three-block RGB CNN predicts seven classes,2–8. No transformer, object detector, feature pop or real-image fitting. Input192×64, batch128, AdamW learning rate0.001/weight decay0.0001, maximum6epochs with patience2 on synthetic selection cross-entropy. One seed731. Models are checkpoint-selected before real inference.

Synthetic text-hash grouping gives89,714train,5,144selection and5,142test images. Repeated strings stay within one partition. Fonts and family profiles are shared across these internal splits, so held-out synthetic accuracy is an in-distribution measurement. The existing renderer's count-dependent placement remains deliberately intact: this is the user's requested empirical baseline, not a shortcut-free benchmark.

The entire real DarkBay `pool` directory is used for frozen inference. The sibling evaluation directory is excluded. Real filenames locate files only; their character strings are not decoded as labels, read into the CNN or used for model selection. Histograms are predictions, not real accuracy. No character-count inference is promoted automatically.

## Evidence

- `protocol.json`: source label hash, split sizes, model and training settings.
- `membership.json`: exact synthetic partition membership.
- `history.json`, `selected.json`, `best.pt`: epoch progress and chosen model.
- `test.json`, `test_predictions.npz`: held-out synthetic evaluation.
- `real_predictions.json`, `real_summary.json`, `darkbay_predictions.png`: real inference and fixed-order visual examples.
- `status.json`: actual progress.

The existing dataset and other classifiers are preserved. Runtime VLM use: none. Assistant qualitative review of the resulting gallery is not independent count ground truth.
