# Specialist features with 107 XGBoost outputs

The exported encoder has 2,179,108 frozen neural parameters and contains no effect, text, font, length or CTC prediction modules. The 107 saved tree models replace all supported image-feature outputs. Optional CTC weights are a separate diagnostic sidecar and are not loaded by the feature predictor.

From `E:/pranav/Final sem/sourcefree_synth`, use the recorded Python environment:

```powershell
& 'E:\pranav\Final sem\sourcefree_synth\experiments_xgb\.venv\Scripts\python.exe' -m feature_bank.iter2.xgb_replacements.predict --bundle 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_xgb_specialists_work\run_20261004_02\bundle' --image 'path-to-image.png' --out 'prediction.json' --device cuda
```

Project Python sources are required. The exact runtime snapshot is in `runtime`; checksums and Torch/XGBoost versions are in `checksums.json`. CUDA bf16 inference matches fitting extraction; the CPU option uses FP32 neural arithmetic and can differ near tree splits. Both CLI paths and italic decoding have been checked.

Trees and the optional reader make the saved package about 291 MiB. All original neural checkpoints remain available. Font category improves, while specialist neural has stronger mean font ordinal results; detailed four-model comparisons are in `../FOUR_MODEL_COMPARISON.md`, with complete verification in `../verification.json` and renderer previews in `../REPORT.md`.
