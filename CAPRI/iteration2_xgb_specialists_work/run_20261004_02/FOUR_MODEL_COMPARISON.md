# All four model runs

Specialist neural has the strongest mean font ordinal result on both benchmarks. Specialist + XGBoost leads font category on both and improves effect intensity, but loses weight/slant/italic quality relative to specialist neural. Simple + XGBoost is severely weaker on slant and effect detection.

All models use the original 100,000-image synthetic corpus. Seen-test has 2,000 images; generalisation has 2,500 with held-out fonts. Cached label arrays match exactly across the two XGBoost experiments on every split. No new fitting or threshold tuning was done for this comparison.

## Seen-test

| Metric | Simple | Simple + XGB | Specialist | Specialist + XGB |
|---|---:|---:|---:|---:|
| Category accuracy | 68.70% | 70.10% | 73.50% | **73.80%** |
| Stroke accuracy | 95.05% | 95.13% | 95.68% | **95.76%** |
| Weight within ±1 | 80.15% | 77.90% | **81.75%** | 79.95% |
| Width within ±1 | 73.70% | 72.90% | **75.10%** | **75.10%** |
| X-height within ±1 | 69.40% | 60.90% | **70.80%** | 64.95% |
| Slant within ±1 | 91.50% | 36.35% | **91.60%** | 88.70% |
| Italic F1 | 72.98% | 71.73% | **76.81%** | 74.48% |
| Font ordinal mean within ±1 | 78.69% | 62.01% | **79.81%** | 77.18% |
| Effect presence F1 (calibrated) | 58.54% | 46.67% | **61.98%** | 61.27% |
| Text presence F1 (calibrated) | 59.05% | 59.07% | **63.12%** | 62.13% |
| Effect intensity within ±1 | 80.47% | 73.83% | 81.96% | **82.27%** |
| Text intensity within ±1 | 61.41% | 58.98% | **65.70%** | 65.31% |
| Length accuracy | 84.90% | 85.65% | 87.75% | **87.95%** |

## Generalisation: held-out fonts

| Metric | Simple | Simple + XGB | Specialist | Specialist + XGB |
|---|---:|---:|---:|---:|
| Category accuracy | 56.56% | 56.40% | 57.24% | **58.32%** |
| Stroke accuracy | 89.20% | 89.14% | **90.11%** | 89.88% |
| Weight within ±1 | 80.12% | 79.52% | **81.88%** | 80.56% |
| Width within ±1 | 72.04% | 71.80% | 72.56% | **73.36%** |
| X-height within ±1 | 58.60% | **65.16%** | 61.20% | 64.52% |
| Slant within ±1 | 91.92% | 37.84% | **92.00%** | 88.16% |
| Italic F1 | 65.70% | 63.70% | **67.17%** | 65.03% |
| Font ordinal mean within ±1 | 75.67% | 63.58% | **76.91%** | 76.65% |
| Effect presence F1 (calibrated) | 59.66% | 47.08% | **62.83%** | 62.42% |
| Text presence F1 (calibrated) | 58.93% | 58.70% | **62.13%** | 61.66% |
| Effect intensity within ±1 | 81.57% | 74.72% | 82.63% | **83.28%** |
| Text intensity within ±1 | 62.06% | 59.20% | **65.34%** | 63.94% |
| Length accuracy | 85.32% | 86.00% | **88.40%** | 88.20% |

Within ±1 permits one ordinal-bin error. Font ordinal mean averages weight, width, x-height and slant; it excludes category, stroke and italic. No all-font-attributes-exact metric was saved. F1 percentages are averages across feature heads, not ordinary classification accuracy.

## Comparison protocol

Effect/text presence uses the same synthetic calibration rule for all four: maximum F1 subject to FPR <= .10. The older Simple + XGBoost summary already reports calibrated decisions; comparing it with raw .5 neural F1 would overstate its gain. This table instead reads calibrated neural reports.

Font italic uses .5 for all four. The older Simple + XGBoost fit also calibrated italic; this comparison replays only its saved italic booster on cached vectors at .5, using selected rounds. Its original reports/model are preserved; historical and matched scores are in four_model_comparison.json.

Simple font/length/ordinal scores come from its official saved final report. The prior XGBoost experiment contains slightly different simple neural arithmetic from another batching path; tiny differences are not evidence of meaningful improvement. No confidence intervals or new independent benchmark are available.

## Differences between XGBoost runs

Simple + XGBoost used frozen simple features: CNN stem mean/max over the image and four vertical bands (960 dimensions) for effects, and transformer task queries for text/font/length. Specialist + XGBoost uses each trained CNN effect query (128 dimensions) and specialist transformer queries. The newer export physically removes neural feature classifiers and executes no CTC reader.

Older fitting allowed 800 rounds, 30-round early stopping and balanced ordinal regressions. New fitting allows 2,000 rounds, 50-round early stopping and balanced/unweighted selection for ordinal regression. Depth, learning rate, sampling and regularisation match. This compares four actual runs; it cannot attribute every difference solely to specialist training.

Old Simple + XGBoost CTC scores belong to the unchanged simple neural reader. New specialist XGBoost inference executes no reader; its optional sidecar preserves specialist reader weights. Boosting did not improve sequence reading.

All 107 output scores on both benchmarks: four_model_per_head.csv. Sources and label alignment: four_model_comparison.json. Replacement bundle QA and preview sheets: REPORT.md.

