# Character-count heatmap comparison 01

Completed user-requested experiment. No pipeline promotion. Automation remains paused.

Compared Otsu foreground horizontal peaks, frozen official DINO ViT-S/8 last-layer CLS attention peaks, an original synthetic-trained density CNN, and the previous synthetic count CNN. DINO cached weights were reused, not downloaded or fine-tuned. Cached upstream commit is unknown; exact source snapshots, weight/source hashes and Apache-2.0 license are saved. All DINO heads plus their mean were eligible; head index3, distance2 patches, prominence0.1 selected only on synthetic selection MAE. CV selected distance20 pixels/prominence0.2. Both thresholds remain frozen for real inference.

Training: 1,344 synthetic images, 168 selection images, fresh 336-image test and 336-image withheld-font test. Seeds use disjoint 11/12/13/14-million bases plus count*10000+index. Counts2–8; same procedural generator as the previous count pilot, including its count-dependent width constraint. Density targets put a normalized Gaussian (sigma1.1 output pixels) at each individual glyph-mask centroid, renormalized to unit mass even at borders. Asserted target sum equals count. Original CNN trained1000steps, batch48, AdamW0.001; loss100*mapMSE+0.1*countMSE. Selected step1000 by synthetic selection MAE. Outputs16x40, positive softplus. Density count is the map sum, NOT number of heatmap peaks.

| Method | Fresh synthetic exact | Withheld-font exact |
|---|---:|---:|
| CV peaks | 50.89% | 48.81% |
| DINO peaks | 53.87% | 53.27% |
| Density CNN | 38.99% | 48.81% |
| Previous count CNN | 88.10% | 91.37% |

Density unrounded MAE0.777/0.655; DINO MAE1.042/1.027; CV MAE0.786/0.726; previous CNN MAE0.182/0.134. Exact density accuracy rounds the sum to nearest integer without clipping. Different training budgets/architectures mean this is a practical pilot, not a controlled architecture superiority claim. No improvement/reliability gate passed.

Real inference: same212 previously exposed fit/reference images across8sources, no labels used, no model/threshold selection on real images, no reserved evaluation access. First2images per source displayed in fixed order, not cherry-picked. Assistant visually inspected both comparison boards, not runtime VLM scoring or ground-truth evaluation. DINO sometimes matches visible glyph groups (Pymlex previews), but also follows interference lines, misses letters and splits one letter into multiple peaks. Roncuevas examples give3 then6 despite both appearing to contain3 glyphs. City gives7 then5; density underestimates City and overcounts noisy Roncuevas. Simple CV is sometimes better (City, Sloth examples). No real accuracy measured. A correct total can conceal incorrect peak locations.

Conclusion: DINO localization is interesting but bright spots are not reliable character instances. This density pilot also fails to improve counting; it is not integrated into generation. Existing count CNN remains experimental because synthetic accuracy does not establish real transfer. Next defensible improvement would broaden synthetic layout/noise and glyph-localization supervision and validate on separate data; do not tune on these displayed examples and call them unseen.

Artifacts: runs/style_count_heatmaps_01/{metrics,selected,protocol,training,real_inference,status}.json; density.pt; comparison_1.png and comparison_2.png; per-family panels and source snapshots. Historical runs unchanged.
