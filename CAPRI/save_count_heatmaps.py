from pathlib import Path
import json,shutil,sys
HERE=Path(__file__).parent;ROOT=Path('E:/pranav/Final sem/sourcefree_synth');src=HERE/'count_heatmaps_01';dst=ROOT/'runs/style_count_heatmaps_01'
m=json.loads((src/'metrics.json').read_text());r=json.loads((src/'real_inference.json').read_text());assert sum(map(len,r.values()))==212
assert json.loads((src/'status.json').read_text())['state']=='complete'
assert all(set(v)=={'cv','dino','density','previous_count_cnn'} for v in m.values())
assert not dst.exists();shutil.copytree(src,dst)
# Self-contained local experiment entry points preserved alongside dependencies.
for name in ('compare_count_heatmaps.py','count_density_generator.py','train_character_count.py'):shutil.copy2(HERE/name,dst/name)
report='''# Character-count heatmap comparison 01

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
'''
(ROOT/'docs/STYLE_COUNT_HEATMAPS_01.md').write_text(report,encoding='utf8')
papers='''

## Implemented count-heatmap experiment 01 (supersedes earlier consulted-only status for DINO/density)

- Caron et al., Emerging Properties in Self-Supervised Vision Transformers, ICCV2021: https://arxiv.org/abs/2104.14294 ; official implementation https://github.com/facebookresearch/dino . Used cached Apache-2.0 ViT-S/8 natural-image pretrained backbone and `get_last_selfattention` CLS-to-patch extraction as in official `visualize_attention.py`. Reviewed official code and attention extraction. Our horizontal projection, peak counting and synthetic-only head/threshold selection are original heuristics, not a DINO counting method or paper reproduction. Exact cached commit unknown; source snapshots and SHA256 in run protocol. No CAPTCHA training or OCR. Outcome:53.87% synthetic exact, unreliable real localization/counts; not promoted.
- Lempitsky & Zisserman, Learning To Count Objects in Images, NeurIPS2010: https://www.robots.ox.ac.uk/~vgg/research/counting/NIPS2010.pdf . Read introduction and section2.1/equation1. Directly borrowed density-as-count-integral and Gaussian targets centered at point annotations. Here points come from synthetic glyph masks. Deviations: original positive CNN with map/count MSE, no linear feature regression, MESA loss, cutting-plane solver or convex QP. We renormalize boundary Gaussian mass to1 because synthetic glyphs are fully visible; paper permits boundary mass loss. Outcome38.99% exact synthetic, poor real transfer; not promoted.
- Darcet et al. remains consulted-only; no register-based model implemented.

Implementation and outcomes: runs/style_count_heatmaps_01/compare_count_heatmaps.py and docs/STYLE_COUNT_HEATMAPS_01.md. This records an unsuccessful counting baseline, not evidence that heatmap peaks reliably equal characters.
'''
for name,text in [('THESIS_PAPERS.md',papers),('STYLE_DECISIONS.md','\n\nD070 — User-requested DINO/CV/density count comparison completed. Synthetic-only threshold/checkpoint selection prevents real-family tuning. Reused cached frozen open-source DINO; trained density on synthetic glyph centers only. DINO53.87%, density38.99% vs priorCNN88.10% fresh synthetic exact; real212 mixed. Reason for rejection: attention follows noise/parts, density transfer fails. No promotion or automation restart. See STYLE_COUNT_HEATMAPS_01.md for provenance, deviations, validation and visual limitations.\n'),('STYLE_HANDOFF.md','\n\nLATEST: count heatmap comparison COMPLETE, runs/style_count_heatmaps_01; DINO/CV/density all fail to replace existing experimental countCNN. Read STYLE_COUNT_HEATMAPS_01.md. Real212 inference only, no real labels or accuracy. No active training, automation PAUSED. Active Phase3 remains count-only, optional refinement outsidepipeline.\n')]:
    with (ROOT/'docs'/name).open('a',encoding='utf8') as f:f.write(text)
sys.path.insert(0,str(ROOT));from style_pipeline.monitor import update
update('count_heatmaps_complete','DINO/CV/density comparison complete; no reliable replacement or pipeline promotion. Automation paused.',running=False,run=str(dst),real_images=212)
print('Saved comparison, report, decisions, thesis papers and monitor.')
