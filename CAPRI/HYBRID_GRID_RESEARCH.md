# Grid and bidirectional hybrid research — 2026-09-28

Status: researched, not implemented or trained. Current models and production remain unchanged.

## Gridded Transformer Neural Processes

Ashman et al., ICML2025, https://proceedings.mlr.press/v267/ashman25a.html . Read final publication abstract and preprint sections4.1–4.3 at https://arxiv.org/html/2410.06731v2 . Distinguish preprint from final publication: final PDF retrieval failed. Official MIT repository: https://github.com/cambridge-mlg/gridded-tnp .

The preprint encodes observations into spatial pseudo-tokens using neighbourhood cross-attention, processes them with ViT/Swin, and decodes predictions at queried locations. Its task is conditional regression, not CAPTCHA multi-label classification. Potential borrowed idea: position-aware aggregation into a compact latent grid. No temporal or weather machinery is needed here. Our dense images already have a grid, so handling irregular observations offers limited immediate benefit. Equivariance/uncertainty claims do not automatically transfer to a modified classifier. No implementation copied or experiment run.

## Visual Conformer

Peng et al., ICCV2021, https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Conformer_Local_Features_Coupling_Global_Representations_for_Visual_Recognition_ICCV_2021_paper.html . Read abstract, indexed paper discussion and official README/source: https://github.com/pengzhiliang/Conformer ; Apache2.0. This is visual Conformer, not the speech architecture with the same name.

Concurrent CNN and transformer branches exchange representations through feature-coupling units. This is the closer match to the user's proposed local-to-global and global-to-local feedback. Candidate adaptation: small CNN preserving fine detail, a coordinate-aware token grid, and repeated feature exchange before15classification heads. This proposed compact variant would not be a faithful paper reproduction. Implementation and performance are pending.

## Proposed discriminating experiment

Keep data and feature definitions fixed initially. Compare CNN alone, one-way CNN-to-transformer, and bidirectional coupling; optionally add learned grid pooling as a separate ablation. Match capacity/compute approximately and disclose differences. Use fresh synthetic tests, repeated seeds, per-feature gates and real-development comparisons after freeze. Specifically inspect block/grid confusions, fine strokes and colour/texture shifts. Label quality and generator coverage remain alternative failure causes. Renderer-parameter regression remains a separate unimplemented hypothesis. No new training launched by this research review.
