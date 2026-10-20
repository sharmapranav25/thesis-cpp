from pathlib import Path
p=Path('E:/pranav/Final sem/sourcefree_synth/docs/THESIS_PAPERS.md')
text='''

## Count heatmaps discussion: consulted, NOT implemented

- Caron et al., Emerging Properties in Self-Supervised Vision Transformers, ICCV2021. Primary abstract/search-indexed paper summary consulted: https://arxiv.org/abs/2104.14294 . DINO attention can expose object regions; no evidence from this paper that one CAPTCHA character equals one attention peak. No weights downloaded or DINO code used.
- Darcet et al., Vision Transformers Need Registers, arXiv2309.16588. Primary abstract opened: https://arxiv.org/abs/2309.16588 . Documents high-norm/background artifacts in attention/features and register remedy. Relevant caution for heatmap interpretation, not implemented.
- Lempitsky and Zisserman, Learning To Count Objects in Images, NeurIPS2010. Primary Oxford abstract consulted: https://robots.ox.ac.uk/~vgg/publications/2010/Lempitsky10b/ . Density integrates to count; point annotations supervise localization/counting. Proposed future adaptation: synthetic glyph centers and unit-mass density targets, no real labels or character recognition. No density model implemented yet; would be an original neural adaptation, not reproduction of their convex optimization method.

User suggests DINO heatmap counting. Assessment: worth an experimental baseline, not assumed reliable or simpler computationally. Compare with simple projection/component baseline and synthetic-supervised center/density map; do not promote on appearance alone. Existing direct count model remains experimental. Automation remains paused.
'''
with p.open('a',encoding='utf-8') as f:f.write(text)
