"""Public API for the synthetic captcha generator.

Two entry points, both expose `render(t, n)` and `render_sample(t)`:

    from src.generator import ParametricGenerator
    gen = ParametricGenerator(seed=0)
    img, label, bboxes = gen.render_sample(t=1.0)   # one
    samples = gen.render(t=0.5, n=64)               # batch

    from src.generator import RefinedGenerator
    gen = RefinedGenerator(seed=0)                  # uses default v4-best ckpt
    img, label, bboxes = gen.render_sample(t=1.0)

`t in [0, 1]` is the difficulty knob. EASY = clean / single font / no rotation;
HARD = the DINOv2-tuned config. RefinedGenerator additionally applies a trained
mask-aware U-Net for high t (alpha = max(0, 2t - 1)).

For external projects (e.g. `model 1/`) — see `mvp_api.md` at the project root.
"""
from .parametric import ParametricGenerator
from .real_bg import RealBgGenerator
from .refined import RefinedGenerator

__all__ = ["ParametricGenerator", "RefinedGenerator", "RealBgGenerator"]
