"""captcha-effects: parameterised CAPTCHA background, interference and deformation effects with exact per-pixel labels
and a calibrated 0-10 intensity per effect. Text is left to the caller (e.g. real CAPTCHA libraries)."""
from captcha_effects.canvas import Canvas
from captcha_effects.effects import REGISTRY
from captcha_effects import warp as _warp  # noqa: F401  (registers post effects)
from captcha_effects.render import (ALL_EFFECTS, UNDER, OVER, POST, new_canvas, draw_under, apply_glyph, draw_over, apply_post,
                                    draw_text, sample_spec)
from captcha_effects.glyph import GLYPH_EFFECTS
from captcha_effects.labels import STYLE_SCHEMA, style_labels
from captcha_effects.measure import visible_amount
from captcha_effects.text import FontBank, sample_text_spec, draw_text_spec, text_labels
from captcha_effects.captcha import render_captcha

__version__ = "0.5.0"
__all__ = ["Canvas", "REGISTRY", "ALL_EFFECTS", "UNDER", "OVER", "POST", "GLYPH_EFFECTS", "new_canvas", "draw_under",
           "apply_glyph", "draw_over", "apply_post", "draw_text", "sample_spec", "STYLE_SCHEMA", "style_labels", "visible_amount", "FontBank", "sample_text_spec", "draw_text_spec", "text_labels",
           "render_captcha", "__version__"]
