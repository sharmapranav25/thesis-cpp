"""Director (pipeline step 2) — the VLM reads the real captcha and emits a STRUCTURED
generation spec + a difficulty score, replacing the hand-authored per-scheme SCHEME_CFG.

Discipline (DESIGN.md §2): the VLM only picks COARSE forced-choice categories; this
module maps those onto concrete, discretised php_gen knob ranges (no free-form floats,
which make critic loops oscillate). Exact colours are read from PIXELS, not the VLM.
Difficulty = VLM ordinal bucket fused with a free classical complexity prior.

`Spec.to_php_cfg(W,H,fonts)` yields a dict with EXACTLY the keys php_gen_ext.php reads
(mirrors cryptbb_sourcefree/src/php_pool.py:29 SCHEME_CFG + a new `colors` block).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import vlm

# --- coarse-category -> concrete knob presets (the discretised grid) ---------- #
_WAVE = {
    "none":   dict(on=False, ampx=[0.0, 0.0], ampy=[0.0, 0.0], per=[1.0, 1.0]),
    "mild":   dict(on=True,  ampx=[0.0, 1.0], ampy=[0.4, 1.6], per=[0.9, 1.8]),
    "strong": dict(on=True,  ampx=[0.0, 1.5], ampy=[0.8, 2.6], per=[0.7, 1.5]),
}
_ROT = {"none": 4, "mild": 10, "strong": 18}
_NOISE = {
    "none":  dict(lines=[0, 0], arcs=[0, 0], dots=[0.0, 0.0]),
    "light": dict(lines=[0, 2], arcs=[0, 1], dots=[0.0, 0.008]),
    "heavy": dict(lines=[1, 4], arcs=[0, 3], dots=[0.0, 0.020]),
}
_SIZE = {"thin": (0.58, 0.82), "regular": (0.60, 0.86), "bold": (0.64, 0.90)}


@dataclass
class Spec:
    difficulty: int = 3               # 1 (easy) .. 5 (hard)
    bg: str = "pale"                  # pale | dark
    bg_texture: str = "solid"         # solid | mosaic | noise | lines
    charset_style: str = "bw"         # vivid | bw   (php glyph colouring)
    serif: str = "sans"               # serif | sans | mono   (font-branch hint)
    weight: str = "regular"           # thin | regular | bold
    case: str = "mixed"               # upper | lower | mixed | digits
    wave: str = "mild"                # none | mild | strong
    noise: str = "light"             # none | light | heavy
    colored: bool = False
    colors: dict = field(default_factory=lambda: {"bg_hex": "#f0f0f0",
                                                   "glyph_hex": "#202020",
                                                   "noise_hex": "#808080"})
    length_modes: list = field(default_factory=list)   # filled by infer_length later
    raw: dict = field(default_factory=dict)

    def to_php_cfg(self, W: int, H: int, fonts: list[str]) -> dict:
        smin, smax = _SIZE.get(self.weight, _SIZE["regular"])
        cfg = dict(
            W=W, H=H, fonts=fonts,
            charset_style="vivid" if self.colored else "bw",
            bg="dark" if self.bg == "dark" else "pale",
            min_size_frac=smin, max_size_frac=smax,
            max_rot=_ROT.get(self.wave, 10),
            wave=_WAVE.get(self.wave, _WAVE["mild"]),
            grid=bool(self.bg == "pale" and self.bg_texture in ("mosaic", "lines")),
            colors=self.colors,
        )
        cfg.update(_NOISE.get(self.noise, _NOISE["light"]))
        return cfg


# --------------------------------------------------------------------------- #
# Classical complexity prior (free, deterministic — hedges VLM miscalibration)
# --------------------------------------------------------------------------- #
def _classical_difficulty(img: Image.Image) -> int:
    a = np.asarray(img.convert("RGB"))
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    edge = (cv2.Canny(g, 80, 160) > 0).mean()                      # 0..~0.3
    q = (a // 40).reshape(-1, 3)                                    # coarse colour quant
    ncol = len({tuple(c) for c in q[:: max(1, len(q) // 2000)]})
    var = g.std() / 128.0
    score = 1.5 + edge * 8 + min(ncol, 40) / 12 + var              # heuristic blend
    return int(np.clip(round(score), 1, 5))


def _dominant_colors(img: Image.Image, mask_border: bool = True) -> dict:
    """bg_hex from image border median; glyph_hex from the darkest cluster."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = a.shape[:2]
    border = np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3),
                             a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    # glyph guess = mean of the 15% pixels most distant from bg
    d = np.linalg.norm(a.reshape(-1, 3) - bg, axis=1)
    thr = np.quantile(d, 0.85)
    fg = a.reshape(-1, 3)[d >= thr].mean(axis=0) if (d >= thr).any() else bg
    hexf = lambda c: "#%02x%02x%02x" % tuple(int(np.clip(v, 0, 255)) for v in c)
    return {"bg_hex": hexf(bg), "glyph_hex": hexf(fg), "noise_hex": hexf((bg + fg) / 2)}


# --------------------------------------------------------------------------- #
# The director
# --------------------------------------------------------------------------- #
_ATTR_PROMPT = """You are analysing CAPTCHA images from ONE scheme to help recreate them.
Return ONLY a JSON object; for each field choose EXACTLY ONE of the listed values (output the value, not the list):

{
  "bg": one of ["pale","dark"],
  "bg_texture": one of ["solid","mosaic","noise","lines"],
  "serif": one of ["serif","sans","mono"],
  "weight": one of ["thin","regular","bold"],
  "case": one of ["upper","lower","mixed","digits"],
  "wave": one of ["none","mild","strong"],
  "noise": one of ["none","light","heavy"],
  "colored": one of [true, false]
}

Rules: "colored" = characters use vivid/multiple colours (not black/grey). "wave" = how wavy/rotated the baseline is. "noise" = amount of distractor lines/dots/arcs. Do NOT read or transcribe the text. Respond with ONLY the JSON."""

_ATTR_ALLOWED = {
    "bg": {"pale", "dark"},
    "bg_texture": {"solid", "mosaic", "noise", "lines"},
    "serif": {"serif", "sans", "mono"},
    "weight": {"thin", "regular", "bold"},
    "case": {"upper", "lower", "mixed", "digits"},
    "wave": {"none", "mild", "strong"},
    "noise": {"none", "light", "heavy"},
}
_DIFF_BUCKETS = ["very-easy", "easy", "medium", "hard", "very-hard"]
_DIFF_PROMPT = ("How hard is this CAPTCHA to faithfully recreate (font exoticness, warping, "
                "colour, noise)? Reply with exactly one word: very-easy, easy, medium, hard, "
                "or very-hard.")


def direct(images: list[Image.Image], *, use_vlm: bool = True) -> Spec:
    """Read a few real captchas -> a generation Spec. `use_vlm=False` = classical only."""
    ref = images[0]
    spec = Spec()
    spec.colors = _dominant_colors(ref)
    cls_diff = _classical_difficulty(ref)

    if use_vlm and vlm.available():
        obj = vlm.ask_json(_ATTR_PROMPT, images[:2], max_new_tokens=200, cache_tag="director")
        for k, allowed in _ATTR_ALLOWED.items():
            setattr(spec, k, vlm.validate_str(obj.get(k), allowed, getattr(spec, k)))
        spec.colored = bool(obj.get("colored", spec.colored)) if isinstance(obj.get("colored"), bool) \
            else str(obj.get("colored")).lower() == "true"
        bucket, _ = vlm.ask_choice(_DIFF_PROMPT, images[:1], _DIFF_BUCKETS, k=3)
        vlm_diff = _DIFF_BUCKETS.index(bucket) + 1
        spec.difficulty = int(round((vlm_diff + cls_diff) / 2))
        spec.raw = {"attrs": obj, "vlm_diff": vlm_diff, "cls_diff": cls_diff}
    else:
        spec.difficulty = cls_diff
        spec.raw = {"cls_diff": cls_diff, "vlm": False}

    # pixels beat the VLM for things pixels can measure directly: bg lightness,
    # and whether glyphs are actually coloured (chroma) vs grey.
    r, g, b = (int(spec.colors["bg_hex"][i:i + 2], 16) for i in (1, 3, 5))
    spec.bg = "pale" if (0.299 * r + 0.587 * g + 0.114 * b) > 128 else "dark"
    gr, gg, gb = (int(spec.colors["glyph_hex"][i:i + 2], 16) for i in (1, 3, 5))
    chroma = max(gr, gg, gb) - min(gr, gg, gb)
    spec.colored = bool(chroma > 40)   # grey glyph => not "coloured"
    return spec


# --------------------------------------------------------------------------- #
# CLI:  python -m director --images a.png b.png ...
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()
    imgs = [Image.open(p) for p in args.images]
    spec = direct(imgs, use_vlm=not args.no_vlm)
    print(json.dumps(asdict(spec), indent=2))
    print("php_cfg:", json.dumps(spec.to_php_cfg(200, 60, ["<font>"]), indent=2))
