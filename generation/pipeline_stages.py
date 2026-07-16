"""Visualise the pipeline stage-by-stage for ONE real captcha.

Lays out the 7 architecture steps with their real intermediate artifacts:
  [1]  real input captcha
  [2]  director spec (difficulty + bg/colour/distortion knobs, from VLM + pixels)
        --- pipeline splits ---
  [3a] background extracted (glyphs masked + inpainted)     [3b] glyph crops extracted
  [4a] background GENERATED (php, VLM-guided knobs)         [4b] glyph source (real-crop bank)
        --- branches merge ---
  [5]  COMPOSE -> synthetic captcha + label

  python -m pipeline_stages --scheme cryptbb
"""
from __future__ import annotations

import argparse
import glob
import random
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import compose
import director
import extract
import pipeline

HERE = Path(__file__).resolve().parent


def _font(sz=14):
    for p in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def _bg_plate(img: Image.Image) -> Image.Image:
    """3a artifact: mask the glyph components and inpaint -> background only."""
    gray = np.asarray(img.convert("L"))
    best = None
    for inv in (False, True):
        labels, comps = extract._components(gray, inv)
        if comps and (best is None or len(comps) > len(best[1])):
            best = (labels, comps)
    mask = np.zeros(gray.shape, np.uint8)
    if best:
        for (x, y, w, h, idx) in best[1]:
            mask[y:y + h, x:x + w] = ((best[0][y:y + h, x:x + w] == idx) * 255).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), 1)
    rgb = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    inp = cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))


def _swatch(hexcol: str, s=22) -> Image.Image:
    im = Image.new("RGB", (s, s), hexcol if hexcol.startswith("#") else "#888888")
    ImageDraw.Draw(im).rectangle([0, 0, s - 1, s - 1], outline="black")
    return im


def _tile(imgs, cw, ch, gap=4):
    """horizontal strip of images scaled to height ch."""
    scaled = []
    for im in imgs:
        s = ch / im.height
        scaled.append(im.convert("RGB").resize((max(4, int(im.width * s)), ch)))
    W = sum(i.width + gap for i in scaled) + gap
    strip = Image.new("RGB", (max(cw, W), ch), "white")
    x = gap
    for i in scaled:
        strip.paste(i, (x, 0)); x += i.width + gap
    return strip


def build(scheme: str, use_vlm: bool = True, seed: int = 0):
    cfg = pipeline.SCHEMES[scheme]
    rng = random.Random(seed)
    paths = pipeline._list_images(cfg["dir"])
    ref = Image.open(paths[seed % len(paths)]).convert("RGB")
    W, H = pipeline._median_canvas(paths)

    # [2] director
    spec = director.direct([Image.open(p) for p in paths[:6]], use_vlm=use_vlm)
    spec.length_modes = extract.length_modes(paths, expected=cfg["expected"])

    # [3a] bg plate, [3b] glyph crops
    plate = _bg_plate(ref)
    crops = extract.char_crops(ref, cfg["expected"])[:8]

    # [4a] generated bg
    outd = HERE / "out" / f"_stages_{scheme}"; outd.mkdir(parents=True, exist_ok=True)
    bgs = pipeline._render_bg_pool(spec, W, H, outd, 4)
    gen_bg = Image.open(bgs[0]).convert("RGB")

    # [4b] glyph source (real-crop bank) + [5] compose
    bank = compose.bank_for(cfg["bank"]) if Path(cfg["bank"], "alphabet_manifest.json").exists() else None
    bank_glyphs, synth = [], None
    if bank:
        chars = [rng.choice(bank.observed_chars) for _ in range(8)]
        bank_glyphs = [compose.to_alpha_smart(bank.render_char(c, seed=rng.randint(0, 1 << 30)))
                       .convert("RGB") for c in chars]
        L = max(3, pipeline._sample_length(spec.length_modes, rng))
        text = compose.sample_text(bank, L, rng)
        synth = compose.compose_one(bank, gen_bg.copy(), text, rng)

    # ---- render the panel ----
    F, Fb = _font(14), _font(16)
    CW, CH = 320, 90          # captcha display size
    lab_w = 210
    pad = 14
    rows = []  # (label, [artifacts as pre-rendered images], height)

    def cap(im):
        s = min(CW / im.width, CH / im.height)
        return im.resize((int(im.width * s), int(im.height * s)))

    # spec text block
    sp = asdict(spec)
    spec_lines = [f"difficulty={sp['difficulty']}  bg={sp['bg']}  colored={sp['colored']}",
                  f"serif={sp['serif']} weight={sp['weight']} case={sp['case']}",
                  f"wave={sp['wave']} noise={sp['noise']} texture={sp['bg_texture']}",
                  f"len_modes={[ (l,round(w,2)) for l,w in sp['length_modes'][:4] ]}"]

    W_panel = lab_w + 2 * (CW + pad) + 3 * pad
    # compute layout height
    row_h = CH + 2 * pad
    n_rows = 6  # 1,2, (3a|3b), (4a|4b), split hdrs, 5
    H_panel = pad + (CH + 26) + row_h + 22 + row_h + row_h + 22 + row_h + pad
    panel = Image.new("RGB", (W_panel, H_panel + 40), "white")
    d = ImageDraw.Draw(panel)
    d.text((pad, 6), f"PIPELINE STAGES  —  {scheme}", font=Fb, fill="black")
    y = 34

    def label(txt, yy, sub=""):
        d.text((pad, yy), txt, font=Fb, fill="black")
        if sub:
            d.text((pad, yy + 20), sub, font=F, fill="#555555")

    # [1]
    label("[1] real input", y)
    panel.paste(cap(ref), (lab_w, y))
    y += CH + pad

    # [2]
    label("[2] director spec", y, "(VLM + pixels)")
    for i, ln in enumerate(spec_lines):
        d.text((lab_w, y + i * 18), ln, font=F, fill="black")
    for i, key in enumerate(["bg_hex", "glyph_hex", "noise_hex"]):
        sw = _swatch(sp["colors"][key])
        panel.paste(sw, (lab_w + 300 + i * 30, y + 4))
        d.text((lab_w + 300, y + 30), "bg / glyph / noise", font=_font(10), fill="#555")
    y += CH + pad

    d.line([(pad, y), (W_panel - pad, y)], fill="#cccccc");
    d.text((W_panel // 2 - 60, y - 2), "— split into 2 branches —", font=_font(11), fill="#999");
    y += 16

    # [3a | 3b]
    label("[3a] bg extracted", y, "glyphs removed")
    panel.paste(cap(plate), (lab_w, y))
    d.text((lab_w + CW + pad, y - 2), "[3b] glyph crops extracted", font=F, fill="black")
    if crops:
        panel.paste(_tile(crops, CW, 60), (lab_w + CW + pad, y + 16))
    y += CH + pad

    # [4a | 4b]
    label("[4a] bg GENERATED", y, "php, VLM knobs")
    panel.paste(cap(gen_bg), (lab_w, y))
    d.text((lab_w + CW + pad, y - 2), "[4b] glyph source (real-crop bank)", font=F, fill="black")
    if bank_glyphs:
        panel.paste(_tile(bank_glyphs, CW, 60), (lab_w + CW + pad, y + 16))
    y += CH + pad

    d.line([(pad, y), (W_panel - pad, y)], fill="#cccccc")
    d.text((W_panel // 2 - 45, y - 2), "— branches merge —", font=_font(11), fill="#999")
    y += 16

    # [5]
    label("[5] COMPOSE -> synth", y, "+ label (ours)")
    if synth is not None:
        panel.paste(cap(synth), (lab_w, y))

    out = HERE / "out" / f"stages_{scheme}.png"
    panel.save(out)
    print(f"[{scheme}] stages -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=list(pipeline.SCHEMES))
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.scheme, use_vlm=not args.no_vlm, seed=args.seed)
