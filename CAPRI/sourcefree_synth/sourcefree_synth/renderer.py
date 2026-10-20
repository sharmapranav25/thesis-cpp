"""Procedural pixels only. Every retry belongs to the same label and configuration."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .fonts import get_font


class QualityError(RuntimeError):
    pass


def default_parameters(font_index=0):
    return {"font_index": font_index, "background": "blocks", "blocks": 16, "grid_step": 12,
            "background_strength": 0.35, "dots": 0.004, "lines": 1,
            "height": 0.65, "width": 0.82, "rotation": 12.0, "jitter": 0.035,
            "gap": 0.08, "stroke": 0, "outline": False, "shadow": False,
            "multicolor": True, "wave_x": 0.015, "wave_y": 0.035,
            "period": 1.5, "blur": 0.0}


def mutate_parameters(base, stage, rng, font_count, hint=None):
    p = dict(base)
    if stage in {"background", "joint"}:
        backgrounds = ["flat", "blocks", "grid", "noise", "blocks_grid"]
        if hint and hint.get("background") in backgrounds:
            backgrounds += [hint["background"]] * 2
        p.update(background=str(rng.choice(backgrounds)), blocks=int(rng.integers(4, 36)),
                 grid_step=int(rng.integers(6, 23)), background_strength=float(rng.uniform(0.1, 0.8)),
                 dots=float(rng.uniform(0, 0.014)), lines=int(rng.integers(0, 4)))
    if stage in {"typography", "joint"}:
        p.update(font_index=int(rng.integers(font_count)), height=float(rng.uniform(0.45, 0.88)),
                 width=float(rng.uniform(0.64, 0.94)), gap=float(rng.uniform(-0.03, 0.22)),
                 stroke=int(rng.choice([0, 0, 0, 1])), outline=bool(rng.random() < 0.2),
                 shadow=bool(rng.random() < 0.25), multicolor=bool(rng.random() < 0.5))
        if hint:
            for key in ("outline", "shadow", "multicolor"):
                if key in hint and rng.random() < 0.6:
                    p[key] = hint[key]
    if stage in {"geometry", "joint"}:
        p.update(rotation=float(rng.uniform(0, 27)), jitter=float(rng.uniform(0.005, 0.09)),
                 wave_x=float(rng.uniform(0, 0.07)), wave_y=float(rng.uniform(0, 0.10)),
                 period=float(rng.uniform(0.6, 3.5)), blur=float(rng.choice([0, 0, 0.3, 0.6])))
        if hint and hint.get("distortion") == "low" and rng.random() < 0.6:
            p["rotation"] *= 0.35
            p["wave_x"] *= 0.35
            p["wave_y"] *= 0.35
    return p


def _color(values):
    return tuple(int(v) for v in np.clip(values, 0, 255))


def _background(profile, params, rng):
    w, h = profile["width"], profile["height"]
    palette = profile["background"]
    base = np.array(palette[0], dtype=float)
    image = Image.new("RGB", (w, h), _color(base))
    draw = ImageDraw.Draw(image)
    strength = params["background_strength"]
    if params["background"] in {"blocks", "blocks_grid"}:
        for _ in range(params["blocks"]):
            color = np.array(palette[int(rng.integers(len(palette)))])
            color = base * (1 - strength) + color * strength + rng.normal(0, 5, 3)
            x, y = int(rng.integers(w)), int(rng.integers(h))
            draw.rectangle((x, y, x + int(rng.integers(4, max(5, w // 3))),
                            y + int(rng.integers(4, max(5, h)))), fill=_color(color))
    if params["background"] in {"grid", "blocks_grid"}:
        step = params["grid_step"]
        offset = int(rng.integers(step))
        color = _color(base + rng.choice([-1, 1]) * strength * 70)
        for x in range(offset, w, step):
            draw.line((x, 0, x, h), fill=color)
        for y in range(offset, h, step):
            draw.line((0, y, w, y), fill=color)
    if params["background"] == "noise":
        array = np.asarray(image).astype(float)
        image = Image.fromarray(np.clip(array + rng.normal(0, strength * 35, (h, w, 1)), 0, 255).astype("uint8"))
    return image


def _glyph(char, font, stroke, rotation):
    bbox = font.getbbox(char, stroke_width=stroke)
    if bbox is None:
        raise QualityError("Glyph has no bounding box")
    canvas = Image.new("L", (bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12), 0)
    ImageDraw.Draw(canvas).text((6 - bbox[0], 6 - bbox[1]), char, font=font, fill=255,
                               stroke_width=stroke, stroke_fill=255)
    canvas = canvas.rotate(rotation, resample=Image.Resampling.BILINEAR, expand=True)
    bbox = canvas.getbbox()
    if bbox is None:
        raise QualityError("Empty glyph")
    return canvas.crop(bbox)


def _warp(array, x_map, y_map):
    return array[y_map, x_map]


def render_once(label, params, profile, fonts, seed, trace=None):
    rng = np.random.default_rng(seed)
    w, h = profile["width"], profile["height"]
    bg = _background(profile, params, rng)
    if trace is not None:
        trace['01_background'] = bg.copy()
    image = bg.copy()
    font_path = fonts[params["font_index"]]["path"]
    size = max(10, round(h * params["height"]))
    font = get_font(font_path, size)
    glyphs = [_glyph(c, font, params["stroke"], float(rng.uniform(-params["rotation"], params["rotation"])))
              for c in label]
    max_height = max(g.height for g in glyphs)
    height_scale = h * params["height"] / max_height
    gap = params["gap"] * h
    nominal_width = sum(g.width * height_scale for g in glyphs) + gap * (len(label) - 1)
    width_scale = min(1.5, w * params["width"] / max(nominal_width, 1))
    x = (w - nominal_width * width_scale) / 2
    masks = []
    colors = []
    foreground = profile["foreground"]
    fixed_color = foreground[int(rng.integers(len(foreground)))]
    for glyph in glyphs:
        gw = max(1, round(glyph.width * height_scale * width_scale))
        gh = max(1, round(glyph.height * height_scale))
        glyph = glyph.resize((gw, gh), Image.Resampling.BILINEAR)
        y = round((h - gh) / 2 + rng.uniform(-params["jitter"], params["jitter"]) * h)
        mask = Image.new("L", (w, h), 0)
        mask.paste(glyph, (round(x), y))
        # Fail closed before distortion when layout would remove any substantial glyph mass.
        if np.asarray(mask).sum() < np.asarray(glyph).sum() * 0.97:
            raise QualityError("Glyph clipped during layout")
        color = foreground[int(rng.integers(len(foreground)))] if params["multicolor"] else fixed_color
        color = _color(color)
        if params["shadow"]:
            shadow = Image.new("L", (w, h), 0)
            shadow.paste(mask, (2, 2))
            image.paste((80, 80, 80), mask=shadow)
        if params["outline"]:
            outline = mask.filter(ImageFilter.MaxFilter(3))
            outline_color = (245, 245, 245) if sum(color) < 384 else (15, 15, 15)
            image.paste(outline_color, mask=outline)
        image.paste(color, mask=mask)
        masks.append(np.asarray(mask).copy())
        colors.append(color)
        x += (glyph.width / width_scale + gap) * width_scale
    if trace is not None:
        trace['02_text_layout'] = image.copy()
    yy, xx = np.indices((h, w))
    phase_x, phase_y = rng.uniform(0, 2 * math.pi, 2)
    x_map = np.clip(np.rint(xx + h * params["wave_x"] * np.sin(
        2 * math.pi * params["period"] * yy / h + phase_x)), 0, w - 1).astype(int)
    y_map = np.clip(np.rint(yy + h * params["wave_y"] * np.sin(
        2 * math.pi * params["period"] * xx / w + phase_y)), 0, h - 1).astype(int)
    image = Image.fromarray(_warp(np.asarray(image), x_map, y_map))
    reference = Image.fromarray(_warp(np.asarray(bg), x_map, y_map))
    warped_masks = [_warp(m, x_map, y_map) for m in masks]
    if trace is not None:
        trace['03_warp'] = image.copy()
    # Apply identical interference to text image and background-only reference, allowing
    # the visibility gate to measure text's contribution instead of noise contrast.
    draws = [ImageDraw.Draw(im) for im in (image, reference)]
    for _ in range(params["lines"]):
        points = tuple(int(v) for v in [rng.integers(w), rng.integers(h), rng.integers(w), rng.integers(h)])
        color = _color(rng.integers(30, 226, 3))
        for draw in draws:
            draw.line(points, fill=color, width=1)
    for _ in range(round(w * h * params["dots"])):
        point = (int(rng.integers(w)), int(rng.integers(h)))
        color = _color(rng.integers(0, 256, 3))
        for draw in draws:
            draw.point(point, fill=color)
    if params["blur"]:
        image = image.filter(ImageFilter.GaussianBlur(params["blur"]))
        reference = reference.filter(ImageFilter.GaussianBlur(params["blur"]))
    if trace is not None:
        trace['04_interference'] = image.copy()
        trace['05_background_reference'] = reference.copy()
        trace['06_text_mask'] = Image.fromarray(np.max(warped_masks, axis=0)).convert('RGB')
    delta = np.max(np.abs(np.asarray(image).astype(float) - np.asarray(reference).astype(float)), axis=2)
    binary = np.stack([m > 100 for m in warped_masks])
    union_count = binary.sum(0)
    checks = []
    for i, mask in enumerate(binary):
        ys, xs = np.where(mask)
        if not len(xs):
            raise QualityError(f"Character {i} disappeared")
        width, height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        visible = float(np.mean(delta[mask] >= 12))
        overlap = float(np.mean(union_count[mask] > 1))
        retention = float(warped_masks[i].sum() / max(masks[i].sum(), 1))
        on_edge = bool(np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))
        if width < 2 or height < 6 or visible < 0.55 or overlap > 0.4 or retention < 0.7 or on_edge:
            raise QualityError(f"Character {i}: width={width}, height={height}, visible={visible:.2f}, "
                               f"overlap={overlap:.2f}, retention={retention:.2f}, touches_edge={on_edge}")
        checks.append({"bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                       "visible_fraction": visible, "overlap_fraction": overlap, "area_retention": retention})
    return image, Image.fromarray(np.max(warped_masks, axis=0)), warped_masks, {
        "label": label, "render_seed": int(seed), "font": font_path, "font_sha256": fonts[params["font_index"]]["sha256"],
        "profile_id": profile["id"], "parameters": dict(params), "colors": colors, "quality": checks}


def render_checked(label, params, profile, fonts, seed, retries=8, trace=None):
    if not label or any(not c.isascii() or not c.isalnum() for c in label):
        raise ValueError("Labels must contain ASCII letters/digits only")
    errors = []
    for attempt in range(retries):
        # No best-image fallback, label resampling, schema substitution or persistent state.
        attempt_seed = int(np.random.SeedSequence([seed, attempt]).generate_state(1)[0])
        try:
            if trace is not None:
                trace.clear()
            result = render_once(label, params, profile, fonts, attempt_seed, trace=trace)
            result[3]["sample_seed"] = int(seed)
            result[3]["attempt"] = attempt + 1
            result[3]["rejected_attempts"] = errors
            return result
        except QualityError as exc:
            errors.append(str(exc))
    raise QualityError(f"All {retries} attempts rejected for the same label/configuration: {errors[-1]}")
