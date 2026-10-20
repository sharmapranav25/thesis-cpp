"""Static iteration-2 dataset pilot. No training or curriculum.

Run from the sourcefree_synth root. Outputs are immutable new run directories.
Requested quotas and accepted label distributions are reported separately.
"""
import argparse
import gzip
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from feature_bank.iter2 import generate as G
from captcha_effects import render as R
from captcha_effects.effects import REGISTRY
from captcha_effects import text as T
from captcha_effects.captcha import render_captcha

VERSION = "readable_static_corpus_v7_fast"
APPEARANCE = ["neutral", "per_char_rotation", "vertical_jitter", "size_variation",
              "text_on_curve", "slant", "overlap_tight_spacing", "wide_spacing",
              "font_mixture", "multicolor_text", "two_layer"]
# Condensed/expanded are consequences of width fitting, not independently toggled.
DAMAGING = {"thin", "hollow", "translucent", "fill_texture", "strike", "noise_over",
            "occluders", "cuts", "wave", "morph", "perspective", "swirl", "blur",
            "pixelate", "resample", "grain", "jpeg", "quantize", "invert"}
GEOMETRIC = {"per_char_rotation", "vertical_jitter", "size_variation", "text_on_curve",
             "slant", "overlap_tight_spacing", "two_layer"}
_BANK = None


def bank():
    global _BANK
    if _BANK is None:
        _BANK = T.FontBank()
    return _BANK


def schedule(n, seed, font_split="train", split="smoke"):
    """Balanced fixed ticket list; rejected attempts retain their ticket.

    The CNN list is balanced by head, not by effect-function name (scribbles has two).
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for name in R.SAMPLED:
        groups.setdefault(REGISTRY[name]["head"], []).append(name)
    counts = Counter()
    axes = {}
    for name, vals in [("length", list(range(2, 9))), ("category", G.CATEGORIES),
                       ("content", ["digits", "upper", "lower", "mixed"]),
                       ("appearance", APPEARANCE), ("weight", list(range(11))),
                       ("width", list(range(11))), ("italic", [False, False, False, True])]:
        arr = [vals[i % len(vals)] for i in range(n)]
        rng.shuffle(arr)
        axes[name] = arr
    # Two-layer is a real two-row render only for lengths >=4 in the renderer.
    for i in range(n):
        if axes["appearance"][i] == "two_layer" and axes["length"][i] < 4:
            j = next(j for j in range(n) if axes["length"][j] >= 4 and axes["appearance"][j] != "two_layer")
            axes["appearance"][i], axes["appearance"][j] = axes["appearance"][j], axes["appearance"][i]
    supported_xh = sorted(k for k, v in Counter(T.x_height_level(f) for f in bank().fonts if f["split"] == font_split).items()
                          if k is not None and v >= 20)
    # Coupled integer quotas. A font-mixture ticket needs >=2 fonts of the
    # requested category, x-height and italic status. No fallback changes labels.
    from scipy.optimize import milp, LinearConstraint, Bounds
    cells = []
    for cat in G.CATEGORIES:
        fonts = bank().pool(font_split, category=cat)
        for lv in supported_xh:
            if sum(T.x_height_level(f) == lv for f in fonts) < 3: continue
            for italic in [False, True]:
                count = sum(T.x_height_level(f) == lv and bool(f.get("italic")) == italic for f in fonts)
                for mixture in [False, True]:
                    if count >= (2 if mixture else 1): cells.append((cat, lv, italic, mixture, count))
    cat_mix = Counter((cat, appearance == "font_mixture") for cat, appearance in zip(axes["category"], axes["appearance"]))
    b_xh = Counter(supported_xh[i % len(supported_xh)] for i in range(n))
    pairs = [(cat, mixture) for cat in G.CATEGORIES for mixture in [False, True]]
    constraints = [[int(cat == cell[0] and mixture == cell[3]) for cell in cells] for cat, mixture in pairs]
    constraints += [[int(lv == cell[1]) for cell in cells] for lv in supported_xh]
    constraints += [[int(cell[2]) for cell in cells]]
    margins = [cat_mix[pair] for pair in pairs] + [b_xh[lv] for lv in supported_xh] + [sum(axes["italic"])]
    solution = milp([-np.log(cell[4]) for cell in cells], integrality=np.ones(len(cells)),
                    bounds=Bounds(0, n), constraints=LinearConstraint(constraints, margins, margins),
                    options={"time_limit": 30})
    if not solution.success:
        raise RuntimeError("Category/x-height/italic quota allocation infeasible: " + solution.message)
    allocation = np.rint(solution.x).astype(int)
    if not np.array_equal(np.array(constraints) @ allocation, np.array(margins)):
        raise RuntimeError("Coupled allocation did not yield exact integer quotas")
    assigned = {pair: [] for pair in pairs}
    for (cat, lv, italic, mixture, _), count in zip(cells, allocation):
        assigned[cat, mixture] += [(int(lv), bool(italic))] * int(count)
    for values in assigned.values(): rng.shuffle(values)
    coupled = [assigned[cat, appearance == "font_mixture"].pop() for cat, appearance in zip(axes["category"], axes["appearance"])]
    axes["x_height"] = [x[0] for x in coupled]
    axes["italic"] = [x[1] for x in coupled]
    tickets = []
    for i in range(n):
        k = 3 + i % 3
        chosen = []
        heads = list(groups)
        for _ in range(k):
            rng.shuffle(heads)
            options = sorted(heads, key=lambda h: counts[h])
            for h in options:
                if h in [REGISTRY[x]["head"] for x in chosen]:
                    continue
                names = list(groups[h]); rng.shuffle(names)
                good = [x for x in names if not any(x in grp and any(y in grp for y in chosen)
                                                   for grp in R.EXCLUSIVE_K)]
                if good:
                    chosen.append(good[0]); counts[h] += 1
                    break
            else:
                raise RuntimeError("Cannot satisfy effect-count ticket")
        rng.shuffle(chosen)
        ticket = {key: value[i] for key, value in axes.items()}
        ticket.update(index=i, effects=chosen, k=k, seed=seed, font_split=font_split, split=split)
        tickets.append(ticket)
    return tickets, dict(counts)


@lru_cache(maxsize=65536)
def fitted_glyph_ratio(font_key, ch, size, frac, stroke, xs, slant):
    meta = bank().by_key[font_key]
    axes = [a for a in meta["axes"] if a["tag"] == "wght"]
    weight = None if frac is None else axes[0]["min"] + frac * (axes[0]["max"] - axes[0]["min"])
    mask = T.glyph_mask(meta, ch, size, ss=2, weight=weight, stroke=stroke,
                        x_scale=xs, slant=slant)
    return None if mask is None else T.stroke_width_px(mask) / max(1, mask.shape[0])


def transformed_weight(meta, text, size, target, xs, slant):
    """Fit transformed glyphs using endpoints/interpolation, then optional stroke."""
    axes = [a for a in meta["axes"] if a["tag"] == "wght"]
    wanted = T.weight_target(target)
    measured = []
    def probe(frac, stroke):
        ratios = [fitted_glyph_ratio(meta["key"], c, size, frac, stroke, xs, slant) for c in text[:3]]
        if any(value is None for value in ratios):
            return None
        value = float(np.median(ratios))
        measured.append((abs(value - wanted), frac, stroke))
        return value
    if axes:
        lo, hi = probe(0., 0), probe(1., 0)
        if lo is None or hi is None:
            return None
        if min(lo, hi) < wanted < max(lo, hi):
            probe(float((wanted - lo) / (hi - lo)), 0)
        elif wanted > max(lo, hi):
            frac = 1. if hi >= lo else 0.
            for stroke in [1, 2]:
                probe(frac, stroke)
                if min(item[0] for item in measured) <= .00625:
                    break
    else:
        for stroke in [0, 1, 2]:
            probe(None, stroke)
            if measured and min(item[0] for item in measured) <= .00625:
                break
    return min(measured, key=lambda item: item[0]) if measured else None


def text_spec(ticket, rng, height):
    b = bank()
    content = ticket["content"]
    text = T.make_text(rng, ticket["length"], content)
    if content == "mixed":
        # The label means both cases, not merely drawing from an alphanumeric alphabet.
        chars = list(text)
        chars[0] = str(rng.choice(list(T.UPPER)))
        chars[1] = str(rng.choice(list(T.LOWER)))
        rng.shuffle(chars); text = "".join(chars)
    spec = T.sample_text_spec(rng, b, split=ticket.get("font_split", "train"), length=ticket["length"], charset=content,
                              layout="line", font_mix=1)
    spec.update(text=text, size=.62, size_jitter=0., rotation=0., jitter=0., spacing=.02,
                curve=0., slant=0., multicolour=False, palette="contrast", min_contrast=100.)
    active = ticket["appearance"]
    level = int(rng.integers(1, 4 if ticket.get("revision", 0) else 11))
    # Sampling spans the range; accepted measured levels are the audit authority.
    if active == "per_char_rotation": spec["rotation"] = 3 + 4.2 * level
    elif active == "vertical_jitter": spec["jitter"] = .015 * level
    elif active == "size_variation": spec["size_jitter"] = .04 * level
    elif active == "text_on_curve": spec.update(layout="curve", curve=.02 * level)
    elif active == "slant": spec["slant"] = float(rng.choice([-1, 1])) * .04 * level
    elif active == "overlap_tight_spacing": spec["spacing"] = -.025 * level
    elif active == "wide_spacing": spec["spacing"] = .1 * level
    elif active == "multicolor_text": spec["multicolour"] = True
    elif active == "two_layer": spec.update(layout="two_layer", size=.45)
    pool = b.pool(ticket.get("font_split", "train"), category=ticket["category"], caseless=content in ["digits", "upper"])
    near = [f for f in pool if T.x_height_level(f) == ticket["x_height"]]
    if not near: raise ValueError("Impossible category/x-height cell")
    pool = near
    preferred = [f for f in pool if bool(f.get("italic")) == ticket["italic"]]
    minimum_fonts = 2 if active == "font_mixture" else 1
    if len(preferred) < minimum_fonts: raise ValueError("Impossible category/x-height/italic ticket")
    pool = preferred
    candidates = list(rng.choice(pool, size=min(8, len(pool)), replace=False))
    best = None
    for f in candidates:
        wanted = .38 + .077 * ticket["width"]
        xs = float(np.clip(wanted / (f.get("width_rel") or .8), .6, 1.45))
        fit = transformed_weight(f, text, height * spec["size"], ticket["weight"], xs, spec["slant"])
        if fit is None: continue
        err = fit[0] / .025 + abs((f.get("width_rel") or .8) * xs - wanted) / .077
        if best is None or err < best[0]: best = (err, f, xs, fit)
        if best[0] <= .5:
            break
    if best is None: raise ValueError("No usable font")
    _, f, xs, (_, frac, stroke) = best
    fonts = [f["key"]]
    if active == "font_mixture":
        others = [g["key"] for g in pool if g["key"] != f["key"]]
        if others: fonts.append(str(rng.choice(others)))
    spec.update(fonts=fonts, weight=frac, bold_stroke=stroke, x_scale=xs,
                font_targets=dict(category=ticket["category"], weight=ticket["weight"],
                                  width=ticket["width"], slant=level if active == "slant" else 0,
                                  italic=ticket["italic"], x_height=ticket["x_height"]))
    return spec, level


def readability(out):
    """Conservative per-glyph engineering checks; human review remains required."""
    rgb = out["image"].astype(float)
    lum = rgb @ np.array([.299, .587, .114])
    records = [r for r in out["records"] if r["head"] == "text"]
    contrasts, heights, strokes = [], [], []
    for rec in records:
        m = out["opaque"] == rec["id"]
        ys, xs = np.nonzero(m)
        if not len(xs): return False, {"reason": "missing_glyph"}
        ring = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))) > 0
        ring &= ~m
        contrast = abs(float(np.median(lum[m])) - float(np.median(lum[ring]))) if ring.any() else 0.
        contrasts.append(contrast); heights.append(int(ys.max() - ys.min() + 1))
        strokes.append(T.stroke_width_px(m[ys.min():ys.max()+1, xs.min():xs.max()+1]))
    retained = out["text"]["legibility"]["min_visible"]
    clipped = max(r["params"].get("clipped_fraction", 0.) for r in records)
    report = dict(min_glyph_contrast=round(min(contrasts), 2),
                  min_post_glyph_effect_retained=retained, max_clipped=clipped,
                  min_glyph_height=min(heights), min_final_stroke=round(min(strokes), 2))
    ok = min(contrasts) >= 70 and retained >= .8 and min(heights) >= 12 and clipped <= .05 and min(strokes) >= 1.2
    return ok, report


def render_ticket(ticket):
    ticket = dict(ticket)
    rejected = Counter()
    for attempt in range(ticket.get("max_attempts", 24)):
        rng = np.random.default_rng([ticket["seed"], ticket["index"], attempt, 867 + ticket.get("revision", 0)])
        height = int(rng.integers(65, 86))
        size = (int(height * rng.uniform(3.8, 5.0)), height)
        try:
            ts, text_level = text_spec(ticket, rng, height)
            levels = {name: int(rng.integers(1, 5)) for name in ticket["effects"]}
            focus = ticket["effects"][attempt % len(ticket["effects"]) ]
            # One effect spans the ladder; the remaining ingredients are mild.
            levels[focus] = 1 + ((ticket["index"] // 3 + attempt) % 10)
            if ticket["appearance"] in GEOMETRIC and text_level >= 6:
                for name in levels:
                    if name in DAMAGING: levels[name] = min(levels[name], 3)
            if ticket.get("revision", 0):
                for name in levels:
                    if name in DAMAGING: levels[name] = min(levels[name], 2)
            # Whole-canvas fields must precede local shapes, otherwise they erase their labels.
            ordered = sorted(ticket["effects"], key=lambda name: 0 if name in ["gradient", "texture", "pattern"] else 1)
            es = dict(background="random", effects=[dict(name=name, level=levels[name]) for name in ordered])
            out = render_captcha(rng, bank(), size, effects_spec=es, text_spec=ts)
            if out is None or not out["text"]["usable"]:
                rejected["renderer_guard"] += 1; continue
            ok, check = readability(out)
            if not ok: rejected["per_glyph_guard"] += 1; continue
            actual = {h for h, e in out["effects"].items() if e["present"]}
            intended = {REGISTRY[name]["head"] for name in ticket["effects"]}
            if actual != intended:
                rejected["effect_visibility"] += 1; continue
            label = out["text"]
            active = ticket["appearance"]
            if active != "neutral" and not label["heads"][active]["present"]:
                rejected["text_focus_invisible"] += 1; continue
            if len(label["text"]) != ticket["length"] or label["style"]["dominant_category"] != ticket["category"]:
                rejected["text_ticket_mismatch"] += 1; continue
            if label["style"]["median_x_height"] != ticket["x_height"]:
                rejected["x_height_ticket_mismatch"] += 1; continue
            if label["style"]["italic_any"] != ticket["italic"]:
                rejected["italic_ticket_mismatch"] += 1; continue
            head = {"digits": "digits_only", "upper": "uppercase_only", "lower": "lowercase_only",
                    "mixed": "mixed_case_alnum"}[ticket["content"]]
            if not label["heads"][head]["present"]:
                rejected["content_ticket_mismatch"] += 1; continue
            r = G.pack(out, ticket.get("split", "smoke"), ticket["index"], [ticket["seed"], ticket["index"], attempt, 867 + ticket.get("revision", 0)], size, attempt)
            r["label"]["quality"] = check
            r["label"]["ticket"] = ticket
            r["label"]["requested_effect_levels"] = levels
            # Hash exactly what the trainer sees, not the pre-resize image.
            r["label"]["sha256"] = hashlib.sha256(r["image"].tobytes()).hexdigest()
            r["rejections"] = dict(rejected)
            return r
        except (ValueError, OSError) as exc:
            rejected[type(exc).__name__] += 1
    return dict(failed=True, index=ticket["index"], ticket=ticket, rejections=dict(rejected))


def write_cache(root, rows, planned, tickets, seconds, cache_name="smoke"):
    cache = root / "iter2_h96" / cache_name
    cache.mkdir(parents=True)
    good = [r for r in rows if not r["failed"]]
    if not good: raise RuntimeError("No accepted images")
    heads = G.effect_heads()
    labels = [G.label_rows(r["label"], heads) for r in good]
    widths = np.array([r["image"].shape[1] for r in good], np.int32)
    offsets = np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64)
    np.ascontiguousarray(np.concatenate([r["image"] for r in good], axis=1).transpose(1, 0, 2)).tofile(cache / "images.u8")
    values = dict(offsets=offsets, widths=widths, native=np.array([r["label"]["size"] for r in good], np.int16))
    for j, (key, dt) in enumerate([("eff_present", np.int8), ("eff_level", np.int8),
                                  ("txt_present", np.int8), ("txt_level", np.int8),
                                  ("chars", np.int16), ("length", np.int8),
                                  ("font", np.int8), ("glyph", np.int8), ("glyph_box", np.float16)]):
        values[key] = np.array([row[j] for row in labels], dtype=dt)
    np.savez_compressed(cache / "index.npz", **values)
    with gzip.open(cache / "labels.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in good: f.write(json.dumps(r["label"]) + "\n")
    meta = dict(n=len(good), requested=len(rows), h=96, effect_heads=heads, text_heads=G.TEXT_HEADS,
                font_cols=["cat", "stroke", "weight", "width", "x_height", "slant", "italic"],
                sha256=[r["label"]["sha256"] for r in good], version=VERSION, scheme=VERSION,
                seconds=round(seconds, 2), usable_rate=len(good) / sum((r["label"]["attempt"] + 1) if not r["failed"] else r["ticket"].get("max_attempts", 60) for r in rows),
                sampler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                font_meta_sha256=hashlib.sha256((G.CE_ROOT / "fonts/font_meta.json").read_bytes()).hexdigest(),
                text_renderer_sha256=hashlib.sha256((G.CE_ROOT / "captcha_effects/text.py").read_bytes()).hexdigest(),
                chars=G.CHARS, categories=G.CATEGORIES, font_split=tickets[0].get("font_split", "train"), split=tickets[0].get("split", "smoke"))
    (cache / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report = dict(version=VERSION, requested=len(rows), accepted=len(good),
                  failures=[r for r in rows if r["failed"]], planned_effect_positives=planned,
                  rejections=dict(sum((Counter(r["rejections"]) for r in rows), Counter())),
                  measured_effect_positives={h: int(values["eff_present"][:, i].sum()) for i, h in enumerate(heads)},
                  measured_text_positives={h: int(values["txt_present"][:, i].sum()) for i, h in enumerate(G.TEXT_HEADS)},
                  effect_levels={h: np.bincount(values["eff_level"][:, i], minlength=11).tolist() for i, h in enumerate(heads)},
                  text_levels={h: np.bincount(values["txt_level"][:, i][values["txt_level"][:, i] >= 0], minlength=11).tolist() for i, h in enumerate(G.TEXT_HEADS)},
                  font_levels={h: dict(Counter(values["font"][:, j].tolist())) for j, h in enumerate(meta["font_cols"])},
                  length=dict(Counter(values["length"].tolist())),
                  content=dict(Counter(r["label"]["ticket"]["content"] for r in good)),
                  character_tokens=dict(Counter(c for r in good for c in r["label"]["text"]["text"])),
                  duplicates=len(good) - len(set(meta["sha256"])),
                  target_pm1={key: float(np.mean([abs(r["label"]["ticket"][key] - r["label"]["text"]["style"]["median_" + key]) <= 1 for r in good]))
                              for key in ["weight", "width", "x_height"]},
                  warning="Pilot only. Final-image feature visibility and font/appearance quota feasibility require review. No full-corpus approval.")
    report["gates"] = dict(all_tickets_filled=len(good) == len(rows), unique_images=report["duplicates"] == 0,
                           weight_target_pm1=report["target_pm1"]["weight"] >= .8,
                           font_weight_all_levels=all(int((values["font"][:, 2] == lv).sum()) >= .05 * len(good) for lv in range(11)),
                           x_height_exact=report["target_pm1"]["x_height"] == 1,
                           final_visibility_validated=False, user_visual_review=False)
    report["full_corpus_ready"] = all(report["gates"].values())
    (root / ("balance_report.json" if cache_name == "smoke" else cache_name + "_balance_report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / ("tickets.json" if cache_name == "smoke" else cache_name + "_tickets.json")).write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    # Individual images, at 3x native size, with labels beneath. No contact sheet.
    review = root / ("review" if cache_name == "smoke" else cache_name + "_review"); review.mkdir()
    selected = []
    for mode in APPEARANCE:
        candidates = [r for r in good if r["label"]["ticket"]["appearance"] == mode]
        if candidates:
            selected.append(candidates[0])
    for name in ["blur", "thin", "occluders", "cuts", "wave", "hollow", "pixelate", "resample"]:
        candidates = [r for r in good if name in r["label"]["ticket"]["effects"]]
        if candidates:
            selected.append(max(candidates, key=lambda r: r["label"]["requested_effect_levels"][name]))
    for pos, r in enumerate(selected):
        label = r["label"]; native = Image.fromarray(r["image"]).resize(tuple(label["size"]), Image.Resampling.LANCZOS)
        im = native.resize((native.width * 3, native.height * 3), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (max(1050, im.width), im.height + 130), "white")
        canvas.paste(im, (0, 0)); draw = ImageDraw.Draw(canvas)
        lines = [f"Answer: {label['text']['text']} | text focus: {label['ticket']['appearance']}",
                 "Effects: " + ", ".join(f"{h}={e['level']}" for h, e in label["effects"].items() if e["present"]),
                 f"Font: {label['text']['style']['dominant_category']} | weight={label['text']['style']['median_weight']} | width={label['text']['style']['median_width']}",
                 "Checks: " + json.dumps(label["quality"])]
        # Separate metadata preserves complete labels even if a long line exceeds the preview.
        for j, line in enumerate(lines): draw.text((8, im.height + 8 + j * 24), line, fill="black")
        canvas.save(review / f"{pos:02d}_{label['id']}.png")
        (review / f"{pos:02d}_{label['id']}.json").write_text(json.dumps(label, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=420)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=False)
    tickets, planned = schedule(a.n, a.seed)
    rows = []
    started = time.monotonic()
    with ProcessPoolExecutor(a.workers) as pool:
        for i, row in enumerate(pool.map(render_ticket, tickets, chunksize=1)):
            rows.append(row)
            if (i + 1) % 20 == 0: print(f"Tickets {i+1}/{a.n}; accepted {sum(not r['failed'] for r in rows)}", flush=True)
    report = write_cache(root, rows, planned, tickets, time.monotonic() - started)
    print(json.dumps({k: report[k] for k in ["requested", "accepted", "duplicates", "target_pm1", "rejections"]}), flush=True)


if __name__ == "__main__":
    main()
