"""Fit -> freeze -> render. Fitting never uses labels or validation-based selection."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import __version__
from .data import load_image, new_output, prepare_split, read_json, sha256_file, write_json
from .features import DistributionMetric, FeatureExtractor, cluster_styles, palette_profile
from .fonts import discover_fonts, verify_fonts
from .renderer import QualityError, default_parameters, mutate_parameters, render_checked


def validate_text_prior(charset, lengths):
    if not charset or len(set(charset)) != len(charset) or any(not c.isascii() or not c.isalnum() for c in charset):
        raise ValueError("charset must be unique ASCII letters/digits")
    if not lengths or len(set(lengths)) != len(lengths) or any(type(n) is not int or not 1 <= n <= 16 for n in lengths):
        raise ValueError("lengths must be distinct integers from 1 to 16")


def sample_label(rng, charset, lengths):
    return "".join(rng.choice(list(charset), int(rng.choice(lengths))))


def make_trials(count):
    # A baseline followed by staged proposals. Each stage preserves the current best.
    stages = ["background", "typography", "geometry", "joint"]
    return [stages[min(3, (i * 4) // (count - 1))] for i in range(count - 1)]


def render_probe(params, profiles, fonts, labels, seeds, profile_indices):
    images, errors = [], []
    for label, seed, index in zip(labels, seeds, profile_indices):
        try:
            image, _, _, _ = render_checked(label, params, profiles[int(index)], fonts, int(seed), retries=4)
            images.append(image)
        except QualityError as exc:
            errors.append(str(exc))
    return images, errors


def pareto_shortlist(rows, limit=3):
    valid = [r for r in rows if r["eligible"]]
    if not valid:
        return []
    keys = list(valid[0]["scores"])
    keys.remove("total")
    frontier = []
    for row in valid:
        dominated = any(all(other["scores"][k] <= row["scores"][k] for k in keys)
                        and any(other["scores"][k] < row["scores"][k] for k in keys) for other in valid)
        if not dominated:
            frontier.append(row)
    # Keep alternatives only when close to the best aggregate score; don't reward a single proxy at any cost.
    frontier.sort(key=lambda r: r["objective"])
    best = min(valid, key=lambda r: r["objective"])
    selected = [best]
    for row in frontier:
        if row["trial"] != best["trial"] and row["objective"] <= best["objective"] * 1.25 + 0.02:
            selected.append(row)
        if len(selected) == limit:
            break
    return selected


def _sheet(images, captions, output, columns=4, title=""):
    cell_w, cell_h = 340, 116
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h + 32), "#ececec")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for i, (image, caption) in enumerate(zip(images, captions)):
        x, y = (i % columns) * cell_w, (i // columns) * cell_h + 32
        thumb = image.copy()
        thumb.thumbnail((cell_w - 12, cell_h - 30))
        canvas.paste(thumb, (x + 6, y + 5))
        draw.text((x + 6, y + cell_h - 20), caption, fill="black")
    canvas.save(output)


def fit(args):
    validate_text_prior(args.charset, args.lengths)
    if args.trials < 5 or args.samples < 4 or not 1 <= args.clusters <= 8:
        raise ValueError("Need trials >= 5, samples >= 4, clusters 1..8")
    if args.vlm_images < args.clusters or args.vlm_images > 64:
        raise ValueError("vlm-images must be at least clusters and at most 64")
    split = prepare_split(args.real_dir, args.exclude_dir, args.seed, args.validation_fraction, args.max_images)
    fonts, rejected_fonts = discover_fonts(args.charset, args.font_dir)
    out = new_output(args.out, [args.real_dir] + args.exclude_dir)
    write_json(out / "split.json", split)
    write_json(out / "fonts.json", {"accepted": fonts, "rejected": rejected_fonts})
    status = {"status": "running", "version": __version__, "mode": args.mode, "started": time.time()}
    write_json(out / "status.json", status)
    if getattr(args, 'trace_search', False):
        (out / 'search_previews').mkdir()
    extractor = None
    try:
        images = [load_image(r["path"]) for r in split["fit"]]
        validation_images = [load_image(r["path"]) for r in split["validation"]]
        profiles = [palette_profile(im, r["pixel_sha256"]) for im, r in zip(images, split["fit"])]
        # Clustering and VLM representative selection use CPU descriptors from fitting images only.
        pixels = FeatureExtractor().batch(images)
        clusters = cluster_styles(pixels, args.clusters, args.seed)
        hints = {}
        if args.mode == "light-vlm":
            from .vlm import observe
            report = observe(images, split["fit"], pixels, clusters, args.vlm_images,
                             args.vlm_model, args.cache_dir, args.allow_download)
            write_json(out / "vlm_observations.json", report)
            if not report["valid_observations"]:
                raise RuntimeError("No valid VLM style observations. Inspect vlm_observations.json; no silent mode fallback.")
            hints = report["hints"]
        # The VLM is unloaded before an optional feature model is loaded.
        extractor = FeatureExtractor(args.features, args.cache_dir, args.allow_download)
        real_features = pixels if args.features == "pixels" else extractor.batch(images)
        metric = DistributionMetric(real_features, extractor.groups)
        config = {"version": __version__, "mode": args.mode, "seed": args.seed, "features": args.features,
                  "charset": args.charset, "lengths": args.lengths,
                  "text_prior": "explicit sampling prior, NOT recovered from real labels or connected components",
                  "fonts": fonts, "styles": [], "input_directories": [str(Path(args.real_dir).resolve())],
                  "excluded_directories": split["excluded_directories"],
                  "fit_image_ids": [r["pixel_sha256"] for r in split["fit"]],
                  "validation_image_ids": [r["pixel_sha256"] for r in split["validation"]],
                  "selection": "training-image descriptors only; validation report computed after freezing",
                  "renderer": "Pillow native-resolution procedural renderer, no real pixels reused"}
        all_trials = []
        for cluster in sorted(set(clusters.tolist())):
            indices = np.where(clusters == cluster)[0]
            local_profiles = [profiles[i] for i in indices]
            target = real_features[indices]
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, cluster]))
            labels = [sample_label(rng, args.charset, args.lengths) for _ in range(args.samples)]
            seeds = rng.integers(0, 2 ** 32 - 1, args.samples)
            choices = rng.integers(len(local_profiles), size=args.samples)
            rows = []
            best_params = default_parameters()
            best_score = float("inf")
            for trial, stage in enumerate(["baseline"] + make_trials(args.trials)):
                params = dict(best_params) if stage == "baseline" else mutate_parameters(
                    best_params, stage, rng, len(fonts), hints.get(str(cluster)))
                probe, errors = render_probe(params, local_profiles, fonts, labels, seeds, choices)
                acceptance = len(probe) / args.samples
                scores = metric.compare(target, extractor.batch(probe)) if probe else None
                objective = scores["total"] + 2 * (1 - acceptance) if scores else None
                eligible = acceptance >= 0.85 and scores is not None
                row = {"cluster": int(cluster), "trial": trial, "stage": stage, "parameters": params,
                       "acceptance": acceptance, "scores": scores, "objective": objective,
                       "eligible": eligible, "rejection_examples": errors[:3]}
                rows.append(row)
                all_trials.append(row)
                if getattr(args, 'trace_search', False) and probe:
                    _sheet(probe[:8], ['Accepted probe'] * min(8, len(probe)),
                           out / 'search_previews' / f'style_{cluster:02d}_trial_{trial:02d}_{stage}.png',
                           title=f'Style {cluster + 1}, trial {trial + 1}, {stage}: {len(probe)}/{args.samples} accepted')
                with (out / "trials.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                if eligible and objective < best_score:
                    best_score, best_params = objective, dict(params)
                print(f"Style {cluster + 1}/{len(set(clusters.tolist()))}: trial {trial + 1}/{args.trials}, "
                      f"{stage}, accepted {len(probe)}/{args.samples}, objective={objective}", flush=True)
            shortlist = pareto_shortlist(rows)
            if not shortlist:
                raise RuntimeError(f"Style {cluster}: no configuration met acceptance >= 85%. "
                                   "Inspect trials.jsonl; widen the search rather than emitting rejected images.")
            config["styles"].append({"id": int(cluster), "weight": len(indices) / len(images),
                                      "profiles": local_profiles, "hint": hints.get(str(cluster), {}),
                                      "candidates": [{"parameters": r["parameters"], "fit_scores": r["scores"],
                                                      "fit_acceptance": r["acceptance"], "trial": r["trial"]}
                                                     for r in shortlist]})
        # Freeze now. No validation statistic above has influenced configuration selection.
        write_json(out / "generator.json", config)
        validation_features = extractor.batch(validation_images)
        diagnostics, examples, example_labels = [], [], []
        rng = np.random.default_rng(args.seed + 177)
        for style in config["styles"]:
            for candidate in style["candidates"]:
                labels = [sample_label(rng, args.charset, args.lengths) for _ in range(args.samples)]
                probe, errors = render_probe(candidate["parameters"], style["profiles"], fonts, labels,
                                              rng.integers(0, 2 ** 32 - 1, args.samples),
                                              rng.integers(len(style["profiles"]), size=args.samples))
                diagnostics.append({"style": style["id"], "trial": candidate["trial"],
                                    "fresh_seed_acceptance": len(probe) / args.samples,
                                    "rejection_examples": errors[:3]})
        # Independent draws from the frozen mixture give a whole-distribution validation report.
        mixture, failed = [], 0
        for i in range(max(32, args.samples)):
            label, style, candidate, profile, seed = choose_sample(config, args.seed + 999, i)
            try:
                im, _, _, _ = render_checked(label, candidate["parameters"], profile, fonts, seed)
                mixture.append(im)
                if len(examples) < 32:
                    examples.append(im)
                    example_labels.append(label)
            except QualityError:
                failed += 1
        if not mixture:
            raise RuntimeError("Frozen mixture failed all independent probes")
        synthetic_features = extractor.batch(mixture)
        fit_gap = metric.compare(real_features, synthetic_features)
        validation_gap = metric.compare(validation_features, synthetic_features)
        # Reference floor from real-vs-real; diagnostic only, not a normalized accuracy estimate.
        reference = metric.compare(real_features, validation_features)
        report = {"version": __version__, "mode": args.mode, "features": args.features,
                  "fit_images": len(images), "validation_images": len(validation_images),
                  "text_prior": config["text_prior"], "charset": args.charset, "lengths": args.lengths,
                  "fit_gap": fit_gap, "validation_gap": validation_gap, "real_real_reference": reference,
                  "candidate_fresh_seed_checks": diagnostics, "mixture_probe_rejections": failed,
                  "mixture_probe_accepted": len(mixture), "model_selection_used_validation": False,
                  "downstream_recognition_tested": False, "python": platform.python_version(),
                  "limitations": ["Descriptor gaps are style proxies, not recognition accuracy.",
                                  "Generic fonts may fail to reproduce unusual glyph shapes.",
                                  "Charset and length prior remain uncertain unless independently justified.",
                                  "Quality gates detect degeneracy, not human readability.",
                                  "Near-duplicate leakage needs separate review."]}
        write_json(out / "report.json", report)
        _sheet(examples, example_labels, out / "fit_preview.png", title="Synthetic diagnostic samples (known labels)")
        _sheet(validation_images[:32], ["Unlabeled validation image"] * min(32, len(validation_images)),
               out / "real_validation.png", title="Real validation images (never used for fitting)")
        status.update(status="complete", elapsed_seconds=time.time() - status["started"])
        write_json(out / "status.json", status)
        print(f"Frozen generator: {out / 'generator.json'}", flush=True)
        return out
    except Exception as exc:
        status.update(status="failed", error=str(exc), elapsed_seconds=time.time() - status["started"])
        write_json(out / "status.json", status)
        raise
    finally:
        if extractor:
            extractor.close()


def choose_sample(config, seed, index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    label = sample_label(rng, config["charset"], config["lengths"])
    weights = np.array([s["weight"] for s in config["styles"]], dtype=float)
    style = config["styles"][int(rng.choice(len(weights), p=weights / weights.sum()))]
    candidate = style["candidates"][int(rng.integers(len(style["candidates"]))) ]
    profile = style["profiles"][int(rng.integers(len(style["profiles"]))) ]
    return label, style, candidate, profile, int(rng.integers(0, 2 ** 32 - 1))


def generate(args):
    if args.count < 1 or args.retries < 1:
        raise ValueError("count and retries must be positive")
    config = read_json(args.config)
    if config.get("version") != __version__:
        raise ValueError("Unsupported generator config version")
    validate_text_prior(config["charset"], config["lengths"])
    verify_fonts(config["fonts"])
    parent_status = Path(args.config).parent / "status.json"
    if parent_status.exists() and read_json(parent_status).get("status") != "complete":
        raise ValueError("The fitting run did not complete; refusing to generate from it")
    out = new_output(args.out, config["input_directories"] + config["excluded_directories"] + [Path(args.config).parent])
    (out / "images").mkdir()
    (out / "masks").mkdir()
    if args.char_masks:
        (out / "character_masks").mkdir()
    if getattr(args, 'trace_count', 0):
        (out / 'stages').mkdir()
    write_json(out / "generator.json", config)
    run = {"status": "running", "requested": args.count, "written": 0, "seed": args.seed,
           "generator_sha256": sha256_file(args.config), "mode_used_for_fitting": config["mode"],
           "generation_uses_vlm": False, "started": time.time(), "version": __version__}
    write_json(out / "status.json", run)
    examples, captions = [], []
    try:
        with (out / "labels.csv").open("w", encoding="utf-8", newline="") as csv_file, \
                (out / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as metadata:
            writer = csv.DictWriter(csv_file, fieldnames=["filename", "label"])
            writer.writeheader()
            for i in range(args.count):
                label, style, candidate, profile, seed = choose_sample(config, args.seed, i)
                trace = {} if i < getattr(args, 'trace_count', 0) else None
                image, mask, char_masks, info = render_checked(
                    label, candidate["parameters"], profile, config["fonts"], seed, args.retries, trace=trace)
                filename = f"images/{i:08d}.png"
                image.save(out / filename)
                mask.save(out / "masks" / f"{i:08d}.png")
                if trace:
                    for stage, frame in trace.items():
                        frame.save(out / 'stages' / f'{i:08d}_{stage}.png')
                    _sheet(list(trace.values()), list(trace), out / 'stages' / f'{i:08d}_process.png',
                           columns=3, title=f'Exact accepted render stages; chosen label: {label}')
                if args.char_masks:
                    for j, char_mask in enumerate(char_masks):
                        Image.fromarray(char_mask).save(out / "character_masks" / f"{i:08d}_{j:02d}.png")
                info.update(filename=filename, index=i, style=style["id"], candidate_trial=candidate["trial"],
                            image_sha256=sha256_file(out / filename))
                writer.writerow({"filename": filename, "label": label})
                metadata.write(json.dumps(info, allow_nan=False) + "\n")
                run["written"] += 1
                if len(examples) < 64:
                    examples.append(image)
                    captions.append(label)
                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{args.count}", flush=True)
                    write_json(out / "status.json", run)
        _sheet(examples, captions, out / "preview.png", title="Synthetic output: procedural pixels and chosen labels")
        run.update(status="complete", elapsed_seconds=time.time() - run["started"])
        write_json(out / "status.json", run)
        print(f"Generated {args.count} images: {out}", flush=True)
    except Exception as exc:
        run.update(status="failed", error=str(exc), elapsed_seconds=time.time() - run["started"])
        write_json(out / "status.json", run)
        raise
    return out
