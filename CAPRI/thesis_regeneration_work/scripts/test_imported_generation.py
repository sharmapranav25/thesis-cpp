"""Bounded, instrumented test of the original repo's C1/C2 generation paths.

No renderer source edits, family-rule invention, fitting, OCR, VLM, or training.
Unsupported families are explicitly tested with the repo's generic text defaults.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime
import hashlib
import io
import json
import platform
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(r"E:\pranav\Final sem\CaptchaThesis")
PUBLISHED = Path(r"E:\pranav\Final sem\sourcefree_synth")
FONT_DIR = Path(r"E:\pranav\model 1\.venv\Lib\site-packages\matplotlib\mpl-data\fonts\ttf")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))
from sourcefree_synth.pixel_statistics import measure, aggregate, intersection
from experiments.gen_ablation import generate_lmdb as imported
from experiments.gen_ablation.families import FAMILIES, FAMILY_LENGTH, family_charset
from gen_gateonly.text_generator import CHARSET_MAP
from gen_gateonly.config import CharSet
import gen_gateonly.generator as gm

DATASETS = ["cryptbb", "city", "courier", "darkbay", "outlined_text_captcha"]
NAMES = dict(cryptbb="CryptBB", city="City", courier="Courier", darkbay="DarkBay", outlined_text_captcha="Downloaded: outlined text")
STAGES = [("generate_background", "01 Background"), ("apply_background_noise", "02 Background noise"),
          ("render_text_on_image", "03 Text"), ("apply_foreground_noise", "04 Foreground noise"),
          ("apply_post_processing", "05 Post-processing")]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pixels(img):
    return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()

def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")

def schema_record(schema):
    return json.loads(json.dumps(dataclasses.asdict(schema), default=str))

def font(size=18, bold=False):
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / ("arialbd.ttf" if bold else "arial.ttf")), size)

def paste_sample(board, img, box):
    x, y, width, height = box
    sample = img.convert("RGB")
    scale = min(2, width / sample.width, height / sample.height)
    sample = sample.resize((round(sample.width * scale), round(sample.height * scale)), Image.Resampling.NEAREST)
    board.paste(sample, (x + (width - sample.width) // 2, y + (height - sample.height) // 2))

def contact_sheet(paths, captions, title, out, columns=4):
    cell_w, cell_h = 350, 138
    board = Image.new("RGB", (columns * cell_w, 58 + ((len(paths)+columns-1)//columns)*cell_h), "#f3f5f8")
    draw = ImageDraw.Draw(board)
    draw.text((15, 15), title, font=font(22, True), fill="#172536")
    for i, (path, caption) in enumerate(zip(paths, captions)):
        x, y = (i % columns)*cell_w+10, 58+(i//columns)*cell_h
        with Image.open(path) as sample:
            paste_sample(board, sample, (x, y, cell_w-20, 94))
        draw.text((x, y+103), caption, font=font(16), fill="#253448")
    board.save(out)

def reference_run(dataset):
    branch = "external_pixels_outlined_01" if dataset == "outlined_text_captcha" else "pixel_statistics_01"
    return PUBLISHED / "runs" / branch / dataset

def sources():
    paths = list((REPO / "system1_synthetic_generator/previously_produced_generators/gen_gateonly").glob("*.py"))
    paths += [REPO / "system1_synthetic_generator/previously_produced_generators/gen_gateonly/.font_whitelist.json"]
    paths += [REPO / "experiments/gen_ablation" / name for name in ("families.py", "generate_lmdb.py", "__init__.py")]
    paths += list((REPO / "system5_clean_baseline/charsets").glob("*.txt"))
    return {str(p): sha(p) for p in paths if p.is_file()}

def font_mapping():
    whitelist = REPO / "system1_synthetic_generator/previously_produced_generators/gen_gateonly/.font_whitelist.json"
    data = json.loads(whitelist.read_text())
    mapping = []
    alphabet = CHARSET_MAP[CharSet.ALPHANUMERIC_MIXED]
    for old in sorted(set(data["whitelist"])):
        local = FONT_DIR / Path(old).name
        if local.is_file():
            with TTFont(str(local), lazy=True) as face:
                cmap = face.getBestCmap() or {}
                if all(ord(c) in cmap for c in alphabet):
                    mapping.append({"original": old, "local": str(local), "sha256": sha(local)})
    if not mapping:
        raise RuntimeError("No generic fonts mapped with full alphanumeric glyph coverage")
    mapped = dict(data, whitelist=sorted({r["local"] for r in mapping}))
    original_open = open
    def redirected_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).resolve() == whitelist.resolve():
            return io.StringIO(json.dumps(mapped))
        return original_open(file, *args, **kwargs)
    return mapping, redirected_open

def run_cell(out, dataset, condition, seed, count, redirected_open):
    folder = out / dataset / condition
    folder.mkdir(parents=True)
    supported = dataset in FAMILIES
    charset = family_charset(dataset) if supported else CHARSET_MAP[CharSet.ALPHANUMERIC_MIXED]
    length = FAMILY_LENGTH[dataset] if supported else None
    random.seed(seed)
    np.random.seed(seed)
    with patch("builtins.open", redirected_open):
        gen = imported.UniversalCaptchaGenerator(seed=seed, font_dirs=[str(FONT_DIR)])
    if not gen.font_manager._font_catalog:
        raise RuntimeError("Mapped generator discovered no fonts")
    state = {"sample": -1, "attempts": []}
    capture, history = {}, {}
    originals = {}
    for name, stage in STAGES:
        original = getattr(gm, name)
        originals[name] = original
        def observer(*args, _fn=original, _stage=stage, **kwargs):
            result = _fn(*args, **kwargs)
            img = result[0] if isinstance(result, tuple) else result
            capture[_stage] = img.copy()
            return result
        setattr(gm, name, observer)
    original_pipeline, original_generate = gen._render_pipeline, gen.generate
    def observed_pipeline(schema, text):
        capture.clear()
        result = original_pipeline(schema, text)
        entry = {"sample": state["sample"], "label": text, "schema": schema_record(schema),
                 "degenerate": gm._is_degenerate(result[2]), "quality_meta": result[2], "pixel_sha256": pixels(result[0]),
                 "stages": dict(capture)}
        state["attempts"].append(entry)
        history[entry["pixel_sha256"]] = entry
        return result
    def observed_generate(*args, **kwargs):
        state["requested_schema"] = schema_record(kwargs["schema_obj"])
        result = original_generate(*args, **kwargs)
        state["returned_label"] = result[1]
        return result
    gen._render_pipeline = observed_pipeline
    gen.generate = observed_generate
    rows, errors, measurements, histograms = [], [], [], []
    started = time.perf_counter()
    try:
        with (folder / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            for i in range(count):
                state.update(sample=i, attempts=[])
                try:
                    img, exported_label = imported.generate_one(condition, charset, gen=gen, length=length)
                    filename = f"{i:03d}.png"
                    img.save(folder / filename)
                    with Image.open(folder / filename) as check:
                        check.load()
                        if check.size != img.size or pixels(check) != pixels(img):
                            raise RuntimeError("Saved PNG differs from returned image")
                    chosen = history.get(pixels(img))
                    row = {"sample_index": i, "filename": filename, "exported_label": exported_label, "internal_returned_label": state["returned_label"],
                           "label_alignment": exported_label == state["returned_label"],
                           "width": img.width, "height": img.height, "pixel_sha256": pixels(img),
                           "attempts": len(state["attempts"]), "returned_from_sample": chosen["sample"] if chosen else None,
                           "returned_degenerate": chosen["degenerate"] if chosen else None,
                           "retry_changed_schema": any(a["schema"] != state["requested_schema"] for a in state["attempts"][1:]),
                           "text_contract_valid": (len(exported_label) == length and all(c in charset for c in exported_label)) if supported else None}
                    rows.append(row)
                    json_write(folder / f"{i:03d}_trace.json", {
                        "requested_schema": state["requested_schema"], "result": row,
                        "attempts": [{k:v for k,v in a.items() if k != "stages"} for a in state["attempts"]]})
                    m, h = measure(img)
                    measurements.append(m)
                    histograms.append(h)
                    if i < 2 and chosen:
                        stages = folder / f"{i:03d}_stages"
                        stages.mkdir()
                        stage_paths = []
                        for j, (title, frame) in enumerate(chosen["stages"].items()):
                            path = stages / f"{j+1:02d}.png"
                            frame.save(path)
                            stage_paths.append(path)
                        final = stages / "06.png"
                        img.save(final)
                        stage_paths.append(final)
                        contact_sheet(stage_paths, list(chosen["stages"]) + ["06 Returned output"],
                                      f"{NAMES[dataset]} / {condition} / sample {i:03d} / {row['attempts']} attempts", folder / f"{i:03d}_process.png", 3)
                    print(json.dumps(row))
                except Exception as exc:
                    error = {"sample": i, "error": type(exc).__name__ + ": " + str(exc)}
                    errors.append(error)
                    print(json.dumps(error))
    finally:
        for name, original in originals.items():
            setattr(gm, name, original)
    profile = aggregate(measurements, histograms) if rows else None
    if profile:
        json_write(folder / "generated_profile.json", profile)
        real = json.loads((reference_run(dataset) / "validation_profile.json").read_text())
        observed = {(r["width"], r["height"]) for r in real["dimensions"]}
        comparison = {"reference": str(reference_run(dataset) / "validation_profile.json"),
                      "reference_count": real["image_count"], "generated_count": len(rows),
                      "rgb_histogram_intersection": intersection(np.array(profile["mean_rgb_histogram"]), np.array(real["mean_rgb_histogram"])),
                      "reference_brightness": real["mean_image_brightness"], "generated_brightness": profile["mean_image_brightness"],
                      "generated_dimensions_seen_in_validation": sum((r["width"], r["height"]) in observed for r in rows),
                      "meaning": "Descriptive only; color overlap is not recognition accuracy or regeneration fidelity"}
        json_write(folder / "comparison.json", comparison)
        contact_sheet([folder / r["filename"] for r in rows], [r["exported_label"] + (" [LABEL MISMATCH]" if not r["label_alignment"] else "") for r in rows],
                      f"{NAMES[dataset]} / {condition} / " + ("repo text contract" if supported else "GENERIC FALLBACK"), folder / "preview.png")
    else:
        comparison = None
    with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "exported_label", "internal_returned_label", "label_alignment"])
        writer.writerows((r["filename"], r["exported_label"], r["internal_returned_label"], r["label_alignment"]) for r in rows)
    result = {"dataset": dataset, "condition": condition, "seed": seed, "requested": count, "written": len(rows),
              "support": "REPO_TEXT_CONTRACT" if supported else "NO_FAMILY_CONFIG_GENERIC_FALLBACK", "charset": charset,
              "length": length, "length_policy": "Repo FAMILY_LENGTH" if supported else "Unmodified random_schema text.length_range",
              "family_rules_learned_from_images": False, "real_images_used_for_generation": False,
              "font_count": len(gen.font_manager._font_catalog), "label_mismatches": sum(not r["label_alignment"] for r in rows),
              "stale_returns": sum(r["returned_from_sample"] != r["sample_index"] for r in rows),
              "degenerate_returns": sum(r["returned_degenerate"] is True for r in rows),
              "schema_substitutions": sum(r["retry_changed_schema"] for r in rows),
              "duplicate_outputs": len(rows)-len({(r["width"],r["height"],r["pixel_sha256"]) for r in rows}),
              "errors": errors, "samples": rows, "comparison": comparison, "elapsed_seconds": time.perf_counter()-started}
    result["execution_status"] = "PASS" if len(rows) == count and not errors else "FAIL"
    result["internal_integrity_status"] = "PASS" if result["execution_status"] == "PASS" and not any(result[k] for k in ("label_mismatches","stale_returns","degenerate_returns")) else "FAIL"
    json_write(folder / "result.json", result)
    print(json.dumps({k:result[k] for k in ("dataset","condition","written","label_mismatches","stale_returns","degenerate_returns","schema_substitutions","elapsed_seconds")}), flush=True)
    return result

def ab_preview(out):
    board = Image.new("RGB", (1440, 1270), "#f3f5f8")
    draw = ImageDraw.Draw(board)
    draw.text((34, 22), "Real and generated CAPTCHA comparison", font=font(30, True), fill="#172536")
    draw.text((34, 71), "A  REAL / REFERENCE", font=font(25, True), fill="#126168")
    draw.text((744, 62), "B  GENERATED BY THE GENERATOR", font=font(23, True), fill="#68449d")
    draw.text((744, 90), "CURRENTLY ON GITLAB", font=font(23, True), fill="#68449d")
    draw.text((34, 110), "Two predetermined validation samples per dataset", font=font(18), fill="#536272")
    draw.text((744, 124), "Left: C1 random     |     Right: C2 curriculum", font=font(18), fill="#536272")
    selected = []
    for index, dataset in enumerate(DATASETS):
        top = 158+index*202
        draw.rounded_rectangle((24, top, 1416, top+189), radius=12, fill="white")
        draw.text((42, top+12), NAMES[dataset], font=font(23, True), fill="#172536")
        scope = "Repo character set + length; generic visual styles" if dataset in FAMILIES else "NO FAMILY CONFIG: generic fallback, no target rules inferred"
        draw.text((744, top+15), scope, font=font(16), fill="#68449d")
        draw.line((720, top+12, 720, top+174), fill="#dee4eb", width=2)
        run = reference_run(dataset)
        hashes = json.loads((run / "visual_selection.json").read_text())["validation"][:2]
        records = json.loads((run / "split.json").read_text())["validation"]
        lookup = {r["pixel_sha256"]:r for r in records}
        refs = [Path(lookup[h]["path"]) for h in hashes]
        if dataset == "outlined_text_captcha":
            refs = [PUBLISHED / "data/external/outlined_text_captcha/images" / p.name for p in refs]
        generated = [out / dataset / condition / "000.png" for condition in ("c1_random", "c2_curriculum")]
        for column, paths in enumerate((refs, generated)):
            for j, path in enumerate(paths):
                x, y = 42+column*708+j*328, top+56
                draw.rounded_rectangle((x,y,x+312,y+107), radius=7, fill="#edf1f5")
                if path.exists():
                    with Image.open(path) as img:
                        paste_sample(board, img, (x+6,y+6,300,95))
                else:
                    draw.text((x+20,y+40), "Generation failed: see log", font=font(17), fill="#ad243b")
        selected.append({"dataset":dataset,"reference_paths":[str(p) for p in refs],"generated_paths":[str(p) for p in generated]})
    draw.text((34, 1182), "Saved outputs from the imported GitLab generator. Styles were not fitted to A; no VLM or training used.", font=font(20), fill="#536272")
    draw.text((34, 1214), "Preview uses the first output of each mode, without selecting for visual similarity. Images scaled proportionally.", font=font(18), fill="#536272")
    board.save(out / "all_datasets_ab.png")
    json_write(out / "preview_selection.json", selected)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.count <= 64:
        parser.error("This smoke test requires 1..64 images per cell")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = sources()
    mapping, redirected = font_mapping()
    json_write(out / "font_path_mapping.json", mapping)
    # Read-only readiness check. Do not silently render through the default tiny fallback font.
    with contextlib.redirect_stdout(io.StringIO()):
        unadapted = imported.UniversalCaptchaGenerator(seed=42, font_dirs=[str(FONT_DIR)])
    json_write(out / "as_is_preflight.json", {"font_count":len(unadapted.font_manager._font_catalog),
               "generation_skipped":not bool(unadapted.font_manager._font_catalog), "reason":"Original Linux whitelist paths versus installed Windows generic fonts"})
    started = time.perf_counter()
    results = []
    for d, dataset in enumerate(DATASETS):
        for c, condition in enumerate(("c1_random", "c2_curriculum")):
            results.append(run_cell(out, dataset, condition, 20260928 + d*100+c, args.count, redirected))
    after = sources()
    module_paths = {n:str(m.__file__) for n,m in sys.modules.items()
                    if (n.startswith("gen_gateonly") or n.startswith("experiments.gen_ablation")) and getattr(m,"__file__",None)}
    direct = bool(module_paths) and all(REPO.resolve() in Path(p).resolve().parents for p in module_paths.values())
    json_write(out / "source_receipt.json", {"before":before,"after":after,"changed":before != after,"module_paths":module_paths,"imported_directly_from_original_repo":direct,"harness_sha256":sha(__file__)})
    summary = {"date":datetime.datetime.now().astimezone().isoformat(), "scope":"C1/C2 execution and internal-label smoke test; not learned regeneration", "python":platform.python_version(),
               "pillow":PIL.__version__,"numpy":np.__version__,"vlm_calls":0,"training":False,"source_unchanged":before == after,"original_repo_imports_verified":direct,
               "font_change":"In-memory mapping of whitelisted generic font names to installed local paths; no source edits",
               "elapsed_seconds":time.perf_counter()-started,"written":sum(r["written"] for r in results),
               "cases":[{k:v for k,v in r.items() if k != "samples"} for r in results],"full_regeneration_validated":False}
    json_write(out / "summary.json", summary)
    ab_preview(out)
    lines = ["# Imported generation on five dataset contexts", "", "A fresh C1/C2 test importing the original repository directly. This tests rendering, not image-based adaptation.", "",
             "| Dataset | Mode | Family support | Written | Label mismatch | Stale | Degenerate | Changed recipe on retry |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['dataset']} | {r['condition']} | {r['support']} | {r['written']} | {r['label_mismatches']} | {r['stale_returns']} | {r['degenerate_returns']} | {r['schema_substitutions']} |")
    lines += ["", f"Total: {summary['written']} fresh images. Source unchanged: {before == after}. Runtime: {summary['elapsed_seconds']:.2f}s.", "",
              "The imported repo constrains character set and length for CryptBB, City and DarkBay. Courier and outlined-text have no family entry: their runs use the repo's generic alphanumeric text defaults and schema-selected length. No family style is inferred, and no reference image enters generation.", "",
              "The original Linux font whitelist accepts no local fonts in the preflight (see receipt). Runs map matching generic font names to installed Windows paths in memory, verifying character coverage. This is an environment adaptation, not an unmodified-environment success.", "",
              "Internal label checks trace the renderer's returned image and text; they do not prove OCR readability. Every PNG was reopened and checked. Histogram overlap and dimensions are descriptive, not a fidelity score. No training/VLM run occurred; C3 learned weights and the separate VLM code-generation workflow were not tested.", "",
              "## Evidence", "", "- [Fresh A/B preview](all_datasets_ab.png)", "- [Summary](summary.json)", "- [Source hashes and module paths](source_receipt.json)",
              "- Each dataset/mode directory contains all images, labels.csv, result.json, console.log, per-output retry traces, a batch preview, and stage panels for the first two outputs.",
              "- Real-reference manifests come from the earlier validation splits. Their labels were not used.", "", "Qualitative visual review is recorded separately after opening the new images."]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"output":str(out),"written":summary["written"],"source_unchanged":before==after,"direct_imports":direct}), flush=True)
    if before != after or not direct:
        raise SystemExit("Source or module provenance verification failed")

if __name__ == "__main__":
    main()
