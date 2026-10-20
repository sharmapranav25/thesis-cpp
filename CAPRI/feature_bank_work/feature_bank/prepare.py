"""Prediction-independent validation selection; retains historical partitions."""
import hashlib
import platform
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import PIL
import numpy as np
import cv2
import sklearn
import skimage
from .core import FEATURES, digest, read, write, load, pixel_hash, new_run

SEED = "feature-bank-annotations-v1-20260927"
DATASETS = ("cryptbb", "city", "courier", "darkbay", "outlined_text_captcha")

def resolve_image(root, dataset, record):
    # The external manifest retains its former staging location. Relocation is
    # explicit and hash-checked, never a filename-derived semantic feature.
    if dataset == "outlined_text_captcha":
        return root / "data/external/outlined_text_captcha/images" / Path(record["path"]).name
    return Path(record["path"])

def panel(rows, folder, filename, title):
    canvas = Image.new("RGB", (1000, 40 + 125 * ((len(rows) + 3)//4)), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="black")
    for i, row in enumerate(rows):
        x, y = 10 + (i % 4)*250, 40 + (i//4)*125
        im = load(folder / row["image"])
        im.thumbnail((235, 95))
        canvas.paste(im, (x, y+22))
        draw.text((x, y), row["id"], fill="black")
    canvas.save(folder / filename)

def prepare(root, out):
    root, out = Path(root).resolve(), new_run(out)
    write(out / "status.json", {"stage": 1, "state": "RUNNING"})
    sources, images, checks = [], [], []
    all_selected = set()
    for name in DATASETS:
        base = "external_pixels_outlined_01" if name == "outlined_text_captcha" else "pixel_statistics_01"
        path = root / "runs" / base / name / "split.json"
        split = read(path)
        fit = {r["pixel_sha256"] for r in split["fit"]}
        val = {r["pixel_sha256"] for r in split["validation"]}
        if len(val) != len(split["validation"]) or fit & val or len(val) < 40:
            raise ValueError("Invalid historical partition: " + name)
        excluded = set()
        for directory in split["excluded_directories"]:
            for p in Path(directory).rglob("*"):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
                    excluded.add(pixel_hash(load(p)))
        # Validate every historical member, not just the selected sample.
        for row in split["fit"] + split["validation"]:
            p = resolve_image(root, name, row)
            image = load(p)
            if digest(p) != row["file_sha256"] or pixel_hash(image) != row["pixel_sha256"]:
                raise ValueError("Historical input changed: " + str(p))
            if image.size != (row["width"], row["height"]) or row["pixel_sha256"] in excluded:
                raise ValueError("Dimension or external-overlap failure")
        ordered = sorted(split["validation"], key=lambda r: hashlib.sha256(
            (SEED + ":" + name + ":" + r["pixel_sha256"]).encode()).hexdigest())[:40]
        dataset_rows = []
        (out / "images" / name).mkdir(parents=True)
        for i, row in enumerate(ordered):
            if row["pixel_sha256"] in all_selected:
                raise ValueError("Duplicate across evaluation datasets")
            all_selected.add(row["pixel_sha256"])
            sid = f"{name}-{i+1:03d}"
            rel = f"images/{name}/{sid}.png"
            p = resolve_image(root, name, row)
            load(p).save(out / rel)
            record = dict(row, id=sid, dataset=name, image=rel, resolved_path=str(p),
                          selected_partition="validation", export_file_sha256=digest(out/rel))
            images.append(record)
            dataset_rows.append(record)
        for page in range(2):
            panel(dataset_rows[page*20:(page+1)*20], out, f"{name}_{page+1}.png",
                  f"{name} | prediction-blind annotation review | page {page+1}/2")
        sources.append(dict(dataset=name, path=str(path), sha256=digest(path),
                            fit=len(fit), validation=len(val), unique=split["unique_count"]))
        checks.append(dict(dataset=name, verified_images=len(fit)+len(val), selected=40,
                           excluded_unique=len(excluded), split_disjoint=True, changed_inputs=0))
        print(name, checks[-1], flush=True)
    manifest = dict(version="feature-bank-v1", seed=SEED, sources=sources, images=images,
                    selection="lowest SHA256(seed:dataset:pixel_hash), validation only", checks=checks)
    write(out/"manifest.json", manifest)
    write(out/"feature_definitions.json", FEATURES)
    write(out/"environment.json", dict(python=sys.version, platform=platform.platform(),
          packages={m.__name__: m.__version__ for m in (PIL, np, cv2, sklearn, skimage)},
          code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}))
    write(out/"status.json", dict(stage=1, state="SELECTION_VERIFIED_ANNOTATIONS_PENDING",
          images=200, labels_used_for_selection=False, checks=checks))
    return out
