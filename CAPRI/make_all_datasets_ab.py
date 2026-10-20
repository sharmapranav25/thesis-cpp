"""Create one comparison sheet from existing images; no regeneration or fitting."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"E:\pranav\Final sem\sourcefree_synth")
OUT = Path(__file__).parent / "all_datasets_ab.png"
BASELINE = ROOT / "runs/baseline_replay_01/universal/c1_random_whitelist_paths_mapped"
DATASETS = [
    ("CryptBB", ROOT / "runs/pixel_statistics_01/cryptbb"),
    ("City", ROOT / "runs/pixel_statistics_01/city"),
    ("Courier", ROOT / "runs/pixel_statistics_01/courier"),
    ("DarkBay", ROOT / "runs/pixel_statistics_01/darkbay"),
    ("Downloaded: outlined-text CAPTCHA", ROOT / "runs/external_pixels_outlined_01/outlined_text_captcha"),
]

def font(size, bold=False):
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / ("arialbd.ttf" if bold else "arial.ttf")), size)

def read_references(run):
    selection = json.loads((run / "visual_selection.json").read_text())["fit"]
    records = json.loads((run / "split.json").read_text())["fit"]
    lookup = {r["pixel_sha256"]: r for r in records}
    paths = []
    for key in selection[:2]:
        path = Path(lookup[key]["path"])
        if run.name == "outlined_text_captcha":
            path = ROOT / "data/external/outlined_text_captcha/images" / path.name
        paths.append(path)
    return paths

board = Image.new("RGB", (1440, 1160), "#f3f5f8")
draw = ImageDraw.Draw(board)
draw.text((36, 25), "All five datasets | A / B comparison", font=font(32, True), fill="#152334")
draw.text((36, 76), "A  REAL / REFERENCE", font=font(26, True), fill="#126168")
draw.text((744, 76), "B  GENERATED BASELINE", font=font(26, True), fill="#68449d")
draw.text((36, 113), "Two existing samples from each dataset", font=font(19), fill="#536272")
draw.text((744, 113), "Same two imported-generator samples repeated in every row", font=font(19), fill="#536272")
generated = [BASELINE / "000.png", BASELINE / "001.png"]
manifest = {"purpose": "Display existing reference images for all five datasets against a shared generic baseline", "generation_performed": False, "family_fitted": False, "rows": []}

for index, (name, run) in enumerate(DATASETS):
    top = 162 + index * 183
    draw.rounded_rectangle((24, top, 1416, top + 169), radius=12, fill="white")
    draw.text((42, top + 12), name, font=font(23, True), fill="#1e2b3b")
    draw.line((720, top + 12, 720, top + 154), fill="#dee4eb", width=2)
    refs = read_references(run)
    for column, paths in enumerate((refs, generated)):
        for sample_index, path in enumerate(paths):
            x = 42 + column * 708 + sample_index * 328
            y = top + 51
            draw.rounded_rectangle((x, y, x + 312, y + 101), radius=7, fill="#edf1f5")
            with Image.open(path) as source:
                sample = source.convert("RGB")
            scale = min(1.8, 296 / sample.width, 87 / sample.height)
            sample = sample.resize((round(sample.width * scale), round(sample.height * scale)), Image.Resampling.NEAREST)
            board.paste(sample, (x + (312 - sample.width) // 2, y + (101 - sample.height) // 2))
    manifest["rows"].append({"dataset": name, "A": [str(p) for p in refs], "B": [str(p) for p in generated], "reference_selection": str(run / "visual_selection.json")})

draw.text((36, 1093), "B is a shared baseline, not an output fitted separately to each dataset. No new images were generated for this preview.", font=font(20), fill="#536272")
draw.text((36, 1125), "Images preserve their aspect ratios; examples are independent and do not share text labels.", font=font(18), fill="#536272")
board.save(OUT)
OUT.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
OUT.with_suffix(".md").write_text("# All-dataset A/B preview\n\nA shows two preselected fitting examples for each of CryptBB, City, Courier, DarkBay, and the downloaded outlined-text dataset. B repeats the same two existing imported C1 baseline outputs in every row, so it does not imply five separately fitted generators. No training, fitting, or regeneration was performed.\n\nImages are resized proportionally for display. Selection and original file paths are recorded in `all_datasets_ab.json`. This is a qualitative comparison, not a regeneration validation result.\n", encoding="utf-8")
print(OUT)
print("Verified all five dataset rows and loaded 10 reference images; reused the same two generated baseline images.")
