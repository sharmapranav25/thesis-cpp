"""Compose existing reference and generated samples; do not regenerate images."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\pranav\Final sem\sourcefree_synth")
REFERENCE_RUN = ROOT / "runs/external_pixels_outlined_01/outlined_text_captcha"
GENERATED_RUN = ROOT / "runs/baseline_replay_01/universal/c1_random_whitelist_paths_mapped"
OUTPUT = Path(__file__).parent / "real_vs_generated_ab.png"

selection = json.loads((REFERENCE_RUN / "visual_selection.json").read_text())
split = json.loads((REFERENCE_RUN / "split.json").read_text())
by_hash = {item["pixel_sha256"]: item for item in split["fit"]}
references = [
    ROOT / "data/external/outlined_text_captcha/images" / Path(by_hash[key]["path"]).name
    for key in selection["fit"][:6]
]
generated = [GENERATED_RUN / f"{index:03d}.png" for index in range(6)]

def font(size, bold=False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / filename), size)

board = Image.new("RGB", (1120, 940), "#f4f6f8")
draw = ImageDraw.Draw(board)
draw.text((36, 23), "CAPTCHA comparison", font=font(28, True), fill="#17202e")
draw.rounded_rectangle((28, 76, 545, 872), radius=14, fill="white")
draw.rounded_rectangle((575, 76, 1092, 872), radius=14, fill="white")
draw.text((48, 92), "A  REAL / REFERENCE", font=font(25, True), fill="#156167")
draw.text((595, 92), "B  GENERATED", font=font(25, True), fill="#63419b")
draw.text((48, 129), "Downloaded outlined-text dataset", font=font(18), fill="#526170")
draw.text((595, 129), "Existing imported-renderer baseline", font=font(18), fill="#526170")

for row, pair in enumerate(zip(references, generated)):
    y = 170 + row * 114
    for column, path in enumerate(pair):
        x = 48 + column * 547
        draw.rounded_rectangle((x, y, x + 477, y + 102), radius=8, fill="#f4f6f8")
        with Image.open(path) as source:
            sample = source.convert("RGB")
        scale = min(2.0, 450 / sample.width, 84 / sample.height)
        sample = sample.resize((round(sample.width * scale), round(sample.height * scale)), Image.Resampling.NEAREST)
        board.paste(sample, (x + (477 - sample.width) // 2, y + (102 - sample.height) // 2))

draw.text((36, 891), "Independent examples; B has not been adapted to A. Images scaled proportionally for display.", font=font(18), fill="#526170")
board.save(OUTPUT)
OUTPUT.with_suffix(".json").write_text(json.dumps({
    "purpose": "A/B display of existing images, not a new generation run",
    "A": [str(p) for p in references],
    "B": [str(p) for p in generated],
    "selection": "First six already selected fit-panel examples and baseline samples 000-005; no similarity selection",
    "display": "Aspect-preserving resize, nearest-neighbor, maximum 2x",
}, indent=2), encoding="utf-8")
print(OUTPUT)
