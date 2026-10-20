"""Small frozen-hybrid -> learned block-controller -> renderer experiment."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

HERE = Path(__file__).resolve().parent
SYNTH = Path(r"E:\pranav\Final sem\sourcefree_synth")
JOINT = SYNTH / "feature_joint_mapping_01"
RUN = JOINT / "run_conformer"
PAIRS = RUN / "real_generation" / "cryptbb"
PHASE1 = SYNTH / "releases" / "CAPRI_phase123_v1" / "phase1"
sys.path.insert(0, str(JOINT))
from corpus import FEATURES  # noqa: E402
from models import Joint  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_array(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGB")
    return np.asarray(image.convert("RGB").resize((192, 96), Image.Resampling.BILINEAR)).copy()


def block_style(base, count, width):
    style = copy.deepcopy(base)
    if count <= 0:
        style["background_block_evidence"] = {"state": "unsupported"}
    else:
        colors = [[191, 155, 223], [183, 218, 180], [229, 183, 200]]
        style["background_block_evidence"] = {
            "state": "supported",
            "count": int(count),
            "candidates": [
                {"width_fraction": float(width), "height_fraction": float(width * 1.8), "color": color}
                for color in colors
            ],
        }
    return style


def features(model, images):
    arrays = np.stack([load_array(image) for image in images])
    model_values = []
    with torch.inference_mode():
        for start in range(0, len(arrays), 24):
            tensor = torch.from_numpy(arrays[start:start + 24].transpose(0, 3, 1, 2)).float() / 255
            logits, _ = model(tensor)
            model_values.append(logits.sigmoid().numpy())
    model_values = np.concatenate(model_values)
    # Cheap independent whole-image statistics supplement the frozen hybrid.
    rgb = arrays.astype(np.float32) / 255
    means = rgb.mean(axis=(1, 2))
    stds = rgb.std(axis=(1, 2))
    chroma = (rgb.max(axis=3) - rgb.min(axis=3)).mean(axis=(1, 2))[:, None]
    high_chroma = ((rgb.max(axis=3) - rgb.min(axis=3)) > .2).mean(axis=(1, 2))[:, None]
    grey = rgb.mean(axis=3)
    edges = (np.abs(np.diff(grey, axis=1)).mean(axis=(1, 2)) + np.abs(np.diff(grey, axis=2)).mean(axis=(1, 2)))[:, None]
    x = np.concatenate([model_values, means, stds, chroma, high_chroma, edges], axis=1)
    return x.astype(np.float32), model_values


class Controller(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 2), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(731)
    rng = np.random.default_rng(731)
    checkpoint = RUN / "model_731.pt"
    assert sha(checkpoint) == read(RUN / "real_generation" / "protocol.json")["model_sha256"]
    frozen = Joint("conformer").eval()
    frozen.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    render = module(RUN / "real_generation" / "renderer_snapshot.py", "saved_renderer").render
    recipes = read(PAIRS / "recipes.json")
    original = read(SYNTH / "runs" / "style_multilabel_phase12_01" / "cryptbb" / "recipes.json")
    fonts = read(PHASE1 / "base" / "assets" / "fonts.json")
    assert len(recipes) == 42 and len(original) == 21
    assert all(recipes[2*i]["reference"] == original[i]["reference"] for i in range(21))

    images, y, groups = [], [], []
    counts = [0, 4, 8, 12]
    widths = [.06, .12]
    for i in range(21):
        base = recipes[2*i]["style"]
        case = original[i]
        font = PHASE1 / "base" / fonts[case["font_index"]]["path"]
        for count in counts:
            for width in widths:
                style = block_style(base, count, width)
                seed = 100000 + i * 100 + count * 2 + (width > .06)
                image, _, _ = render(style, case["text"], str(font), seed, 3)
                images.append(image)
                y.append([count / 12, width / .12])
                groups.append(i)
    x, _ = features(frozen, images)
    y = np.asarray(y, np.float32)
    groups = np.asarray(groups)
    # Entire source-style groups are held out; no variant crosses partitions.
    train = groups < 15
    val = (groups >= 15) & (groups < 18)
    test = groups >= 18
    mean, std = x[train].mean(0), x[train].std(0).clip(.01)
    scaled = (x - mean) / std
    model = Controller(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    xt = torch.from_numpy(scaled)
    yt = torch.from_numpy(y)
    best, best_state, best_epoch = math.inf, None, 0
    for epoch in range(160):
        model.train()
        for batch in rng.permutation(np.where(train)[0]).reshape(-1, 24):
            optimizer.zero_grad()
            prediction = model(xt[batch])
            loss = ((prediction - yt[batch]) ** 2).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(((model(xt[val]) - yt[val]) ** 2).mean())
        if val_loss < best:
            best, best_epoch = val_loss, epoch + 1
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        test_pred = model(xt[test]).numpy()
    test_truth = y[test]
    count_mae = float(np.abs(test_pred[:, 0] - test_truth[:, 0]).mean() * 12)
    width_mae = float(np.abs(test_pred[:, 1] - test_truth[:, 1]).mean() * .12)
    baseline_count_mae = float(np.abs(y[train, 0].mean() - test_truth[:, 0]).mean() * 12)
    baseline_width_mae = float(np.abs(y[train, 1].mean() - test_truth[:, 1]).mean() * .12)

    real_paths = [Path(recipes[2*i]["reference"]) for i in range(21)]
    real_x, real_scores = features(frozen, real_paths)
    with torch.inference_mode():
        real_params = model(torch.from_numpy((real_x - mean) / std)).numpy()
    generated, fixed, baseline, visible, chosen = [], [], [], [], []
    for i in range(21):
        case = original[i]
        base = recipes[2*i]["style"]
        count = int(np.clip(np.rint(real_params[i, 0] * 12), 0, 12))
        width = float(np.clip(real_params[i, 1] * .12, .04, .14))
        style = block_style(base, count, width)
        font = PHASE1 / "base" / fonts[case["font_index"]]["path"]
        image, _, trace = render(style, case["text"], str(font), case["recipe"]["seed"], 3)
        fixed_image, _, _ = render(block_style(base, 8, .09), case["text"], str(font), case["recipe"]["seed"], 3)
        baseline_path = PAIRS / f"{i:03d}_median.png"
        generated.append(image)
        fixed.append(fixed_image)
        baseline.append(baseline_path)
        visible.append(min(g["visible_fraction"] for g in trace["glyphs"]))
        chosen.append({"reference": str(real_paths[i]), "count": count, "width_fraction": width, "actual_block_operations": sum(op["kind"] == "block" for op in trace["operations"])})
        image.save(HERE / f"controller_{i:03d}.png")
        fixed_image.save(HERE / f"fixed_{i:03d}.png")
    _, base_scores = features(frozen, baseline)
    _, controlled_scores = features(frozen, generated)
    _, fixed_scores = features(frozen, fixed)
    results = read(RUN / "results.json")
    passing = [name for name, passed in results["per_feature_gate"].items() if passed]
    thresholds = results["results"]["731"]["thresholds"]
    j = FEATURES.index("background_blocks")
    block_prevalence = {name: float((scores[:, j] >= thresholds["background_blocks"]).mean()) for name, scores in (("real", real_scores), ("baseline", base_scores), ("fixed_eight_blocks", fixed_scores), ("controller", controlled_scores))}
    score_gap = {name: float(np.mean([abs(real_scores[:, FEATURES.index(feature)].mean() - scores[:, FEATURES.index(feature)].mean()) for feature in passing])) for name, scores in (("baseline", base_scores), ("fixed_eight_blocks", fixed_scores), ("controller", controlled_scores))}
    report = {
        "setup": "Frozen seed-731 hybrid; small MLP controller trained on 168 synthetic renders; 15 style groups train, 3 validation, 3 synthetic test; 21 exposed real CryptBB references for diagnostic inference",
        "synthetic_test": {"samples": int(test.sum()), "count_mae": count_mae, "width_fraction_mae": width_mae, "constant_count_mae": baseline_count_mae, "constant_width_fraction_mae": baseline_width_mae, "best_epoch": best_epoch},
        "real_diagnostic": {"block_detection_fraction": block_prevalence, "seven_feature_mean_score_gap": score_gap, "minimum_glyph_visibility": float(min(visible)), "controller_block_count_mean": float(np.mean([r["count"] for r in chosen]))},
        "choices": chosen,
        "limits": ["Synthetic control recovery does not establish real parameter accuracy.", "The frozen hybrid both guides and scores the controller; real feature truth is unavailable.", "All real images have historical development exposure; no downstream OCR or GAN test was run.", "Only block count and shared size were controlled; colours and placement policy remained fixed."],
    }
    (HERE / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "checkpoint_sha256": sha(checkpoint)}, HERE / "controller.pt")
    sheet = Image.new("RGB", (1200, 110 * 6), "white")
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(("Real reference", "Existing median render", "Fixed eight blocks", "Trained block controller")):
        draw.text((column * 300 + 4, 3), label, fill="black")
    for row, i in enumerate((2, 4, 9, 11, 16, 18)):
        for col, source in enumerate((real_paths[i], baseline[i], fixed[i], generated[i])):
            image = Image.open(source).convert("RGB") if isinstance(source, Path) else source.copy()
            image.thumbnail((285, 85))
            sheet.paste(image, (col * 300 + 4, row * 110 + 20))
    sheet.save(HERE / "comparison.png")
    print(json.dumps({"synthetic_test": report["synthetic_test"], "real_diagnostic": report["real_diagnostic"]}, indent=2))


if __name__ == "__main__":
    main()
