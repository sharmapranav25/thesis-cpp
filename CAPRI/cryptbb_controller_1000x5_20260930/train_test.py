"""Five-epoch, 1,000-image CryptBB block-controller proof of concept."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import string
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "cryptbb_controller_poc_20260930"
sys.path.insert(0, str(PRIOR))
from train_test import block_style, load_array, read, sha  # noqa: E402

SYNTH = Path(r"E:\pranav\Final sem\sourcefree_synth")
JOINT = SYNTH / "feature_joint_mapping_01"
RUN = JOINT / "run_conformer"
PAIRS = RUN / "real_generation" / "cryptbb"
PHASE1 = SYNTH / "releases" / "CAPRI_phase123_v1" / "phase1"
sys.path.insert(0, str(JOINT))
from corpus import FEATURES  # noqa: E402
from models import Joint  # noqa: E402


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def encode(frozen, images):
    arrays = np.stack([load_array(image) for image in images])
    features, scores = [], []
    captured = []
    hook = frozen.stem.register_forward_hook(lambda _module, _input, output: captured.append(output))
    with torch.inference_mode():
        for start in range(0, len(arrays), 32):
            batch = torch.from_numpy(arrays[start:start + 32].transpose(0, 3, 1, 2)).float() / 255
            logits, _ = frozen(batch)
            local = captured.pop()
            pooled = torch.cat([local.mean(dim=(2, 3)), local.std(dim=(2, 3))], dim=1)
            scores.append(logits.sigmoid().numpy())
            features.append(pooled.numpy())
    hook.remove()
    scores = np.concatenate(scores)
    features = np.concatenate(features)
    rgb = arrays.astype(np.float32) / 255
    means = rgb.mean(axis=(1, 2))
    stds = rgb.std(axis=(1, 2))
    chroma = (rgb.max(axis=3) - rgb.min(axis=3)).mean(axis=(1, 2))[:, None]
    x = np.concatenate([scores, features, means, stds, chroma], axis=1)
    return x.astype(np.float32), scores


class Controller(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(size, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def loss_fn(pred, truth):
    count_loss = ((pred[:, 0] - truth[:, 0]) ** 2).mean()
    positive = (truth[:, 0] > 0).float()
    width_loss = (((pred[:, 1] - truth[:, 1]) ** 2) * positive).sum() / positive.sum().clamp(min=1)
    return count_loss + width_loss


def main():
    torch.set_num_threads(4)
    torch.manual_seed(731)
    rng = np.random.default_rng(731)
    checkpoint = RUN / "model_731.pt"
    assert sha(checkpoint) == read(RUN / "real_generation" / "protocol.json")["model_sha256"]
    frozen = Joint("conformer").eval()
    frozen.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    render = module(RUN / "real_generation" / "renderer_snapshot.py", "saved_renderer_1000").render
    profile = read(PHASE1 / "profiles" / "conformer" / "cryptbb.json")
    fonts = read(PHASE1 / "base" / "assets" / "fonts.json")
    styles = profile["entries"]
    assert len(styles) == 96
    order = rng.permutation(96)
    groups = {"train": order[:70].tolist(), "validation": order[70:83].tolist(), "test": order[83:].tolist()}
    targets = {"train": 1000, "validation": 200, "test": 200}
    images, labels, roles, source_ids = [], [], [], []
    rejected_renders = 0
    alphabet = string.ascii_letters + string.digits
    for role in ("train", "validation", "test"):
        for number in range(targets[role]):
            count = int(rng.choice([0, 4, 8, 12]))
            width = float(rng.choice([.05, .08, .11, .14]))
            for attempt in range(40):
                idx = int(rng.choice(groups[role]))
                style = copy.deepcopy(styles[idx]["style"])
                text = "".join(rng.choice(list(alphabet), 5))
                font_id = int(rng.integers(len(fonts)))
                font = PHASE1 / "base" / fonts[font_id]["path"]
                seed = 200000 + int(rng.integers(2**28))
                try:
                    image, _, _ = render(block_style(style, count, width), text, str(font), seed, 3)
                    break
                except ValueError:
                    rejected_renders += 1
            else:
                raise RuntimeError(f"No valid render after 40 attempts for {role}/{number}")
            images.append(image)
            labels.append([count / 12, width / .14])
            roles.append(role)
            source_ids.append(idx)
    assert len(images) == 1400
    x, _ = encode(frozen, images)
    y = np.asarray(labels, np.float32)
    roles = np.asarray(roles)
    train = roles == "train"
    val = roles == "validation"
    test = roles == "test"
    mean, std = x[train].mean(0), x[train].std(0).clip(.01)
    xt = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(y)
    model = Controller(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    history = []
    best_val, best_epoch, best_state = math.inf, None, None
    for epoch in range(1, 6):
        model.train()
        # User-requested full batch: exactly 1,000 training examples per update.
        batches = [rng.permutation(np.where(train)[0])]
        losses = []
        for batch in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            value = float(loss_fn(model(xt[val]), yt[val]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": value})
        if value < best_val:
            best_val, best_epoch = value, epoch
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predicted = model(xt[test]).numpy()
    truth = y[test]
    positive = truth[:, 0] > 0
    synthetic_test = {
        "samples": int(test.sum()),
        "count_mae": float(np.abs(predicted[:, 0] - truth[:, 0]).mean() * 12),
        "constant_count_mae": float(np.abs(y[train, 0].mean() - truth[:, 0]).mean() * 12),
        "width_fraction_mae_on_positive": float(np.abs(predicted[positive, 1] - truth[positive, 1]).mean() * .14),
        "constant_width_fraction_mae_on_positive": float(np.abs(y[train & (y[:, 0] > 0), 1].mean() - truth[positive, 1]).mean() * .14),
        "best_epoch": best_epoch,
    }

    recipes = read(PAIRS / "recipes.json")
    original = read(SYNTH / "runs" / "style_multilabel_phase12_01" / "cryptbb" / "recipes.json")
    real = [Path(recipes[2*i]["reference"]) for i in range(21)]
    real_x, real_scores = encode(frozen, real)
    with torch.inference_mode():
        real_pred = model(torch.from_numpy((real_x - mean) / std)).numpy()
    baseline = [PAIRS / f"{i:03d}_median.png" for i in range(21)]
    fixed, controlled, visibility, choices = [], [], [], []
    for i in range(21):
        case = original[i]
        base = recipes[2*i]["style"]
        font = PHASE1 / "base" / fonts[case["font_index"]]["path"]
        count = int(np.clip(round(float(real_pred[i, 0]) * 12), 0, 12))
        width = float(np.clip(real_pred[i, 1] * .14, .04, .14))
        seed = case["recipe"]["seed"]
        image, _, trace = render(block_style(base, count, width), case["text"], str(font), seed, 3)
        control, _, _ = render(block_style(base, 8, .09), case["text"], str(font), seed, 3)
        controlled.append(image)
        fixed.append(control)
        visibility.append(min(g["visible_fraction"] for g in trace["glyphs"]))
        choices.append({"reference": str(real[i]), "block_count": count, "width_fraction": width})
        image.save(HERE / f"controller_{i:03d}.png")
    _, baseline_scores = encode(frozen, baseline)
    _, fixed_scores = encode(frozen, fixed)
    _, controlled_scores = encode(frozen, controlled)
    results = read(RUN / "results.json")
    selected = [name for name, accepted in results["per_feature_gate"].items() if accepted]
    cutoff = results["results"]["731"]["thresholds"]["background_blocks"]
    index = FEATURES.index("background_blocks")
    score_groups = {"real": real_scores, "baseline": baseline_scores, "fixed_eight_blocks": fixed_scores, "controller": controlled_scores}
    prevalence = {name: float((score[:, index] >= cutoff).mean()) for name, score in score_groups.items()}
    gap = {name: float(np.mean([abs(real_scores[:, FEATURES.index(feature)].mean() - score[:, FEATURES.index(feature)].mean()) for feature in selected])) for name, score in score_groups.items() if name != "real"}
    report = {
        "protocol": "1000 train, 200 validation, 200 test; 70/13/13 disjoint source styles; 5 optimizer epochs, batch 1000 (one update per epoch), best synthetic validation checkpoint",
        "frozen_hybrid_sha256": sha(checkpoint),
        "train_source_styles": groups["train"],
        "validation_source_styles": groups["validation"],
        "test_source_styles": groups["test"],
        "history": history,
        "rejected_synthetic_render_attempts": rejected_renders,
        "synthetic_test": synthetic_test,
        "real_diagnostic": {"samples": 21, "block_detection_fraction": prevalence, "seven_feature_mean_score_gap": gap, "minimum_glyph_visibility": float(min(visibility)), "controller_counts": {str(count): sum(row["block_count"] == count for row in choices) for count in sorted({row["block_count"] for row in choices})}},
        "choices": choices,
        "limits": ["Synthetic control recovery does not establish real parameter accuracy.", "The same frozen hybrid guides and grades renders; no independent real block labels were used.", "Real references are historically exposed; no downstream OCR or GAN test was performed.", "Only block count and common size were learned; colours and placement remained fixed."],
    }
    (HERE / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "frozen_hybrid_sha256": sha(checkpoint)}, HERE / "controller.pt")
    sheet = Image.new("RGB", (1200, 110 * 6), "white")
    draw = ImageDraw.Draw(sheet)
    for col, name in enumerate(("Real", "Original render", "Fixed eight blocks", "Trained controller")):
        draw.text((col * 300 + 4, 3), name, fill="black")
    for row, i in enumerate((2, 4, 9, 11, 16, 18)):
        for col, source in enumerate((real[i], baseline[i], fixed[i], controlled[i])):
            image = Image.open(source).convert("RGB") if isinstance(source, Path) else source.copy()
            image.thumbnail((285, 85))
            sheet.paste(image, (col * 300 + 4, row * 110 + 20))
    sheet.save(HERE / "comparison.png")
    print(json.dumps({"synthetic_test": synthetic_test, "real_diagnostic": report["real_diagnostic"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
