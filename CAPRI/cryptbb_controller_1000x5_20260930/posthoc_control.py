"""Post-hoc fixed five-wide-block check; no retraining."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("full_run", Path(__file__).with_name("train_test.py"))
full_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_run)
HERE, JOINT, PAIRS, PHASE1, RUN, SYNTH = (getattr(full_run, key) for key in ("HERE", "JOINT", "PAIRS", "PHASE1", "RUN", "SYNTH"))
Joint, block_style, encode, module, read = (getattr(full_run, key) for key in ("Joint", "block_style", "encode", "module", "read"))
from corpus import FEATURES


def main():
    torch.set_num_threads(4)
    frozen = Joint("conformer").eval()
    frozen.load_state_dict(torch.load(RUN / "model_731.pt", map_location="cpu", weights_only=True))
    render = module(RUN / "real_generation" / "renderer_snapshot.py", "posthoc_saved_renderer").render
    recipes = read(PAIRS / "recipes.json")
    original = read(SYNTH / "runs" / "style_multilabel_phase12_01" / "cryptbb" / "recipes.json")
    fonts = read(PHASE1 / "base" / "assets" / "fonts.json")
    images = []
    for i in range(21):
        case = original[i]
        font = PHASE1 / "base" / fonts[case["font_index"]]["path"]
        image, _, _ = render(block_style(recipes[2*i]["style"], 5, .13), case["text"], str(font), case["recipe"]["seed"], 3)
        image.save(HERE / f"posthoc_fixed5wide_{i:03d}.png")
        images.append(image)
    _, scores = encode(frozen, images)
    refs = [Path(recipes[2*i]["reference"]) for i in range(21)]
    _, real_scores = encode(frozen, refs)
    results = read(RUN / "results.json")
    cutoff = results["results"]["731"]["thresholds"]["background_blocks"]
    passing = [key for key, accepted in results["per_feature_gate"].items() if accepted]
    block = FEATURES.index("background_blocks")
    report = {
        "status": "post-hoc diagnostic, fixed parameters chosen after seeing controller outputs",
        "settings": {"count": 5, "width_fraction": .13},
        "block_detection_fraction": float((scores[:, block] >= cutoff).mean()),
        "block_detection_count": int((scores[:, block] >= cutoff).sum()),
        "seven_feature_mean_score_gap": float(np.mean([abs(real_scores[:, FEATURES.index(feature)].mean() - scores[:, FEATURES.index(feature)].mean()) for feature in passing])),
    }
    (HERE / "posthoc_ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
