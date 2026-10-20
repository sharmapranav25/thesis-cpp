"""Read-only CryptBB frozen-hybrid feedback-loop proof of concept."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(r"E:\pranav\Final sem\sourcefree_synth\feature_joint_mapping_01")
RUN = ROOT / "run_conformer"
PAIRS = RUN / "real_generation" / "cryptbb"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from corpus import FEATURES  # noqa: E402
from models import Joint  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((192, 96), Image.Resampling.BILINEAR)
        return np.asarray(image).copy()


def main() -> None:
    torch.set_num_threads(4)
    checkpoint = RUN / "model_731.pt"
    protocol = json.loads((RUN / "real_generation" / "protocol.json").read_text())
    assert sha(checkpoint) == protocol["model_sha256"], "checkpoint hash changed"
    results = json.loads((RUN / "results.json").read_text())
    passing = [name for name, passed in results["per_feature_gate"].items() if passed]
    thresholds = results["results"]["731"]["thresholds"]
    recipes = json.loads((PAIRS / "recipes.json").read_text())
    assert len(recipes) == 42
    assert all(recipes[2*i]["mode"] == "median" and recipes[2*i+1]["mode"] == "learned" for i in range(21))
    assert all(recipes[2*i]["reference"] == recipes[2*i+1]["reference"] for i in range(21))
    assert all(recipes[2*i]["text"] == recipes[2*i+1]["text"] for i in range(21))

    paths = [Path(recipes[2*i]["reference"]) for i in range(21)]
    paths += [PAIRS / f"{i:03d}_median.png" for i in range(21)]
    paths += [PAIRS / f"{i:03d}_learned.png" for i in range(21)]
    assert all(path.is_file() for path in paths)
    model = Joint("conformer").eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    scores = []
    with torch.inference_mode():
        for start in range(0, len(paths), 16):
            images = np.stack([load_image(path) for path in paths[start:start + 16]])
            tensor = torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255
            logits, _ = model(tensor)
            scores.append(logits.sigmoid().numpy())
    scores = np.concatenate(scores)
    assert scores.shape == (63, 15)
    groups = {"real": scores[:21], "median": scores[21:42], "learned": scores[42:]}
    report = {
        "scope": "21 historically exposed CryptBB real references and 42 saved paired Phase 1 renders; no GAN trained",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "pair_protocol": str(RUN / "real_generation" / "protocol.json"),
        "sample_count_per_group": 21,
        "passing_features": passing,
        "features": {},
        "limits": [
            "The frozen hybrid both chose the feature flags and scores the images; agreement is self-consistency, not independent accuracy.",
            "No real feature labels, real text labels, GAN output, or downstream reader score are used.",
            "The saved pair uses the historical Phase 1 renderer and is not a fresh test of the official Phase 2 broad-font renderer.",
        ],
    }
    for name in passing:
        j = FEATURES.index(name)
        cutoff = thresholds[name]
        flags = np.array([bool(recipes[2*i]["flags"][name]) for i in range(21)])
        detected = {group: values[:, j] >= cutoff for group, values in groups.items()}
        report["features"][name] = {
            "threshold": cutoff,
            "requested_fraction": float(flags.mean()),
            "prevalence": {group: float(values.mean()) for group, values in detected.items()},
            "mean_score": {group: float(values[:, j].mean()) for group, values in groups.items()},
            "render_request_agreement": {group: float((detected[group] == flags).mean()) for group in ("median", "learned")},
        }
    for measure in ("prevalence", "mean_score"):
        report[f"mean_abs_real_gap_{measure}"] = {
            group: float(np.mean([abs(row[measure]["real"] - row[measure][group]) for row in report["features"].values()]))
            for group in ("median", "learned")
        }
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "sample_count_per_group": 21,
        "passing_features": passing,
        "mean_abs_real_gap_prevalence": report["mean_abs_real_gap_prevalence"],
        "mean_abs_real_gap_mean_score": report["mean_abs_real_gap_mean_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
