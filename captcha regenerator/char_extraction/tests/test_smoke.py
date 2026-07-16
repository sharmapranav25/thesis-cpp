"""End-to-end smoke tests on synthetic images.

These are not extraction-quality tests — they validate the pipeline
plumbing: types, registry, router, I/O writer, CLI dataclass shapes.
Method-quality is measured separately by the eval harness against
labeled samples (see M2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from char_extraction import extract
from char_extraction.api import _DEFAULT_CONFIG_DIR
from char_extraction.io_utils import load_image, save_image, write_extraction
from char_extraction.methods import list_methods
from char_extraction.types import CharBox, ExtractionResult


def _make_synthetic(h: int = 60, w: int = 200) -> np.ndarray:
    """Light bg with a few dark rectangles (stand-in characters)."""
    img = np.full((h, w, 3), 230, dtype=np.uint8)
    rects = [(20, 15, 40, 45), (60, 12, 80, 48), (100, 14, 120, 46), (140, 13, 160, 47)]
    for x1, y1, x2, y2 in rects:
        img[y1:y2, x1:x2] = (30, 30, 30)
    return img


def test_otsu_cc_registered():
    methods = list_methods()
    assert "otsu_cc" in methods


def test_extract_returns_charboxes():
    img = _make_synthetic()
    result = extract(image=img, captcha_id="synth", captcha_type=None)
    assert isinstance(result, ExtractionResult)
    assert result.captcha_id == "synth"
    assert result.method == "otsu_cc"
    assert result.image_size == (200, 60)
    assert len(result.characters) >= 4
    for c in result.characters:
        assert isinstance(c, CharBox)
        assert c.width > 0 and c.height > 0


def test_charboxes_left_to_right():
    img = _make_synthetic()
    result = extract(image=img, captcha_id="synth", captcha_type=None)
    centers = [c.x_center for c in result.characters]
    assert centers == sorted(centers), "characters must be left-to-right ordered"
    indices = [c.index for c in result.characters]
    assert indices == list(range(len(result.characters))), "indices must be 0..n-1 in order"


def test_write_extraction_layout(tmp_path: Path):
    img = _make_synthetic()
    result = extract(image=img, captcha_id="synth", captcha_type=None)
    out_base = write_extraction(result, img, tmp_path)
    assert (out_base / "original.png").exists()
    assert (out_base / "mask.png").exists()
    assert (out_base / "overlay.png").exists()
    assert (out_base / "metadata.json").exists()
    chars_dir = out_base / "characters"
    assert chars_dir.is_dir()
    assert len(list(chars_dir.glob("*.png"))) == len(result.characters)
    meta = json.loads((out_base / "metadata.json").read_text())
    assert meta["captcha_id"] == "synth"
    assert meta["image_size"] == [200, 60]
    assert len(meta["characters"]) == len(result.characters)
    for entry in meta["characters"]:
        assert "bbox" in entry and len(entry["bbox"]) == 4
        assert entry["crop_path"].startswith("characters/")


def test_default_config_loads():
    cfg_path = _DEFAULT_CONFIG_DIR / "_default.yaml"
    assert cfg_path.exists(), f"default config missing at {cfg_path}"


def test_per_type_configs_load():
    for t in ("city", "courier", "darkbay", "cryptbb"):
        assert (_DEFAULT_CONFIG_DIR / f"{t}.yaml").exists(), f"missing config for {t}"


def test_image_roundtrip(tmp_path: Path):
    img = _make_synthetic()
    p = tmp_path / "synth.png"
    save_image(img, p)
    loaded = load_image(p)
    assert loaded.shape == img.shape
    assert loaded.dtype == img.dtype
