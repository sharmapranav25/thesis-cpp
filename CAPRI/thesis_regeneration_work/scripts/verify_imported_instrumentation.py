"""Check that stage observers did not change two complete test cells."""
import contextlib
import io
import json
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np

import test_imported_generation as test

run = test.ROOT / "runs/imported_all_datasets_01"
receipt = run / "observer_equivalence.json"
if receipt.exists():
    raise RuntimeError("Verification receipt exists; preserve prior results")
_, redirected = test.font_mapping()
cases = []
for dataset, mode in (("cryptbb", "c1_random"), ("city", "c2_curriculum")):
    saved = json.loads((run / dataset / mode / "result.json").read_text())
    random.seed(saved["seed"])
    np.random.seed(saved["seed"])
    with patch("builtins.open", redirected), contextlib.redirect_stdout(io.StringIO()):
        generator = test.imported.UniversalCaptchaGenerator(seed=saved["seed"], font_dirs=[str(test.FONT_DIR)])
    matches = []
    for expected in saved["samples"]:
        image, label = test.imported.generate_one(mode, saved["charset"], gen=generator, length=saved["length"])
        matches.append(label == expected["exported_label"] and image.size == (expected["width"], expected["height"]) and test.pixels(image) == expected["pixel_sha256"])
    cases.append({"dataset":dataset,"mode":mode,"checked":len(matches),"exact_matches":sum(matches)})
result = {"status":"PASS" if all(c["checked"] == c["exact_matches"] for c in cases) else "FAIL", "cases":cases,
          "scope":"Re-execute original generator without stage/generate observers; compare labels, dimensions, decoded pixels. No extra images saved.",
          "verification_script_sha256":test.sha(__file__)}
test.json_write(receipt, result)
print(json.dumps(result))
if result["status"] != "PASS":
    raise SystemExit(1)
