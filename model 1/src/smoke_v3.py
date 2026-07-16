"""Smoke-test all v3 data loaders + model forward passes before kicking off
the overnight pipeline. Catches:
  - detector data loader breakage on mixed-size images (new code path)
  - char-CNN crop logic on 400×120 images
  - PARSeq dataset hits a 400×120 image and resizes correctly
  - Each model runs forward on the loaded batch
Total runtime ~30s. Exits non-zero on any failure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "parseq"))

import torch
import yaml


def green(s): return f"\033[92m{s}\033[0m"
def red(s): return f"\033[91m{s}\033[0m"


def smoke_parseq():
    print("[1/5] PARSeq dataset + forward pass...")
    from src.parseq.train import build_parseq, CaptchaDir, captcha_collate
    from torch.utils.data import DataLoader
    cfg = yaml.safe_load(Path("configs/parseq_v2.yaml").read_text())
    ds = CaptchaDir(Path(cfg["data_dir"]), tuple(cfg["img_size"]), augment=False)
    loader = DataLoader(ds, batch_size=4, num_workers=0, collate_fn=captcha_collate)
    imgs, labels = next(iter(loader))
    assert imgs.shape == (4, 3, 32, 128), f"bad shape: {imgs.shape}"
    assert all(len(l) == 5 for l in labels), f"bad labels: {labels}"
    print(f"   ds={len(ds)} samples, batch shape {tuple(imgs.shape)}, labels {labels[:2]}")
    model = build_parseq(cfg).cuda().eval()
    with torch.inference_mode():
        out = model(imgs.cuda())
    print(f"   PARSeq fwd OK, logits shape {tuple(out.shape)}")
    return True


def smoke_detector():
    print("[2/5] detector dataset (mixed sizes!) + forward pass...")
    from src.detector.data import CaptchaBoxDataset
    from src.detector.model import CharBoxNet
    from torch.utils.data import DataLoader
    cfg = yaml.safe_load(Path("configs/detector_v2.yaml").read_text())
    ds = CaptchaBoxDataset(Path(cfg["data_dir"]), augment=False)
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    x, y, names = next(iter(loader))
    assert x.shape == (4, 3, 60, 200), f"bad x shape: {x.shape}"
    assert y.shape == (4, 5, 4), f"bad y shape: {y.shape}"
    assert (y[..., :2] >= 0).all() and (y[..., :2] <= 1).all(), "bbox out of [0,1]"
    print(f"   ds={len(ds)} samples, batch x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"   sample bbox (cx,cy,w,h): {y[0,0].tolist()}")
    model = CharBoxNet().cuda().eval()
    with torch.inference_mode():
        out = model(x.cuda())
    print(f"   detector fwd OK, output shape {tuple(out.shape)}")
    return True


def smoke_charcls(scale: int):
    print(f"[{scale_to_idx[scale]}/5] char-CNN @ {scale}×{scale} dataset + forward...")
    from src.charcls.data import CharCropDataset
    from src.charcls.model import build_charcls
    from torch.utils.data import DataLoader
    cfg = yaml.safe_load(Path(f"configs/charcls_{scale}_v2.yaml").read_text())
    ds = CharCropDataset(
        Path(cfg["data_dir"]),
        crop_size=cfg["crop_size"],
        pad_frac=cfg["pad_frac"],
        jitter_frac=0.0,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    x, y_l, y_c = next(iter(loader))
    assert x.shape == (4, 3, scale, scale), f"bad x shape: {x.shape}"
    assert y_l.shape == (4,) and y_c.shape == (4,)
    print(f"   ds={len(ds)} samples (5x rows), batch x={tuple(x.shape)}, y_l={y_l.tolist()}, y_c={y_c.tolist()}")
    model = build_charcls().cuda().eval()
    with torch.inference_mode():
        logit_l, logit_c = model(x.cuda())
    print(f"   char-CNN fwd OK, letter logits {tuple(logit_l.shape)}, case logits {tuple(logit_c.shape)}")
    return True


scale_to_idx = {64: 3, 96: 4, 128: 5}


if __name__ == "__main__":
    failures = []
    for name, fn in [
        ("parseq", smoke_parseq),
        ("detector", smoke_detector),
        ("charcls64", lambda: smoke_charcls(64)),
        ("charcls96", lambda: smoke_charcls(96)),
        ("charcls128", lambda: smoke_charcls(128)),
    ]:
        try:
            fn()
            print(green(f"  [OK] {name}\n"))
        except Exception as e:  # noqa: BLE001
            print(red(f"  [FAIL] {name}: {e}\n"))
            failures.append(name)
            import traceback; traceback.print_exc()

    if failures:
        print(red(f"\nFAIL: {failures}"))
        sys.exit(1)
    print(green("\nALL SMOKE TESTS PASSED — safe to launch overnight pipeline."))
