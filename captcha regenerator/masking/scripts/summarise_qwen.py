"""Aggregate per-type Qwen characterisation distributions across batch outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarise_type(type_dir: Path) -> dict:
    char_colors: Counter[str] = Counter()
    noise_color: Counter[str] = Counter()
    noise_types: Counter[str] = Counter()
    paths: Counter[str] = Counter()
    chosen: Counter[str] = Counter()
    n = 0
    for d in sorted(type_dir.iterdir()):
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text())
        ch = m.get("characterisation", {})
        for c in ch.get("char_colors", []):
            char_colors[c] += 1
        noise_color[ch.get("noise_color", "unknown")] += 1
        for nt in ch.get("noise_types", []):
            noise_types[nt] += 1
        paths[m.get("path", "?")] += 1
        chosen[m.get("chosen_color", "?")] += 1
        n += 1
    return {
        "n": n,
        "char_colors": dict(char_colors.most_common()),
        "noise_color": dict(noise_color.most_common()),
        "noise_types": dict(noise_types.most_common()),
        "path": dict(paths),
        "chosen_color": dict(chosen.most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    for t in ["city", "courier", "cryptbb", "darkbay"]:
        d = args.root / t
        if not d.exists():
            continue
        s = summarise_type(d)
        print(f"=== {t} (n={s['n']}) ===")
        print(f"  char_colors: {s['char_colors']}")
        print(f"  noise_color: {s['noise_color']}")
        print(f"  noise_types: {s['noise_types']}")
        print(f"  path:        {s['path']}")
        print(f"  chosen:      {s['chosen_color']}")


if __name__ == "__main__":
    main()
