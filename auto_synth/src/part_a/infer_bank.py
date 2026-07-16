"""Infer the full character bank from observed chars.

User's insight: captchas almost always use a structured alphabet (digits,
uppercase letters, lowercase letters, or some combination). If we observed
ANY digit in our cluster->char mapping, the actual bank is 0-9. If we saw any
uppercase, the bank includes A-Z. Etc.

Strategy: detect which type-classes appear in observed chars, expand each
class to the full set, save the inferred bank.

For chars in the inferred bank that we DON'T have a cluster for ("missing
glyphs"), this script flags them. compose.py can either:
  (a) skip them and reduce the effective bank to what we have (default), or
  (b) fall back to a TTF render (configurable per scheme).
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path


DIGITS = set(string.digits)
UPPER = set(string.ascii_uppercase)
LOWER = set(string.ascii_lowercase)


def expand_bank(observed: set[str]) -> set[str]:
    bank: set[str] = set(observed)
    if observed & DIGITS:
        bank |= DIGITS
    if observed & UPPER:
        bank |= UPPER
    if observed & LOWER:
        bank |= LOWER
    return bank


def infer_bank(scheme_dir: Path):
    scheme_dir = Path(scheme_dir)
    manifest_path = scheme_dir / "alphabet_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cluster_to_char = manifest.get("cluster_to_char", {})
    observed = {v["char"] for v in cluster_to_char.values()
                if v.get("char") and v["char"] != "?"}
    full_bank = expand_bank(observed)
    missing = full_bank - observed
    summary = {
        "observed_chars": sorted(observed),
        "inferred_full_bank": sorted(full_bank),
        "missing_glyphs": sorted(missing),
        "n_observed": len(observed),
        "n_inferred": len(full_bank),
        "n_missing": len(missing),
    }
    manifest["inferred_bank"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[infer_bank] {scheme_dir.name}:")
    print(f"  observed ({len(observed)}): {''.join(sorted(observed))}")
    print(f"  inferred bank ({len(full_bank)}): {''.join(sorted(full_bank))}")
    print(f"  missing glyphs ({len(missing)}): {''.join(sorted(missing))}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme-dir", required=True, type=Path)
    args = ap.parse_args()
    infer_bank(args.scheme_dir)
