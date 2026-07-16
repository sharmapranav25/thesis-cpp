"""Driver: run cluster->char mapping for all schemes."""
from __future__ import annotations

from pathlib import Path

from src.part_a.map_clusters import map_clusters


SCHEMES = [
    {"name": "courier_market", "allowlist": "0123456789"},
    {"name": "city", "allowlist": "0123456789"},
    {"name": "cryptbb_unsolved", "allowlist": None},   # mixed-case alphanum
    {"name": "joey_retitling", "allowlist": None},
]

DATA_ROOT = Path("E:/pranav/auto_synth/data")


def main():
    for sc in SCHEMES:
        print("=" * 70)
        print(f"[map] scheme: {sc['name']}, allowlist={sc['allowlist']!r}")
        print("=" * 70)
        try:
            map_clusters(DATA_ROOT / sc["name"],
                         n_per_cluster=8,
                         allowlist=sc["allowlist"])
        except Exception as e:
            print(f"[map] FAILED for {sc['name']}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
