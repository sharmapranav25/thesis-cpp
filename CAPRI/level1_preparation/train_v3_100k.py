"""Launch PARSeq on the full v3 100k configuration."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from runner import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "config.v3_100k.json")
