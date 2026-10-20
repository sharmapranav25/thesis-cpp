"""Launch the shared trainer with an explicitly selected configuration file."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from runner import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "config.v3_first10k.json")
