"""Resume the City PARSeq run from its last valid checkpoint."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from runner_resume import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "config.city_phase3_100k.json")
