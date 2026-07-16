"""For datasets where filename == label (e.g., the 651 JoeyRetitling captchas),
generate a labels.csv from the filenames."""
import csv
import sys
from pathlib import Path


def main(root: Path) -> None:
    paths = sorted(p for p in root.glob("*") if p.suffix.lower() == ".png")
    out = root / "labels.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for p in paths:
            w.writerow([p.name, p.stem])
    print(f"wrote {out} with {len(paths)} rows")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
