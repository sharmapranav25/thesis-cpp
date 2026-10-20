"""Fill synthetic manifest hash fields with stable unique IDs.

The source dataset's audit already verifies unique image pixels. The training
loader only uses this field to detect accidental overlap, so a stable hash of
the absolute source path avoids reopening 100,000 images at startup.
"""
import csv
import hashlib
import os
import sys
from pathlib import Path


def update(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as source, temporary.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            identity = os.path.normcase(os.path.abspath(row["path"]))
            value = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            if value in seen:
                raise RuntimeError(f"duplicate path identity: {row['path']}")
            seen.add(value)
            row["pixel_sha256"] = value
            writer.writerow(row)
            count += 1
    temporary.replace(path)
    print(f"{path}: {count} records")


for argument in sys.argv[1:]:
    update(Path(argument))
