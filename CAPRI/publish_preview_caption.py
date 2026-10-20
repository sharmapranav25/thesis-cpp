"""Publish only the requested saved-preview caption revision."""
from pathlib import Path
import hashlib
import json
import shutil

from PIL import Image

HERE = Path(__file__).resolve().parent
STAGE = HERE / "thesis_regeneration_work"
TARGET = Path(r"E:\pranav\Final sem\sourcefree_synth")
FILES = (
    "runs/imported_all_datasets_01/all_datasets_ab.png",
    "scripts/test_imported_generation.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    state_path = HERE / "thesis_publish_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if TARGET.resolve() != Path(state["target"]).resolve():
        raise RuntimeError("Unexpected destination")
    for relative in FILES:
        source, target = STAGE / relative, TARGET / relative
        if not source.is_file() or not target.is_file():
            raise RuntimeError("Expected existing source and destination: " + relative)
        if sha(target) not in (sha(source), state["known_existing"].get(relative)):
            raise RuntimeError("Destination independently changed: " + relative)
    with Image.open(STAGE / FILES[0]) as new, Image.open(TARGET / FILES[0]) as old:
        box = (0, 158, 1440, 1157)
        if new.convert("RGB").crop(box).tobytes() != old.convert("RGB").crop(box).tobytes():
            raise RuntimeError("CAPTCHA comparison rows changed")
    for relative in FILES:
        source, target = STAGE / relative, TARGET / relative
        shutil.copyfile(source, target)
        if sha(source) != sha(target):
            raise RuntimeError("Copy verification failed: " + relative)
        state["known_existing"][relative] = sha(target)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"published": list(FILES), "captcha_rows_unchanged": True}))


if __name__ == "__main__":
    main()
