"""Frozen definitions, image identities, provenance and strict annotation gates."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps

FEATURES = {
    "straight_interference": "Added approximately straight strokes; glyph strokes are hard negatives.",
    "curved_interference": "Added curved/wavy strokes; curved glyphs are hard negatives.",
    "dot_scatter": "Small isolated noise marks; punctuation/dotted fonts are distinct.",
    "decorative_shapes": "Added geometric decorations; glyph O is not circular noise.",
    "periodic_grid": "Repeated intersecting/parallel structure; unrelated lines are insufficient.",
    "background_texture": "Surface texture; distinguish foreground clutter when possible.",
    "background_gradient": "Smooth spatial brightness/color change, not global brightness.",
    "background_blocks": "Patch-like color regions, not isolated characters.",
    "hollow_lettering": "Letter strokes surrounding an unfilled interior; not filled bordered text.",
    "glyph_border": "Additional contour around filled lettering; not hollow lettering.",
    "glyph_shadow": "Offset duplicate-like structure associated with text.",
    "nonlinear_text_deformation": "Visibly bent/deformed lettering; intrinsic font design can be uncertain."
}
STATES = {"present", "absent", "uncertain"}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def load(path):
    with Image.open(path) as original:
        rgba = ImageOps.exif_transpose(original).convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")

def pixel_hash(image):
    h = hashlib.sha256(str(image.size).encode("ascii"))
    h.update(image.convert("RGB").tobytes())
    return h.hexdigest()

def new_run(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=False)
    return p

def audit_annotations(manifest, drafts, audited):
    """Validate a separately supplied audit; never promote drafts automatically."""
    ids = {r["id"]: r for r in manifest["images"]}
    if not audited.get("reviewer", "").strip() or audited.get("role") != "user":
        raise ValueError("A named user audit is required")
    if audited.get("manifest_sha256") != drafts["manifest_sha256"]:
        raise ValueError("Audit refers to another manifest")
    rows = audited.get("images", [])
    if len(rows) != len(ids) or {r["id"] for r in rows} != set(ids):
        raise ValueError("Audit must cover each selected image exactly once")
    disagreements = []
    draft_by_id = {r["id"]: r for r in drafts["images"]}
    for row in rows:
        if row.get("reviewed") is not True:
            raise ValueError("Every image needs explicit user review")
        if set(row["labels"]) != set(FEATURES) or not set(row["labels"].values()) <= STATES:
            raise ValueError("Invalid or missing feature labels")
        for feature, label in row["labels"].items():
            before = draft_by_id[row["id"]]["labels"][feature]
            if before != label:
                disagreements.append(dict(id=row["id"], feature=feature, draft=before, audited=label))
    return disagreements
