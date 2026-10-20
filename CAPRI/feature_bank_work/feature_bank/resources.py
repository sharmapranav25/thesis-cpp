"""Source integrity, explicit asset eligibility, and minimal execution lock."""
import string
from pathlib import Path
from fontTools.ttLib import TTFont
from .core import read, write, digest

def inventory(root, output, project):
    root, output, project = Path(root), Path(output), Path(project)
    original = read(root / "source_files.json")
    for row in original:
        if digest(root / row["path"]) != row["sha256"]:
            raise ValueError("Downloaded source changed: " + row["path"])
    frozen = read(project / "vendor/PROVENANCE.json")
    preserved = []
    for row in frozen["files"]:
        for path in (Path(row["original_path"]), project/"vendor/imported_repo"/row["relative_path"]):
            if digest(path) != row["sha256"]:
                raise ValueError("Historical source hash mismatch: " + str(path))
            preserved.append(str(path))
    records, assets, sources = [], [], {}
    for receipt in read(root / "acquisition.json"):
        folder = root / receipt["repository"].replace("/", "_")
        src = next(p for p in folder.iterdir() if p.is_dir())
        sources[receipt["repository"]] = str(src.resolve())
        composer = read(src / "composer.json")
        records.append(dict(receipt, source=str(src), declared_require=composer.get("require", {}),
                            declared_code_license=composer.get("license"),
                            upstream_lock=digest(src/"composer.lock") if (src/"composer.lock").exists() else None,
                            status="SOURCE_AUDITED_NOT_CAPABILITY_VERIFIED"))
        for path in sorted(src.rglob("*")):
            if path.suffix.lower() not in {".ttf", ".otf", ".png"}:
                continue
            row = dict(path=str(path.resolve()), sha256=digest(path), source=receipt["repository"],
                       license_status="unresolved_not_training_eligible")
            if path.suffix.lower() in {".ttf", ".otf"}:
                with TTFont(str(path)) as font:
                    cmap = font.getBestCmap() or {}
                    row["missing_ascii_alphanumeric"] = [c for c in string.ascii_letters+string.digits
                                                          if ord(c) not in cmap or cmap[ord(c)] == ".notdef"]
                    row["family"] = font["name"].getDebugName(1)
                license_path = path.with_name(path.stem+"_license.txt")
                if license_path.exists() and "SIL Open Font License" in license_path.read_text(encoding="utf-8"):
                    row.update(license_status="OFL-1.1_notice_present", license_path=str(license_path),
                               license_sha256=digest(license_path))
                elif path.name == "AHGBold.ttf":
                    row.update(license_status="freeware_with_notice", license_path=str(src/"README.FONT.txt"))
            assets.append(row)
    # Hash lock for the actual minimal direct-load pilot. Not a Composer lock:
    # no dependency resolution is claimed and full packages remain pending.
    lock = dict(schema="direct-load-pilot-v1", php=read(root/"php_receipt.json"),
                sources=records, extensions=["gd", "mbstring", "fileinfo"],
                full_composer_resolution="pending; Gregwar declares unused symfony/finder; Laminas withheld pending dependency installation",
                pilot_source_files_sha256=digest(root/"source_files.json"))
    write(output/"resource_inventory.json", dict(resources=records, assets=assets,
          downloaded_source_files_unchanged=len(original), historical_source_paths_unchanged=len(preserved)))
    write(output/"pilot_runtime.lock.json", lock)
    return sources, assets
