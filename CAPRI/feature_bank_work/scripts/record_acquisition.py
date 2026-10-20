"""Record downloaded source bytes without executing or modifying them."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resources/feature_bank_01"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

receipts = read(RES / "acquisition.json")
files = []
for receipt in receipts:
    name = receipt["repository"].replace("/", "_")
    archive = RES / (name + ".zip")
    assert sha(archive) == receipt["sha256"], name
    for path in sorted((RES / name).rglob("*")):
        if path.is_file():
            files.append(dict(path=path.relative_to(RES).as_posix(), sha256=sha(path)))
(RES / "source_files.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
php = read(RES / "php_receipt.json")
assert sha(RES / "php-8.5.11.zip") == php["sha256"]
lines = ["# Feature-bank resource acquisition 01", "", "Date: 2026-09-28 UTC.", "",
    "**OBSERVED:** Downloaded and extracted five exact-revision source archives. Rechecked all archive hashes and recorded every extracted source file in [source_files.json](../../resources/feature_bank_01/source_files.json). Original archives are retained; extracted sources were not edited or executed.", "",
    "| Resource | Exact revision | Role |", "|---|---|---|"]
for row in receipts:
    repo, rev = row["repository"], row["revision"]
    lines.append(f"| [{repo}](https://github.com/{repo}/tree/{rev}) | `{rev}` | {'Withheld generalization renderer' if row['withheld'] else 'Candidate rendering resource'} |")
lines += ["", "**OBSERVED:** Portable PHP 8.5.11 CLI starts successfully with GD enabled. `gd_info()` reports FreeType and PNG support. The runtime archive SHA-256 matches the publisher value recorded in [php_receipt.json](../../resources/feature_bank_01/php_receipt.json). No system installation, PATH change, or shared Python environment change was made.", "",
    "**READ:** Runtime download URL/checksum came from the [official PHP Windows download page](https://www.php.net/downloads.php?os=windows). This establishes the acquisition reference, not compatibility of the five libraries.", "",
    "**OBSERVED:** The earlier sandbox network request failed with a socket-permission error. A later authorized acquisition succeeded. The interrupted download request left no resource directory before the successful run.", "",
    "**PROPOSED / PENDING:** Inspect code and separate font/background licenses, resolve dependency locks, test adapters, verify randomness and text-label integrity, then render and inspect 64 pilot images per renderer. Downloaded does not mean verified, trained, or eligible for the supervised corpus. No classifier or full-generation gate passed in this acquisition step.", "",
    "These are generic resources; none is identified as the original generator of the target datasets. Shared ancestry with target data remains unknown. Laminas remains withheld from training.", "",
    "## Existing stage-1 work", "",
    "**OBSERVED:** The separate [selection manifest](../../runs/feature_bank_01/manifest.json) contains 40 images from each historical validation partition, selected without classifier predictions. All 6,708 historical unique images matched saved file/pixel hashes and dimensions; exclusion checks and partition checks passed. Ten contact sheets were opened for assistant visual review. Feature-label drafts and user auditing remain pending; no audited real-image scoring is available.", "",
    "## Evidence", "", "- [Archive revisions and hashes](../../resources/feature_bank_01/acquisition.json)",
    "- [Source file hashes](../../resources/feature_bank_01/source_files.json)",
    "- [Runtime receipt](../../resources/feature_bank_01/php_receipt.json)",
    "- [Acquisition script](../../scripts/acquire_feature_bank.ps1)",
    "- [Receipt verification script](../../scripts/record_acquisition.py)", ""]
doc = ROOT / "docs/feature_bank/RESOURCE_ACQUISITION_01.md"
doc.parent.mkdir(parents=True, exist_ok=True)
doc.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(dict(archives_verified=6, source_files_recorded=len(files), report=str(doc))))
