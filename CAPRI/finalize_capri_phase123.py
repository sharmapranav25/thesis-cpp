"""Write provenance and the outer integrity manifest for the unified CAPRI release."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(root):
    project = Path(r"E:\pranav\Final sem\sourcefree_synth")
    city_run = project / "runs/capri_phase3_city_02"
    city_dataset = Path(r"E:\pranav\Final sem\synth_data\city_capri_phase3_5digit_100k")
    target_audit = city_dataset / "audit.json"
    provenance = {
        "release": "CAPRI_phase123_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase1": {
            "source": str(project / "releases/CAPRI_v2"),
            "integrity_sha256": sha(root / "phase1/INTEGRITY.json"),
            "role": "CV measurements and frozen feature inference mapped to general rendering functions",
        },
        "phase2": {
            "source": str(project / "releases/CAPRI_typography_v3"),
            "integrity_sha256": sha(root / "phase2/INTEGRITY.json"),
            "role": "35-font broad synthetic generation with variable lengths and typography",
        },
        "phase3": {
            "source": str(project / "style_pipeline/phase3_reader_v2"),
            "role": "synthetic-only readers, frozen fit-only real inference, uncertainty-aware scheme and targeted generation",
            "architectures": ["CRNN CTC", "SVTR-style CTC", "PARSeq"],
            "variants": ["plain", "auxiliary count and digit/lowercase/uppercase heads"],
            "upstream": {
                "parseq": {
                    "url": "https://github.com/baudm/parseq.git",
                    "commit": "1902db043c029a7e03a3818c616c06600af574be",
                    "license": "Apache-2.0; CRNN subimplementation carries its included MIT license",
                },
                "OpenOCR": {
                    "url": "https://github.com/Topdu/OpenOCR.git",
                    "commit": "1ccfc6ee6161f7133b192e16af3a9265273997ba",
                    "license": "Apache-2.0",
                },
            },
        },
        "verified_city_phase3_example": {
            "run": str(city_run),
            "scheme_sha256": sha(city_run / "inferred_scheme.json"),
            "comparison_sha256": sha(city_run / "comparison.json"),
            "target_dataset": str(city_dataset),
            "target_audit_sha256": sha(target_audit) if target_audit.is_file() else None,
            "target_dataset_available_when_packaged": city_dataset.is_dir(),
            "note": "Historical example only; datasets and checkpoints are not bundled or required by this package.",
        },
        "boundaries": {
            "runtime_vlm": False,
            "target_generator_code_or_fonts": False,
            "real_transcription_labels_for_training_or_scheme": False,
            "validation_predictions_used_for_scheme": False,
            "optional_image_refiners_are_phase3": False,
            "production_recognizer_training_included": False,
        },
        "known_limits": [
            "Real prediction histograms are not ground-truth accuracy.",
            "The current renderer retains count-dependent layout; layout stress metrics do not prove shortcut resistance.",
            "All cached real-family data is historically exposed development data.",
            "Targeted generation requires human review of inferred_scheme.json.",
        ],
    }
    (root / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    manifest = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "PACKAGE_INTEGRITY.json":
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        manifest[path.relative_to(root).as_posix()] = sha(path)
    (root / "PACKAGE_INTEGRITY.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(manifest), "manifest_sha256": sha(root / "PACKAGE_INTEGRITY.json")}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
