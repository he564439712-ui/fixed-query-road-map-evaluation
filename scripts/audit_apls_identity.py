#!/usr/bin/env python
"""Create a compact audit of archived raster-graph APLS identity checks.

Each reference-APLS evaluator scores ``APLS(GT, GT)`` before scoring an image's
prediction and raises if the result differs from one by more than 1e-9.  The
published result JSON records how many per-image checks completed.  This script
verifies those archived invariants and writes a reviewer-facing summary; it
does not rerun or retune any locked-test evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNS = (
    ("Massachusetts / U-Net ensemble", "massachusetts_apls_reference.json"),
    ("DeepGlobe / U-Net ensemble", "deepglobe_unet_apls_reference.json"),
    ("DeepGlobe / DeepLabV3-ResNet50", "deepglobe_deeplabv3_apls_reference.json"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/raw_metrics/apls_identity_audit.json"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    raw = root / "artifacts" / "raw_metrics"
    runs = []
    for label, filename in RUNS:
        path = raw / filename
        record = json.loads(path.read_text(encoding="utf-8"))
        aggregate = record["aggregate"]
        images = int(aggregate["images"])
        passed = int(aggregate["identity_checks_passed"])
        if passed != images:
            raise RuntimeError(
                f"APLS identity audit incomplete for {filename}: {passed}/{images}"
            )
        metric = record["metric"]
        runs.append({
            "label": label,
            "source_result": str(path.relative_to(root)).replace("\\", "/"),
            "images": images,
            "identity_checks_passed": passed,
            "identity_definition": "APLS(GT, GT) within absolute tolerance 1e-9",
            "coordinate_system": metric["coordinate_system"],
            "control_node_sample_size": metric["control_node_sample_size"],
            "max_snap_distance_pixels": metric["max_snap_distance_pixels"],
            "min_path_length_pixels": metric["min_path_length_pixels"],
        })

    result = {
        "audit_id": "APLS-IDENTITY-AUDIT-001",
        "purpose": (
            "Archived implementation sanity check; not a benchmark comparison "
            "or a new locked-test selection step."
        ),
        "implementation": {
            "evaluator": "public SpaceNet APLS reference core on raster-derived graphs",
            "identity_precondition": "Each source evaluator raises before result output if APLS(GT, GT) differs from 1 by more than 1e-9.",
            "scope": "Definition-compatible raster-graph audit, not a native-georeferenced SpaceNet submission.",
        },
        "runs": runs,
        "total_identity_checks_passed": sum(run["identity_checks_passed"] for run in runs),
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
