"""Summarize the post-hoc joint APLS/query diagnostic used in the JARS paper.

This script reads only frozen, saved test outputs.  It is deliberately
descriptive and does not perform a new hypothesis test or alter the protocol.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APLS_PATH = ROOT / "artifacts/raw_metrics/massachusetts_apls_reference.json"
ABLATION_PATH = ROOT / "artifacts/raw_metrics/factorial_ablation.csv"
OUTPUT_PATH = ROOT / "artifacts/raw_metrics/apls_query_joint_diagnostic.json"


def main() -> None:
    apls_payload = json.loads(APLS_PATH.read_text(encoding="utf-8"))
    apls_delta = {
        row["image_id"]: row["repaired"]["apls"] - row["original"]["apls"]
        for row in apls_payload["per_image"]
    }

    successes: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"original": [], "repaired": []}
    )
    with ABLATION_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["score"] != "q" or row["boundary"] != "1":
                continue
            mask = row["mask"]
            if mask not in {"original", "repaired"}:
                continue
            successes[row["image_id"]][mask].append(int(row["success"]))

    success_net_change: dict[str, float] = {}
    for image_id, by_mask in successes.items():
        if not by_mask["original"] or len(by_mask["original"]) != len(by_mask["repaired"]):
            raise RuntimeError(f"Incomplete matched query records for {image_id}")
        success_net_change[image_id] = (
            sum(by_mask["repaired"]) - sum(by_mask["original"])
        ) / len(by_mask["original"])

    if set(apls_delta) != set(success_net_change) or len(apls_delta) != 49:
        raise RuntimeError("Expected the same 49 Massachusetts images in both artifacts")

    apls_increased = {image_id for image_id, delta in apls_delta.items() if delta > 0}
    success_positive = {
        image_id for image_id, delta in success_net_change.items() if delta > 0
    }
    mismatch = {
        image_id
        for image_id in apls_increased
        if success_net_change[image_id] <= 0
    }

    payload = {
        "status": "post_hoc_descriptive",
        "inputs": {
            "apls": str(APLS_PATH.relative_to(ROOT)),
            "query_outcomes": str(ABLATION_PATH.relative_to(ROOT)),
            "matched_query_condition": "Original vs Repaired with score=q and boundary=on",
        },
        "counts": {
            "images": len(apls_delta),
            "apls_increased_images": len(apls_increased),
            "positive_net_success_change_images": len(success_positive),
            "apls_increased_without_positive_net_success_change": len(mismatch),
        },
        "interpretation": (
            "APLS and fixed-query completion summarize different units. This saved-output "
            "diagnostic motivates joint reporting; it is not evidence that APLS fails and "
            "is not a general metric-ranking test."
        ),
        "per_image": [
            {
                "image_id": image_id,
                "apls_delta": apls_delta[image_id],
                "success_rate_delta": success_net_change[image_id],
            }
            for image_id in sorted(apls_delta)
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
