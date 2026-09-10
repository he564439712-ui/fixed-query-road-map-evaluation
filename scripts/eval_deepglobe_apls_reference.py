#!/usr/bin/env python
"""Reference-core APLS audit for frozen DeepGlobe road masks.

The script applies the same raster-to-graph conversion and official SpaceNet
large-graph APLS core used by ``eval_massachusetts_apls_reference.py``. It also
reports Original/Repaired Dice, clDice, and fixed-query connectivity, producing
a single aligned topology-evidence record for a frozen DeepGlobe backbone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_massachusetts_apls_reference import (
    _load_reference_apls,
    bootstrap_interval,
    reference_apls,
)
from src.data.dataset import load_mask
from src.data.deepglobe import build_deepglobe_splits, resolve_deepglobe_paths
from src.data.queries import get_queries_for_image, load_queries
from src.eval.segmentation import compute_cldice, compute_dice
from src.planning.costs import conservative_confidence_lower_bound
from src.planning.topology_repair import repair_topology


LABEL_STRUCT = np.ones((3, 3), dtype=np.uint8)


def connectivity_counts(mask: np.ndarray, queries: list[dict]) -> tuple[int, int]:
    """Return the number of connected fixed queries and their denominator."""
    labels, _ = ndimage.label(mask, structure=LABEL_STRUCT)
    connected = 0
    for query in queries:
        start = labels[int(query["start_row"]), int(query["start_col"])]
        goal = labels[int(query["goal_row"]), int(query["goal_col"])]
        if start and goal and start == goal:
            connected += 1
    return connected, len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("artifacts/queries/deepglobe_queries.csv"))
    parser.add_argument("--backbone", required=True, help="Label recorded in the output provenance.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--tau-gate", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all frozen test images.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    reference_dir = args.reference_dir.resolve()
    prediction_dir = args.prediction_dir if args.prediction_dir.is_absolute() else root / args.prediction_dir
    output_path = args.output if args.output.is_absolute() else root / args.output
    checkpoint_path = None if args.checkpoint is None else (
        args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    )
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    image_ids = build_deepglobe_splits(root)["test_ids"]
    if args.limit:
        image_ids = image_ids[:args.limit]
    queries = load_queries(args.queries if args.queries.is_absolute() else root / args.queries)
    queries_by_image = {image_id: get_queries_for_image(queries, image_id) for image_id in image_ids}
    apls_module = _load_reference_apls(reference_dir)

    rows: list[dict] = []
    completed_ids: set[str] = set()
    parameters = {
        "threshold": args.threshold,
        "kappa": args.kappa,
        "tau_gate": args.tau_gate,
        "backbone": args.backbone,
    }
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("image_ids") != image_ids or checkpoint.get("parameters") != parameters:
            raise RuntimeError("APLS checkpoint does not match this frozen run")
        rows = checkpoint.get("per_image", [])
        completed_ids = {row["image_id"] for row in rows}
        print(f"Resuming from {len(rows)}/{len(image_ids)} checkpointed images", flush=True)

    for index, image_id in enumerate(image_ids, start=1):
        if image_id in completed_ids:
            continue
        probability = np.load(prediction_dir / f"{image_id}_prob.npy").astype(np.float32)
        std_path = prediction_dir / f"{image_id}_std.npy"
        uncertainty = np.load(std_path).astype(np.float32) if std_path.exists() else np.zeros_like(probability)
        gt = load_mask(resolve_deepglobe_paths(root, image_id)[1]).astype(bool)
        original = probability >= args.threshold
        q = conservative_confidence_lower_bound(probability, uncertainty, kappa=args.kappa)
        repaired, bridges, _ = repair_topology(original, q, tau_gate=args.tau_gate, w_p=1.0)
        sample_seed = int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8], 16)
        identity = reference_apls(apls_module, gt, gt, sample_seed)
        if not np.isclose(identity["apls"], 1.0, atol=1e-9):
            raise RuntimeError(f"Identity check failed for {image_id}: {identity['apls']}")
        original_apls = reference_apls(apls_module, gt, original, sample_seed)
        repaired_apls = reference_apls(apls_module, gt, repaired, sample_seed)
        original_connected, total_queries = connectivity_counts(original, queries_by_image[image_id])
        repaired_connected, _ = connectivity_counts(repaired, queries_by_image[image_id])
        rows.append({
            "image_id": image_id,
            "dice_original": compute_dice(original, gt),
            "dice_repaired": compute_dice(repaired, gt),
            "cldice_original": compute_cldice(original, gt)[0],
            "cldice_repaired": compute_cldice(repaired, gt)[0],
            "apls_original": original_apls["apls"],
            "apls_repaired": repaired_apls["apls"],
            "connected_original": original_connected,
            "connected_repaired": repaired_connected,
            "query_count": total_queries,
            "num_bridges": len(bridges),
        })
        print(
            f"[{index}/{len(image_ids)}] {image_id}: "
            f"APLS {original_apls['apls']:.4f}->{repaired_apls['apls']:.4f}; "
            f"queries {original_connected}/{total_queries}->{repaired_connected}/{total_queries}",
            flush=True,
        )
        if checkpoint_path:
            checkpoint_path.write_text(json.dumps({
                "run_status": "in_progress", "image_ids": image_ids,
                "parameters": parameters, "images_completed": len(rows), "per_image": rows,
            }, indent=2), encoding="utf-8")

    metric_pairs = ("dice", "cldice", "apls")
    aggregate = {"images": len(rows), "identity_checks_passed": len(rows)}
    for offset, metric in enumerate(metric_pairs):
        original_values = np.asarray([row[f"{metric}_original"] for row in rows], dtype=float)
        repaired_values = np.asarray([row[f"{metric}_repaired"] for row in rows], dtype=float)
        differences = repaired_values - original_values
        aggregate[metric] = {
            "original_mean": float(original_values.mean()),
            "repaired_mean": float(repaired_values.mean()),
            "delta_mean": float(differences.mean()),
            "delta_bootstrap_95_ci": bootstrap_interval(differences, args.bootstrap_samples, args.seed + offset),
        }
    total_queries = int(sum(row["query_count"] for row in rows))
    connected_original = int(sum(row["connected_original"] for row in rows))
    connected_repaired = int(sum(row["connected_repaired"] for row in rows))
    aggregate["query_connectivity"] = {
        "original": {"connected": connected_original, "total": total_queries, "ratio": connected_original / total_queries},
        "repaired": {"connected": connected_repaired, "total": total_queries, "ratio": connected_repaired / total_queries},
    }
    source_file = reference_dir / "apls" / "apls.py"
    result = {
        "experiment_id": "DEEPGLOBE-APLS-REFERENCE-001",
        "dataset": "DeepGlobe Road Extraction",
        "split": "frozen test",
        "backbone": args.backbone,
        "metric": {
            "name": "SpaceNet APLS reference-core on raster-derived graphs",
            "source_file": str(source_file),
            "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "coordinate_system": "image pixels for both GT and proposal graphs",
            "important_scope": "Definition-compatible graph APLS audit; not a SpaceNet submission with native georeferenced centerlines.",
            "official_large_graph_branch": "make_graphs_yuge",
            "control_node_sample_size": 500,
            "max_snap_distance_pixels": 4,
            "min_path_length_pixels": 10,
        },
        "parameters": {**parameters, "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.seed},
        "aggregate": aggregate,
        "per_image": rows,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()
    print(json.dumps(aggregate, indent=2))
    print(f"Saved: {output_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
