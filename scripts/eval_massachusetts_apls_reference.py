#!/usr/bin/env python
"""Reference-implementation APLS audit for frozen Massachusetts road masks.

The official SpaceNet APLS code compares weighted road graphs using bidirectional
control-point insertion. Massachusetts provides raster masks rather than native
road-centerline GeoJSON, so this script converts every GT, original, and repaired
mask to a simplified skeleton graph in one shared pixel coordinate system. The
official APLS graph-comparison functions and their default graph parameters are
then used without selecting parameters from the locked test result.

This is an APLS-definition-compatible raster-to-graph audit, not a SpaceNet
challenge submission: its graph coordinates and lengths are pixels, not meters.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import random
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_mask
from src.data.splits import build_splits, resolve_image_paths
from src.planning.costs import conservative_confidence_lower_bound
from src.planning.topology_repair import repair_topology


def _load_reference_apls(reference_dir: Path):
    """Import the official APLS core without its unused GeoJSON/GDAL readers."""
    source_dir = reference_dir / "apls"
    if not (source_dir / "apls.py").exists():
        raise FileNotFoundError(f"Missing official APLS source at {source_dir}")

    # The core functions used below are pure NetworkX/Shapely code. The original
    # repository imports optional GIS readers at module import time although they
    # are not used for graph-to-graph scoring.
    for name in (
        "osgeo", "osgeo.gdal", "osgeo.ogr", "osgeo.osr", "geopandas",
        "rasterio", "affine", "cv2", "fiona", "graphTools", "wkt_to_G",
        "topo_metric", "sp_metric",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    osgeo = sys.modules["osgeo"]
    osgeo.gdal = sys.modules["osgeo.gdal"]
    osgeo.ogr = sys.modules["osgeo.ogr"]
    osgeo.osr = sys.modules["osgeo.osr"]

    # The reference code targets NetworkX 2.x, where ``G.node`` was an alias
    # for the node dictionary. NetworkX 3 removes that alias. Restoring this
    # read-only compatibility view changes neither the graph nor APLS logic.
    if not hasattr(nx.MultiGraph, "node"):
        nx.MultiGraph.node = property(lambda graph: graph._node)  # type: ignore[attr-defined]
    if not hasattr(nx.Graph, "node"):
        nx.Graph.node = property(lambda graph: graph._node)  # type: ignore[attr-defined]

    sys.path.insert(0, str(source_dir))
    import apls  # type: ignore[import-not-found]

    return apls


def _neighbors(pixel: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    row, col = pixel
    return [
        (row + dr, col + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and (row + dr, col + dc) in pixels
    ]


def skeleton_to_graph(mask: np.ndarray) -> nx.MultiGraph:
    """Convert an 8-connected raster skeleton to a weighted undirected graph."""
    skeleton = skeletonize(mask.astype(bool))
    rows, cols = np.nonzero(skeleton)
    pixels = set(zip(rows.tolist(), cols.tolist()))
    graph = nx.MultiGraph()
    if not pixels:
        return graph

    degrees = {pixel: len(_neighbors(pixel, pixels)) for pixel in pixels}
    node_pixels = {pixel for pixel, degree in degrees.items() if degree != 2}
    # A simple closed loop has no endpoints or junctions. One arbitrary node
    # creates a valid self-loop representation for its route geometry.
    if not node_pixels:
        node_pixels = {min(pixels)}

    node_ids = {pixel: index for index, pixel in enumerate(sorted(node_pixels))}
    for (row, col), node_id in node_ids.items():
        graph.add_node(node_id, x=float(col), y=float(row))

    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for start in sorted(node_pixels):
        for neighbor in _neighbors(start, pixels):
            directed = (start, neighbor)
            if directed in visited:
                continue
            path = [start, neighbor]
            previous, current = start, neighbor
            visited.add(directed)
            visited.add((neighbor, start))
            while current not in node_pixels:
                options = [candidate for candidate in _neighbors(current, pixels) if candidate != previous]
                if not options:
                    break
                following = options[0]
                visited.add((current, following))
                visited.add((following, current))
                previous, current = current, following
                path.append(current)

            if current not in node_pixels or len(path) < 2:
                continue
            start_id, end_id = node_ids[start], node_ids[current]
            points = [(float(col), float(row)) for row, col in path]
            length = float(sum(
                np.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
                for index in range(len(points) - 1)
            ))
            if length <= 0:
                continue
            graph.add_edge(
                start_id,
                end_id,
                geometry=LineString(points),
                length=length,
                inferred_speed_mps=1.0,
            )
    return graph


def reference_apls(
    apls_module, gt: np.ndarray, proposal: np.ndarray, sample_seed: int
) -> dict[str, float | int]:
    """Score one raster proposal through the official bidirectional APLS core."""
    graph_gt = skeleton_to_graph(gt)
    graph_proposal = skeleton_to_graph(proposal)
    if graph_gt.number_of_edges() == 0 or graph_proposal.number_of_edges() == 0:
        return {
            "apls": 0.0,
            "gt_onto_proposal": 0.0,
            "proposal_onto_gt": 0.0,
            "gt_nodes": graph_gt.number_of_nodes(),
            "proposal_nodes": graph_proposal.number_of_nodes(),
            "gt_edges": graph_gt.number_of_edges(),
            "proposal_edges": graph_proposal.number_of_edges(),
        }

    # ``make_graphs_yuge`` is the official large-graph branch. It preserves the
    # bidirectional APLS comparison while fixing the number of sampled control
    # nodes instead of materializing all-pairs paths for dense raster graphs.
    random.seed(sample_seed)
    with contextlib.redirect_stdout(io.StringIO()):
        graph_data = apls_module.make_graphs_yuge(
            graph_gt,
            graph_proposal,
            weight="length",
            max_nodes=500,
            max_snap_dist=4,
            allow_renaming=True,
            verbose=False,
            super_verbose=False,
        )
    if graph_data is None:
        raise RuntimeError("Official APLS make_graphs returned None")
    with contextlib.redirect_stdout(io.StringIO()):
        score, gt_to_proposal, proposal_to_gt = apls_module.compute_apls_metric(
            *graph_data[6:],
            graph_data[4],
            graph_data[5],
            min_path_length=10,
            verbose=False,
            super_verbose=False,
        )
    return {
        "apls": float(score),
        "gt_onto_proposal": float(gt_to_proposal),
        "proposal_onto_gt": float(proposal_to_gt),
        "gt_nodes": graph_gt.number_of_nodes(),
        "proposal_nodes": graph_proposal.number_of_nodes(),
        "gt_edges": graph_gt.number_of_edges(),
        "proposal_edges": graph_proposal.number_of_edges(),
    }


def bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(samples)])
    return tuple(float(value) for value in np.percentile(means, (2.5, 97.5)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, default=Path("artifacts/predictions/test"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--tau-gate", type=float, default=2.0)
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates every locked-test image")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/raw_metrics/massachusetts_apls_reference.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Write/resume an in-progress per-image score checkpoint.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    reference_dir = args.reference_dir.resolve()
    prediction_dir = args.prediction_dir if args.prediction_dir.is_absolute() else root / args.prediction_dir
    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        (args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint)
        if args.checkpoint
        else None
    )
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    apls_module = _load_reference_apls(reference_dir)

    image_ids = build_splits(root)["test_ids"]
    if args.max_images:
        image_ids = image_ids[:args.max_images]
    rows: list[dict] = []
    completed_ids: set[str] = set()
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("image_ids") != image_ids:
            raise RuntimeError("APLS checkpoint image IDs do not match this run")
        if checkpoint.get("parameters") != {
            "threshold": args.threshold,
            "kappa": args.kappa,
            "tau_gate": args.tau_gate,
        }:
            raise RuntimeError("APLS checkpoint parameters do not match this run")
        rows = checkpoint.get("per_image", [])
        completed_ids = {row["image_id"] for row in rows}
        print(f"Resuming from {len(rows)}/{len(image_ids)} checkpointed images", flush=True)

    for index, image_id in enumerate(image_ids, start=1):
        if image_id in completed_ids:
            continue
        probability = np.load(prediction_dir / f"{image_id}_prob.npy").astype(np.float32)
        uncertainty = np.load(prediction_dir / f"{image_id}_std.npy").astype(np.float32)
        gt = load_mask(resolve_image_paths(root, image_id)[1]).astype(bool)
        original = probability >= args.threshold
        q = conservative_confidence_lower_bound(probability, uncertainty, kappa=args.kappa)
        repaired, bridges, _ = repair_topology(original, q, tau_gate=args.tau_gate, w_p=1.0)

        sample_seed = int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8], 16)
        identity = reference_apls(apls_module, gt, gt, sample_seed)
        if not np.isclose(identity["apls"], 1.0, atol=1e-9):
            raise RuntimeError(f"Identity check failed for {image_id}: {identity['apls']}")
        original_score = reference_apls(apls_module, gt, original, sample_seed)
        repaired_score = reference_apls(apls_module, gt, repaired, sample_seed)
        rows.append({
            "image_id": image_id,
            "original": original_score,
            "repaired": repaired_score,
            "delta": float(repaired_score["apls"] - original_score["apls"]),
            "num_bridges": len(bridges),
        })
        print(
            f"[{index}/{len(image_ids)}] {image_id}: "
            f"APLS {original_score['apls']:.4f}->{repaired_score['apls']:.4f} "
            f"({repaired_score['apls'] - original_score['apls']:+.4f})",
            flush=True,
        )
        if checkpoint_path:
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "run_status": "in_progress",
                        "image_ids": image_ids,
                        "parameters": {
                            "threshold": args.threshold,
                            "kappa": args.kappa,
                            "tau_gate": args.tau_gate,
                        },
                        "images_completed": len(rows),
                        "per_image": rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    original_values = np.asarray([row["original"]["apls"] for row in rows], dtype=float)
    repaired_values = np.asarray([row["repaired"]["apls"] for row in rows], dtype=float)
    delta_values = repaired_values - original_values
    source_file = reference_dir / "apls" / "apls.py"
    result = {
        "experiment_id": "APLS-REFERENCE-001",
        "dataset": "Massachusetts Roads",
        "split": "test",
        "locked_test_split_used": True,
        "development_tuning_after_result": False,
        "metric": {
            "name": "SpaceNet APLS reference-core on raster-derived graphs",
            "source_file": str(source_file),
            "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "coordinate_system": "image pixels for both GT and proposal graphs",
            "important_scope": "Definition-compatible graph APLS audit; not a SpaceNet submission with native georeferenced centerlines.",
            "max_snap_distance_pixels": 4,
            "official_large_graph_branch": "make_graphs_yuge",
            "control_node_sample_size": 500,
            "control_node_sampling": "fixed per image using SHA-256(image_id)",
            "min_path_length_pixels": 10,
        },
        "parameters": {
            "threshold": args.threshold,
            "kappa": args.kappa,
            "tau_gate": args.tau_gate,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
        },
        "aggregate": {
            "images": len(rows),
            "original_mean": float(original_values.mean()),
            "repaired_mean": float(repaired_values.mean()),
            "delta_mean": float(delta_values.mean()),
            "delta_bootstrap_95_ci": bootstrap_interval(delta_values, args.bootstrap_samples, args.seed),
            "positive_image_fraction": float(np.mean(delta_values > 0)),
            "identity_checks_passed": len(rows),
        },
        "per_image": rows,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
