"""Topology repair evaluation metrics.

Implements:
  - Connected component count
  - Correct / incorrect bridge rates
  - Bridge precision / recall
  - APLS improvement
  - Repair statistics
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def connected_component_count(mask: np.ndarray) -> int:
    """Count 8-connected components in a binary mask."""
    labels, count = ndimage.label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    return int(count)


def bridge_metrics(
    bridges: list,
    ground_truth: np.ndarray,
    overlap_threshold: float = 0.5,
    dilation_radius: int = 3,
) -> dict[str, float]:
    """Evaluate bridge quality against GT.

    A bridge is "correct" if the majority of its pixels fall within
    the dilated GT road mask, AND the two components it connects
    are in the same GT connected component.

    Args:
        bridges: list of Bridge objects from topology_repair
        ground_truth: (H, W) binary GT mask
        overlap_threshold: minimum fraction of bridge pixels on dilated GT
        dilation_radius: radius for GT mask dilation

    Returns dict of metrics.
    """
    if not bridges:
        return {
            "num_bridges": 0,
            "correct_bridges": 0,
            "incorrect_bridges": 0,
            "bridge_precision": float("nan"),
            "bridge_recall": float("nan"),
            "mean_bridge_overlap": float("nan"),
            "endpoint_consistency": float("nan"),
        }

    support = bridge_reference_support(
        bridges,
        ground_truth,
        overlap_threshold=overlap_threshold,
        dilation_radius=dilation_radius,
    )
    correct = int(sum(support))
    incorrect = int(len(support) - correct)

    # Keep the descriptive overlap and endpoint-consistency diagnostics used by
    # the original reporting code.  The binary decision itself is centralized
    # in ``bridge_reference_support`` so candidate-pool analyses use exactly
    # the same reference-support definition as accepted-bridge precision.
    gt_bool = ground_truth.astype(bool)
    structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=np.uint8)
    gt_dilated = ndimage.binary_dilation(gt_bool, structure=structure, iterations=1)
    gt_labels, _ = ndimage.label(
        gt_bool, structure=np.ones((3, 3), dtype=np.uint8)
    )

    def endpoint_component(row: int, col: int) -> int:
        if gt_labels[row, col] != 0:
            return int(gt_labels[row, col])
        row0 = max(0, row - dilation_radius)
        row1 = min(gt_labels.shape[0], row + dilation_radius + 1)
        col0 = max(0, col - dilation_radius)
        col1 = min(gt_labels.shape[1], col + dilation_radius + 1)
        window = gt_labels[row0:row1, col0:col1]
        coords = np.argwhere(window > 0)
        if len(coords) == 0:
            return 0
        offsets = coords - np.array([row - row0, col - col0])
        nearest = coords[int(np.argmin(np.sum(offsets.astype(np.float64) ** 2, axis=1)))]
        return int(window[int(nearest[0]), int(nearest[1])])

    overlaps: list[float] = []
    endpoint_matches = 0
    for bridge in bridges:
        on_gt = sum(
            1 for r, c in bridge.curve
            if 0 <= r < gt_dilated.shape[0]
            and 0 <= c < gt_dilated.shape[1]
            and gt_dilated[r, c]
        )
        ratio = on_gt / max(len(bridge.curve), 1)
        overlaps.append(ratio)
        start_component = endpoint_component(bridge.start.row, bridge.start.col)
        end_component = endpoint_component(bridge.end.row, bridge.end.col)
        endpoints_match = start_component != 0 and start_component == end_component
        endpoint_matches += int(endpoints_match)
    total = len(bridges)
    return {
        "num_bridges": total,
        "correct_bridges": correct,
        "incorrect_bridges": incorrect,
        "bridge_precision": correct / max(total, 1),
        "bridge_recall": float("nan"),
        "mean_bridge_overlap": float(np.mean(overlaps)),
        "endpoint_consistency": endpoint_matches / max(total, 1),
    }


def bridge_reference_support(
    bridges: list,
    ground_truth: np.ndarray,
    overlap_threshold: float = 0.5,
    dilation_radius: int = 3,
) -> list[bool]:
    """Return strict GT support labels for each bridge.

    A bridge is reference-supported when at least ``overlap_threshold`` of its
    rasterized centerline lies in the dilated GT road mask and both endpoints
    map to the same GT connected component.  Keeping this per-object helper
    separate makes it possible to evaluate both accepted bridges and the
    generated endpoint-pair candidate pool under one identical criterion.
    """
    if not bridges:
        return []

    gt_bool = ground_truth.astype(bool)
    structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=np.uint8)
    gt_dilated = ndimage.binary_dilation(gt_bool, structure=structure, iterations=1)
    gt_labels, _ = ndimage.label(
        gt_bool, structure=np.ones((3, 3), dtype=np.uint8)
    )

    def endpoint_component(row: int, col: int) -> int:
        if gt_labels[row, col] != 0:
            return int(gt_labels[row, col])
        row0 = max(0, row - dilation_radius)
        row1 = min(gt_labels.shape[0], row + dilation_radius + 1)
        col0 = max(0, col - dilation_radius)
        col1 = min(gt_labels.shape[1], col + dilation_radius + 1)
        window = gt_labels[row0:row1, col0:col1]
        coords = np.argwhere(window > 0)
        if len(coords) == 0:
            return 0
        offsets = coords - np.array([row - row0, col - col0])
        nearest = coords[int(np.argmin(np.sum(offsets.astype(np.float64) ** 2, axis=1)))]
        return int(window[int(nearest[0]), int(nearest[1])])

    labels: list[bool] = []
    for bridge in bridges:
        on_gt = sum(
            1 for r, c in bridge.curve
            if 0 <= r < gt_dilated.shape[0]
            and 0 <= c < gt_dilated.shape[1]
            and gt_dilated[r, c]
        )
        ratio = on_gt / max(len(bridge.curve), 1)
        start_component = endpoint_component(bridge.start.row, bridge.start.col)
        end_component = endpoint_component(bridge.end.row, bridge.end.col)
        labels.append(bool(
            ratio >= overlap_threshold
            and start_component != 0
            and start_component == end_component
        ))
    return labels


def topology_metrics(
    original_mask: np.ndarray,
    repaired_mask: np.ndarray,
    bridges: list,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Compute all topology repair metrics.

    Returns dict of metric_name → value.
    """
    orig_components = connected_component_count(original_mask)
    repaired_components = connected_component_count(repaired_mask)

    bridge_stats = bridge_metrics(bridges, ground_truth)

    # APLS-like improvement
    from src.eval.segmentation import compute_apls
    apls_orig = compute_apls(original_mask, ground_truth)
    apls_repaired = compute_apls(repaired_mask, ground_truth)

    # Total bridge length
    total_bridge_length = sum(len(b.curve) for b in bridges)

    return {
        "original_components": float(orig_components),
        "repaired_components": float(repaired_components),
        "component_reduction": float(orig_components - repaired_components),
        "apls_original": apls_orig,
        "apls_repaired": apls_repaired,
        "apls_improvement": apls_repaired - apls_orig,
        **bridge_stats,
        "total_bridge_length": float(total_bridge_length),
    }
