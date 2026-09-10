"""Path planning baselines for comparison.

Implements all baseline methods from the project plan:
  1. GT Oracle A*          — upper bound (uses GT mask)
  2. Binary-mask A*        — thresholded prediction
  3. Soft-probability A*   — -log(p) cost
  4. Clearance-aware A*    — boundary distance penalty
  5. Entropy-penalized A* — internal uncertainty-regularization control
  6. Topology repair + unified risk A* — proposed method
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from src.planning.grid import (
    Point,
    SearchResult,
    astar_search,
    build_risk_map,
    choose_start_goal,
    largest_connected_road,
    validate_point,
)
from src.planning.costs import (
    build_traversability,
    uncertainty_discounted_score,
    make_binary_cost,
    make_clearance_cost,
    make_probability_cost,
    make_uncertainty_cost,
    make_unified_risk_cost,
    normalized_entropy,
)
from src.planning.topology_repair import repair_topology


def run_baseline_planners(
    probability: np.ndarray,
    ground_truth: np.ndarray,
    start: Point,
    goal: Point,
    uncertainty: np.ndarray | None = None,
    distance: np.ndarray | None = None,
    uncertainty_weight: float = 1.0,
    risk_weight: float = 4.0,
    lambda_r: float = 2.0,
    lambda_b: float = 1.0,
    tau_d: float = 5.0,
    threshold: float = 0.5,
    kappa: float = 1.0,
) -> list[SearchResult]:
    """Run all baseline planners on a single image.

    Args:
        probability: (H, W) road probability map [0, 1]
        ground_truth: (H, W) binary GT road mask
        start, goal: query points
        uncertainty: (H, W) uncertainty map (entropy or epistemic std)
        distance: (H, W) precomputed distance transform
        ...

    Returns:
        List of SearchResult for each baseline.
    """
    results: list[SearchResult] = []

    # Precompute shared structures. Keep ALL predicted road components (not just
    # the largest): dropping fragments would remove many query start/goal points
    # and suppress the very failure mode (fragmented networks) that topology
    # repair is meant to fix.
    binary_mask = probability >= threshold
    traversable = build_traversability(probability, threshold, use_largest_component=False)
    gt_component = largest_connected_road(ground_truth.astype(bool))

    if distance is None:
        from scipy import ndimage
        distance = ndimage.distance_transform_edt(traversable).astype(np.float32)

    # --- 1. GT Oracle A* (upper bound) ---
    gt_risk, gt_distance = build_risk_map(gt_component)
    # Snap to GT road if needed
    try:
        validate_point("start", start, gt_component)
        validate_point("goal", goal, gt_component)
        gt_start, gt_goal = start, goal
    except ValueError:
        # Use original start/goal anyway; failure expected if truly off-road
        gt_start, gt_goal = start, goal

    results.append(
        astar_search(
            name="GT Oracle A*",
            traversable=gt_component,
            risk=gt_risk,
            start=gt_start,
            goal=gt_goal,
        )
    )

    # --- 2. Binary-mask A* ---
    bin_risk, _ = build_risk_map(traversable)
    results.append(
        astar_search(
            name="Binary-mask A*",
            traversable=traversable,
            risk=bin_risk,
            start=start,
            goal=goal,
        )
    )

    # --- 3. Soft-probability A* ---
    prob_cost = make_probability_cost(probability)
    results.append(
        astar_search(
            name="Soft-probability A*",
            traversable=traversable,
            risk=np.zeros_like(probability),
            start=start,
            goal=goal,
            point_cost=prob_cost,
        )
    )

    # --- 4. Clearance-aware A* ---
    clearance_cost_fn = make_clearance_cost(probability, distance, lambda_b, tau_d)
    results.append(
        astar_search(
            name="Clearance-aware A*",
            traversable=traversable,
            risk=np.zeros_like(probability),
            start=start,
            goal=goal,
            point_cost=clearance_cost_fn,
        )
    )

    # --- 5. Entropy-penalized A* (not a URA* reproduction) ---
    if uncertainty is None:
        uncertainty = normalized_entropy(probability)
    ura_cost = make_uncertainty_cost(probability, uncertainty, uncertainty_weight)
    results.append(
        astar_search(
            name="Entropy-penalized A*",
            traversable=traversable,
            risk=np.zeros_like(probability),
            start=start,
            goal=goal,
            point_cost=ura_cost,
        )
    )

    # --- 6. Topology repair + unified risk A* (proposed) ---
    # q(x) = clip(p_cal - kappa*sigma, eps, 1-eps) is an uncertainty-discounted
    # score; plan on the topology-repaired map using q in the cost.
    if uncertainty is None:
        uncertainty = normalized_entropy(probability)
    q = uncertainty_discounted_score(probability, uncertainty, kappa)
    repaired_mask, _bridges, _skeleton = repair_topology(traversable, q)
    repaired_traversable = repaired_mask.astype(bool)
    from scipy import ndimage
    repaired_distance = ndimage.distance_transform_edt(repaired_traversable).astype(
        np.float32
    )
    unified_cost = make_unified_risk_cost(
        q, repaired_distance, lambda_r, lambda_b, tau_d
    )
    results.append(
        astar_search(
            name="Topology+Risk A*",
            traversable=repaired_traversable,
            risk=np.zeros_like(probability),
            start=start,
            goal=goal,
            point_cost=unified_cost,
        )
    )

    return results
