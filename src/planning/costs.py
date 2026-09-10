"""Cost functions and risk map construction for path planning.

Implements all cost variants from the project plan:
  - Binary-mask cost
  - Soft-probability cost
  - Clearance-aware cost
  - Entropy-penalized uncertainty cost
  - Unified risk cost (proposed)
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from src.planning.grid import Point


# ---------------------------------------------------------------------------
# Cost function factories
# ---------------------------------------------------------------------------


def make_binary_cost(mask: np.ndarray) -> Callable[[Point], float]:
    """Cost = 0 on road, inf on non-road."""

    def cost(point: Point) -> float:
        r, c = point
        return 0.0 if mask[r, c] else float("inf")

    return cost


def make_probability_cost(
    probability: np.ndarray, eps: float = 1e-9
) -> Callable[[Point], float]:
    """Cost = -log(p)."""

    def cost(point: Point) -> float:
        r, c = point
        p = np.clip(float(probability[r, c]), eps, 1.0 - eps)
        return float(-math.log(p))

    return cost


def make_clearance_cost(
    probability: np.ndarray,
    distance: np.ndarray,
    lambda_b: float = 1.0,
    tau_d: float = 5.0,
    eps: float = 1e-9,
) -> Callable[[Point], float]:
    """Cost = -log(p) + λ_b * exp(-d/τ_d)."""

    def cost(point: Point) -> float:
        r, c = point
        p = np.clip(float(probability[r, c]), eps, 1.0 - eps)
        d = float(distance[r, c])
        return float(-math.log(p) + lambda_b * math.exp(-d / tau_d))

    return cost


def make_uncertainty_cost(
    probability: np.ndarray,
    uncertainty: np.ndarray,
    lambda_u: float = 2.0,
    eps: float = 1e-9,
) -> Callable[[Point], float]:
    """Entropy-penalized control: cost = -log(p) + λ_u * uncertainty.

    Uses normalized predictive entropy (in [0, 1]) as the uncertainty term.
    Raw epistemic std (bounded by ~0.5) is dominated by -log(p) on low-p
    pixels and makes this control a near-duplicate of Soft-probability A*.
    This local control is not an implementation of URA*.
    λ_u=2.0 keeps the uncertainty term in the same magnitude as the -log(p)
    term over the meaningful probability range.
    """

    def cost(point: Point) -> float:
        r, c = point
        p = np.clip(float(probability[r, c]), eps, 1.0 - eps)
        u = float(uncertainty[r, c])
        return float(-math.log(p) + lambda_u * u)

    return cost


def make_unified_risk_cost(
    calibrated_prob: np.ndarray,
    distance: np.ndarray,
    lambda_r: float = 2.0,
    lambda_b: float = 1.0,
    tau_d: float = 5.0,
    risk_cap: float = 1.0,
    eps: float = 1e-9,
) -> Callable[[Point], float]:
    """Proposed unified risk cost.

    J = λ_r * min(-log(q(x)), risk_cap) + λ_b * exp(-d(x)/τ_d)

    where q(x) is an uncertainty-discounted score. The base
    step cost is added separately by A* (step_cost in astar_search), so this
    function must NOT include a constant base term — otherwise the proposed
    method would pay every step twice and systematically over-divert.

    risk_cap bounds the per-pixel uncertainty penalty. Without it, the raw
    -log(q) term spans [0, ~4.3] for q in [0.01, 1.0] — an order of magnitude
    wider than the baselines' -log(p>=0.5) ∈ [0, 0.69]. On pixels where the
    segmentation is confidently wrong (false negatives with low epistemic std,
    so q≈p is low), the unbounded term makes A* detour extremely long paths
    just to avoid a handful of pixels. Capping the penalty bounds the maximum
    detour cost any single low-confidence pixel can impose.
    """

    return make_matched_score_cost(
        calibrated_prob,
        distance,
        lambda_r=lambda_r,
        lambda_b=lambda_b,
        tau_d=tau_d,
        risk_cap=risk_cap,
        use_boundary=True,
        eps=eps,
    )


def make_matched_score_cost(
    score: np.ndarray,
    distance: np.ndarray,
    lambda_r: float = 2.0,
    lambda_b: float = 1.0,
    tau_d: float = 5.0,
    risk_cap: float = 1.0,
    use_boundary: bool = True,
    eps: float = 1e-9,
) -> Callable[[Point], float]:
    """Build a controlled planning cost for p-vs-q factorial comparisons.

    Both probability sources receive the identical transform

        lambda_r * min(-log(score), risk_cap)

    and the clearance term is controlled by use_boundary. Keeping these
    choices fixed is essential when attributing a difference specifically to
    replacing the ensemble mean p with the uncertainty-discounted score q.

    The base geometric step cost is still added by astar_search.
    """

    def cost(point: Point) -> float:
        r, c = point
        value = np.clip(float(score[r, c]), eps, 1.0 - eps)
        risk = lambda_r * min(-math.log(value), risk_cap)
        if not use_boundary:
            return float(risk)
        d = float(distance[r, c])
        return float(risk + lambda_b * math.exp(-d / tau_d))

    return cost


# ---------------------------------------------------------------------------
# Risk / uncertainty map constructors
# ---------------------------------------------------------------------------


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    """Binary entropy normalized to [0, 1]."""
    p = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)).astype(
        np.float32
    )


def uncertainty_discounted_score(
    calibrated_prob: np.ndarray,
    epistemic_std: np.ndarray,
    kappa: float = 1.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute the uncertainty-discounted heuristic score used by the pipeline.

    q(x) = clip(p_cal(x) - κ * σ_epi(x), ε, 1-ε)

    The score ranks bridge and path candidates. It is not a calibrated
    confidence guarantee or a statistical lower confidence bound.
    """
    raw = calibrated_prob - kappa * epistemic_std
    return np.clip(raw, epsilon, 1.0 - epsilon).astype(np.float32)


def conservative_confidence_lower_bound(
    calibrated_prob: np.ndarray,
    epistemic_std: np.ndarray,
    kappa: float = 1.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Backward-compatible alias for :func:`uncertainty_discounted_score`.

    New code should use ``uncertainty_discounted_score`` because the historical
    name overstated the interpretation of this heuristic quantity.
    """
    return uncertainty_discounted_score(calibrated_prob, epistemic_std, kappa, epsilon)


def build_traversability(
    probability: np.ndarray,
    threshold: float = 0.5,
    use_largest_component: bool = True,
) -> np.ndarray:
    """Build a binary traversability mask from a probability map.

    If use_largest_component, keep only the largest connected road component.
    """
    from src.planning.grid import largest_connected_road

    binary = probability >= threshold
    if use_largest_component and np.any(binary):
        return largest_connected_road(binary)
    return binary


def build_combined_risk_map(
    probability: np.ndarray,
    traversable: np.ndarray,
    probability_weight: float = 0.4,
    uncertainty_weight: float = 0.3,
    boundary_weight: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build multi-component risk map (from the existing demo).

    Combines:
      - 1 - probability (inverse confidence)
      - normalized entropy (uncertainty)
      - 1/distance from boundary

    Returns:
        risk: (H, W) with inf for non-traversable, [0, 10] for traversable
        uncertainty: (H, W) normalized entropy
        boundary: (H, W) boundary component
        distance: (H, W) distance transform
    """
    from scipy import ndimage

    distance = ndimage.distance_transform_edt(traversable).astype(np.float32)
    uncertainty = normalized_entropy(probability)

    boundary_raw = np.zeros_like(probability, dtype=np.float32)
    boundary_raw[traversable] = 1.0 / (distance[traversable] + 1e-3)

    # Normalize boundary component
    boundary = np.zeros_like(boundary_raw, dtype=np.float32)
    selected = boundary_raw[traversable]
    if selected.size > 0:
        low = float(selected.min())
        high = float(selected.max())
        if high > low:
            boundary[traversable] = (selected - low) / (high - low)

    probability_risk = 1.0 - probability
    total_weight = probability_weight + uncertainty_weight + boundary_weight
    combined = (
        probability_weight * probability_risk
        + uncertainty_weight * uncertainty
        + boundary_weight * boundary
    ) / max(total_weight, 1e-6)

    risk = np.full(probability.shape, np.inf, dtype=np.float32)
    risk[traversable] = 10.0 * combined[traversable]
    return risk, uncertainty, boundary, distance
