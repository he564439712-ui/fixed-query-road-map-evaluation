"""Conservative topology repair for road segmentation.

Implements the pipeline from the project plan:
  1. Skeletonize the predicted binary mask
  2. Detect degree-1 endpoints
  3. Generate candidate endpoint pairs within radius
  4. Compute connection cost C(e) with 5 terms
  5. Accept connections with C(e) < τ
  6. Build repaired road graph
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


@dataclass
class Endpoint:
    row: int
    col: int
    direction: tuple[float, float]  # unit vector pointing outward
    width: float  # estimated road width at this point
    component_id: int  # connected component label


@dataclass
class Bridge:
    start: Endpoint
    end: Endpoint
    curve: list[tuple[int, int]]  # pixel path of the bridge
    cost: float
    cost_components: dict[str, float]


# ---------------------------------------------------------------------------
# Skeletonization and endpoint detection
# ---------------------------------------------------------------------------


def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to single-pixel skeleton."""
    return skeletonize(mask.astype(bool))


def estimate_direction(
    skeleton: np.ndarray, point: tuple[int, int], window: int = 5
) -> tuple[float, float] | None:
    """Estimate road direction at an endpoint.

    Looks at local skeleton pixels within `window` and computes
    the average direction vector from the endpoint outward.
    """
    r, c = point
    h, w = skeleton.shape
    vectors = []
    for dr in range(-window, window + 1):
        for dc in range(-window, window + 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and skeleton[nr, nc]:
                vectors.append((float(dr), float(dc)))

    if not vectors:
        return None

    avg_dr = float(np.mean([v[0] for v in vectors]))
    avg_dc = float(np.mean([v[1] for v in vectors]))
    norm = math.hypot(avg_dr, avg_dc)
    if norm < 1e-6:
        return None
    # At a degree-1 endpoint the only skeleton pixels nearby lie along the
    # branch, i.e. INTO the road body (the gap has no skeleton). The direction
    # convention is "pointing outward" (toward the gap), so negate the mean.
    return (-avg_dr / norm, -avg_dc / norm)


def estimate_width(
    mask: np.ndarray, point: tuple[int, int], max_radius: int = 15
) -> float:
    """Estimate road width at a point by measuring cross-section.

    Uses the distance transform as a proxy for half-width.
    """
    distance = ndimage.distance_transform_edt(mask)
    r, c = point
    h, w = mask.shape
    r0 = max(0, r - max_radius)
    r1 = min(h, r + max_radius + 1)
    c0 = max(0, c - max_radius)
    c1 = min(w, c + max_radius + 1)
    local_dist = distance[r0:r1, c0:c1]
    local_mask = mask[r0:r1, c0:c1]
    if not np.any(local_mask):
        return 1.0
    return float(2.0 * np.max(local_dist[local_mask]))


def detect_endpoints(
    skeleton: np.ndarray,
    mask: np.ndarray,
    component_labels: np.ndarray,
) -> list[Endpoint]:
    """Detect all degree-1 endpoints in the skeleton.

    For each endpoint, estimate direction, width, and component ID.
    """
    h, w = skeleton.shape
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndimage.convolve(
        skeleton.astype(np.uint8), kernel, mode="constant", cval=0
    )
    # Subtract self
    neighbor_count = neighbor_count - skeleton.astype(np.uint8)

    endpoint_mask = skeleton & (neighbor_count == 1)
    endpoint_coords = np.argwhere(endpoint_mask)

    endpoints: list[Endpoint] = []
    for r, c in endpoint_coords:
        direction = estimate_direction(skeleton, (int(r), int(c)))
        if direction is None:
            # Default: point away from the single neighbor
            direction = _direction_from_neighbor(skeleton, int(r), int(c))

        width = estimate_width(mask, (int(r), int(c)))
        comp_id = int(component_labels[int(r), int(c)])

        endpoints.append(
            Endpoint(
                row=int(r),
                col=int(c),
                direction=direction,
                width=width,
                component_id=comp_id,
            )
        )

    return endpoints


def _direction_from_neighbor(
    skeleton: np.ndarray, r: int, c: int
) -> tuple[float, float]:
    """Fallback: direction points away from the single neighbor."""
    h, w = skeleton.shape
    for dr, dc, _ in [
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
        (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
    ]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and skeleton[nr, nc]:
            return (float(-dr), float(-dc))
    return (1.0, 0.0)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def generate_candidate_pairs(
    endpoints: list[Endpoint],
    max_radius: float = 50.0,
    min_radius: float = 2.0,
    exclude_same_component: bool = True,
) -> list[tuple[Endpoint, Endpoint, float]]:
    """Generate candidate endpoint pairs for potential bridging.

    Returns list of (ep_a, ep_b, euclidean_distance).
    """
    candidates: list[tuple[Endpoint, Endpoint, float]] = []
    for a, b in combinations(endpoints, 2):
        if exclude_same_component and a.component_id == b.component_id:
            continue
        dist = math.hypot(a.row - b.row, a.col - b.col)
        if min_radius <= dist <= max_radius:
            candidates.append((a, b, dist))
    # Sort by distance (closer pairs first)
    candidates.sort(key=lambda x: x[2])
    return candidates


# ---------------------------------------------------------------------------
# Connection cost
# ---------------------------------------------------------------------------


def compute_connection_cost(
    ep_a: Endpoint,
    ep_b: Endpoint,
    gap_length: float,
    confidence_map: np.ndarray,
    component_labels: np.ndarray,
    w_p: float = 1.0,
    w_l: float = 0.5,
    w_theta: float = 1.0,
    w_w: float = 0.5,
    w_g: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Compute the connection cost C(e) for bridging two endpoints.

    C(e) = w_p * mean(-log(q)) + w_l * L_e/r + w_theta * (1-cos(θ)) + w_w * ΔW - w_g * G_e

    Returns (total_cost, component_costs).
    """
    # --- Term 1: Confidence along the gap ---
    # Sample points along the straight line between endpoints
    num_samples = max(2, int(gap_length))
    row_samples = np.linspace(ep_a.row, ep_b.row, num_samples).astype(np.int32)
    col_samples = np.linspace(ep_a.col, ep_b.col, num_samples).astype(np.int32)

    # Clip to image bounds
    h, w = confidence_map.shape
    valid = (
        (row_samples >= 0) & (row_samples < h) & (col_samples >= 0) & (col_samples < w)
    )
    if not np.any(valid):
        return float("inf"), {}

    conf_values = confidence_map[row_samples[valid], col_samples[valid]]
    # -log(q), clamped to avoid blowup
    eps = 1e-9
    log_conf = -np.log(np.clip(conf_values, eps, 1.0 - eps))
    mean_log_inv = float(np.mean(log_conf))

    # --- Term 2: Normalized gap length ---
    # L_e / max_radius (using same max_radius as candidate generation)
    # We'll use a fixed reference; the caller can rescale
    length_term = gap_length / 50.0  # normalized by max_radius

    # --- Term 3: Direction inconsistency ---
    # Direction from a to b
    gap_dr = ep_b.row - ep_a.row
    gap_dc = ep_b.col - ep_a.col
    gap_norm = math.hypot(gap_dr, gap_dc)
    if gap_norm > 0:
        gap_dir = (gap_dr / gap_norm, gap_dc / gap_norm)
    else:
        gap_dir = (1.0, 0.0)

    dot_a = ep_a.direction[0] * gap_dir[0] + ep_a.direction[1] * gap_dir[1]
    dot_b = ep_b.direction[0] * (-gap_dir[0]) + ep_b.direction[1] * (-gap_dir[1])

    # θ = acos(dot), but we want (1 - cos(θ)) for each endpoint
    # For direction consistency, both should point toward each other
    dir_term_a = 0.5 * (1.0 - max(-1.0, min(1.0, dot_a)))
    dir_term_b = 0.5 * (1.0 - max(-1.0, min(1.0, dot_b)))
    theta_term = dir_term_a + dir_term_b

    # --- Term 4: Width difference ---
    width_diff = abs(ep_a.width - ep_b.width) / max(ep_a.width, ep_b.width, 1.0)

    # --- Term 5: Connectivity gain ---
    # Reward if endpoints belong to different connected components. Read the
    # CURRENT labels from the array (updated in place by prior accepted
    # bridges), NOT the stale Endpoint fields set once at detection time:
    # otherwise every candidate scores gain=1 forever and already-merged
    # fragments keep getting redundant parallel bridges.
    cid_a = int(component_labels[ep_a.row, ep_a.col])
    cid_b = int(component_labels[ep_b.row, ep_b.col])
    gain = 1.0 if cid_a != cid_b else 0.0

    components = {
        "confidence": float(w_p * mean_log_inv),
        "length": float(w_l * length_term),
        "direction": float(w_theta * theta_term),
        "width": float(w_w * width_diff),
        "connectivity_gain": float(-w_g * gain),
    }

    total = (
        components["confidence"]
        + components["length"]
        + components["direction"]
        + components["width"]
        + components["connectivity_gain"]
    )

    return total, components


# ---------------------------------------------------------------------------
# Bridge construction
# ---------------------------------------------------------------------------


def find_bridge_curve(
    ep_a: Endpoint,
    ep_b: Endpoint,
    confidence_map: np.ndarray,
    num_points: int = 50,
    mode: str = "straight",
    geodesic_margin: int = 10,
    geodesic_risk_weight: float = 1.0,
) -> list[tuple[int, int]]:
    """Construct a bridge curve between two accepted endpoints.

    ``straight`` preserves the frozen paper implementation. ``geodesic`` runs
    a deterministic 8-connected A* search inside a padded endpoint bounding
    box. Its step cost is

        step_length * (1 + risk_weight * -log(q)),

    so it can bend toward locally supported road pixels without changing
    endpoint detection, candidate generation, or the existing acceptance
    score. Keeping acceptance fixed isolates bridge geometry in the Gate
    experiment.
    """
    if mode == "geodesic":
        return find_geodesic_bridge_curve(
            ep_a,
            ep_b,
            confidence_map,
            margin=geodesic_margin,
            risk_weight=geodesic_risk_weight,
        )
    if mode != "straight":
        raise ValueError(f"Unknown bridge curve mode: {mode!r}")

    points: list[tuple[int, int]] = []
    for i in range(num_points + 1):
        t = i / num_points
        r = int(round(ep_a.row + t * (ep_b.row - ep_a.row)))
        c = int(round(ep_a.col + t * (ep_b.col - ep_a.col)))
        # Clamp to image bounds
        r = max(0, min(r, confidence_map.shape[0] - 1))
        c = max(0, min(c, confidence_map.shape[1] - 1))
        if not points or points[-1] != (r, c):
            points.append((r, c))
    return points


def find_geodesic_bridge_curve(
    ep_a: Endpoint,
    ep_b: Endpoint,
    confidence_map: np.ndarray,
    margin: int = 10,
    risk_weight: float = 1.0,
) -> list[tuple[int, int]]:
    """Find a local confidence-supported bridge with deterministic A*.

    The search is deliberately local and training-free. The Euclidean
    heuristic is admissible because every move costs at least its geometric
    step length. If reconstruction ever fails, the function falls back to the
    frozen straight curve rather than returning a disconnected bridge.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if risk_weight < 0:
        raise ValueError("risk_weight must be non-negative")

    h, w = confidence_map.shape
    r0 = max(0, min(ep_a.row, ep_b.row) - margin)
    r1 = min(h, max(ep_a.row, ep_b.row) + margin + 1)
    c0 = max(0, min(ep_a.col, ep_b.col) - margin)
    c1 = min(w, max(ep_a.col, ep_b.col) + margin + 1)

    start = (ep_a.row - r0, ep_a.col - c0)
    goal = (ep_b.row - r0, ep_b.col - c0)
    local_q = np.asarray(confidence_map[r0:r1, c0:c1], dtype=np.float64)
    local_h, local_w = local_q.shape

    g_score = np.full((local_h, local_w), np.inf, dtype=np.float64)
    parent_r = np.full((local_h, local_w), -1, dtype=np.int32)
    parent_c = np.full((local_h, local_w), -1, dtype=np.int32)
    closed = np.zeros((local_h, local_w), dtype=bool)

    def heuristic(row: int, col: int) -> float:
        return math.hypot(goal[0] - row, goal[1] - col)

    neighbors = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )
    g_score[start] = 0.0
    queue: list[tuple[float, float, int, int]] = [
        (heuristic(*start), 0.0, start[0], start[1])
    ]
    eps = 1e-6

    while queue:
        _, current_g, row, col = heapq.heappop(queue)
        if closed[row, col] or current_g > g_score[row, col] + 1e-12:
            continue
        if (row, col) == goal:
            break
        closed[row, col] = True
        for dr, dc, step_length in neighbors:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < local_h and 0 <= nc < local_w):
                continue
            mean_q = 0.5 * (local_q[row, col] + local_q[nr, nc])
            risk = -math.log(float(np.clip(mean_q, eps, 1.0)))
            tentative = current_g + step_length * (1.0 + risk_weight * risk)
            if tentative + 1e-12 >= g_score[nr, nc]:
                continue
            g_score[nr, nc] = tentative
            parent_r[nr, nc] = row
            parent_c[nr, nc] = col
            heapq.heappush(
                queue,
                (tentative + heuristic(nr, nc), tentative, nr, nc),
            )

    if not np.isfinite(g_score[goal]):
        return find_bridge_curve(ep_a, ep_b, confidence_map, mode="straight")

    local_path = [goal]
    cursor = goal
    while cursor != start:
        pr = int(parent_r[cursor])
        pc = int(parent_c[cursor])
        if pr < 0 or pc < 0:
            return find_bridge_curve(ep_a, ep_b, confidence_map, mode="straight")
        cursor = (pr, pc)
        local_path.append(cursor)
    local_path.reverse()
    return [(row + r0, col + c0) for row, col in local_path]


def conservative_bridge_selection(
    candidates: list[tuple[Endpoint, Endpoint, float]],
    confidence_map: np.ndarray,
    component_labels: np.ndarray,
    tau_gate: float = 2.0,
    min_q: float | None = None,
    w_p: float = 1.0,
    w_l: float = 0.5,
    w_theta: float = 1.0,
    w_w: float = 0.5,
    w_g: float = 1.0,
    curve_mode: str = "straight",
    geodesic_margin: int = 10,
    geodesic_risk_weight: float = 1.0,
) -> list[Bridge]:
    """Select bridges conservatively using soft and optional hard gates.

    A candidate must satisfy ``C(e) < tau_gate``.  When ``min_q`` is not
    ``None``, every sampled pixel on its bridge curve must additionally have
    uncertainty-discounted score ``q >= min_q``.  This makes the paper's optional
    "high-confidence bridge" variant an actual hard constraint instead of a
    description of the soft confidence term in ``C(e)``.

    Bridges are evaluated in order of increasing gap length.
    After each accepted bridge, component labels are updated so
    subsequent bridges can account for the new connectivity.
    """
    accepted: list[Bridge] = []

    for ep_a, ep_b, gap_length in candidates:
        live_id_a = int(component_labels[ep_a.row, ep_a.col])
        live_id_b = int(component_labels[ep_b.row, ep_b.col])
        if live_id_a == 0 or live_id_b == 0 or live_id_a == live_id_b:
            continue

        total_cost, cost_components = compute_connection_cost(
            ep_a,
            ep_b,
            gap_length,
            confidence_map,
            component_labels,
            w_p=w_p,
            w_l=w_l,
            w_theta=w_theta,
            w_w=w_w,
            w_g=w_g,
        )

        if total_cost < tau_gate:
            curve = find_bridge_curve(
                ep_a,
                ep_b,
                confidence_map,
                mode=curve_mode,
                geodesic_margin=geodesic_margin,
                geodesic_risk_weight=geodesic_risk_weight,
            )
            curve_min_q = float(min(confidence_map[r, c] for r, c in curve))
            if min_q is not None and curve_min_q < min_q:
                continue
            cost_components = dict(cost_components)
            cost_components["curve_min_q"] = curve_min_q
            cost_components["curve_length"] = float(len(curve))
            cost_components["curve_tortuosity"] = float(
                max(len(curve) - 1, 0) / max(gap_length, 1e-6)
            )
            bridge = Bridge(
                start=ep_a,
                end=ep_b,
                curve=curve,
                cost=total_cost,
                cost_components=cost_components,
            )
            accepted.append(bridge)

            # Merge component labels (union-find light)
            old_id = max(live_id_a, live_id_b)
            new_id = min(live_id_a, live_id_b)
            component_labels[component_labels == old_id] = new_id

    return accepted


def apply_bridges_to_mask(
    mask: np.ndarray,
    bridges: list[Bridge],
    bridge_width: int = 2,
) -> np.ndarray:
    """Apply accepted bridges to a binary mask, thickening to bridge_width."""
    repaired = mask.copy()
    for bridge in bridges:
        for r, c in bridge.curve:
            r0 = max(0, r - bridge_width)
            r1 = min(repaired.shape[0], r + bridge_width + 1)
            c0 = max(0, c - bridge_width)
            c1 = min(repaired.shape[1], c + bridge_width + 1)
            repaired[r0:r1, c0:c1] = True
    return repaired


# ---------------------------------------------------------------------------
# Full repair pipeline
# ---------------------------------------------------------------------------


def repair_topology(
    mask: np.ndarray,
    confidence_map: np.ndarray,
    max_radius: float = 50.0,
    min_radius: float = 2.0,
    tau_gate: float = 2.0,
    min_q: float | None = None,
    w_p: float = 1.0,
    w_l: float = 0.5,
    w_theta: float = 1.0,
    w_w: float = 0.5,
    w_g: float = 1.0,
    bridge_width: int = 2,
    curve_mode: str = "straight",
    geodesic_margin: int = 10,
    geodesic_risk_weight: float = 1.0,
) -> tuple[np.ndarray, list[Bridge], np.ndarray]:
    """Run the full uncertainty-discounted topology repair pipeline.

    Args:
        mask: binary road mask (H, W)
        confidence_map: calibrated confidence lower bound q(x) (H, W)
        ...

    Returns:
        repaired_mask: (H, W) mask with bridges applied
        bridges: list of accepted Bridge objects
        skeleton: (H, W) skeleton of the original mask
    """
    skeleton = compute_skeleton(mask)
    component_labels, num_components = ndimage.label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    endpoints = detect_endpoints(skeleton, mask, component_labels)

    if len(endpoints) < 2:
        return mask.copy(), [], skeleton

    candidates = generate_candidate_pairs(
        endpoints, max_radius, min_radius, exclude_same_component=True
    )

    bridges = conservative_bridge_selection(
        candidates,
        confidence_map,
        component_labels,
        tau_gate=tau_gate,
        min_q=min_q,
        w_p=w_p,
        w_l=w_l,
        w_theta=w_theta,
        w_w=w_w,
        w_g=w_g,
        curve_mode=curve_mode,
        geodesic_margin=geodesic_margin,
        geodesic_risk_weight=geodesic_risk_weight,
    )

    repaired = apply_bridges_to_mask(mask, bridges, bridge_width)
    return repaired, bridges, skeleton


def repair_topology_geometric(
    mask: np.ndarray,
    max_radius: float = 50.0,
    min_radius: float = 2.0,
    tau_gate: float = 1.0,
    w_l: float = 0.5,
    w_theta: float = 1.0,
    w_w: float = 0.5,
    bridge_width: int = 2,
) -> tuple[np.ndarray, list[Bridge], np.ndarray]:
    """Run a confidence-free geometric endpoint-reconnection baseline.

    The baseline shares skeletonization, endpoint detection, candidate
    generation, bridge rasterization, and sequential component updates with
    :func:`repair_topology`. Its selection cost uses only gap length,
    endpoint-direction consistency, and endpoint-width mismatch. It does not
    consume ensemble confidence and receives no connectivity reward, isolating
    the empirical value of confidence weighting from geometric repair.

    ``tau_gate`` must be selected on a calibration split before any test-set
    comparison.
    """
    skeleton = compute_skeleton(mask)
    component_labels, _ = ndimage.label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    endpoints = detect_endpoints(skeleton, mask, component_labels)
    if len(endpoints) < 2:
        return mask.copy(), [], skeleton

    candidates = generate_candidate_pairs(
        endpoints, max_radius, min_radius, exclude_same_component=True
    )
    # This array is a shape carrier only because w_p=0. Ones make the stored
    # curve_min_q neutral and avoid accidental zero-confidence penalties.
    neutral_confidence = np.ones(mask.shape, dtype=np.float32)
    bridges = conservative_bridge_selection(
        candidates,
        neutral_confidence,
        component_labels,
        tau_gate=tau_gate,
        w_p=0.0,
        w_l=w_l,
        w_theta=w_theta,
        w_w=w_w,
        w_g=0.0,
    )
    repaired = apply_bridges_to_mask(mask, bridges, bridge_width)
    return repaired, bridges, skeleton
