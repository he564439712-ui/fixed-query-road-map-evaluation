"""A* path planning on 2D grid maps.

Ported and extended from experiments/risk_aware_astar_single.py.

Supports:
  - 8-connected grid with Euclidean and diagonal step costs
  - Vanilla A* (risk_weight=0)
  - Risk-aware A* with configurable risk weight
  - Multiple cost function variants for baselines
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
from scipy import ndimage

Point = tuple[int, int]

# 8-connected neighborhood: (dr, dc, step_cost)
NEIGHBORS_8: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class SearchResult:
    name: str
    path: list[Point]
    planning_time_ms: float
    expanded_nodes: int
    success: bool


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------


def largest_connected_road(road_mask: np.ndarray) -> np.ndarray:
    """Extract the largest 8-connected component from a binary road mask."""
    labels, count = ndimage.label(
        road_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if count == 0:
        raise ValueError("road mask contains no traversable pixels")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    return labels == int(np.argmax(sizes))


def build_risk_map(
    road_mask: np.ndarray, epsilon: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Build a distance-based risk map.

    Returns:
        risk: (H, W) — inf for non-road, [0, 10] for road pixels
        distance: (H, W) — Euclidean distance to nearest non-road pixel
    """
    distance = ndimage.distance_transform_edt(road_mask).astype(np.float32)
    risk = np.full(road_mask.shape, np.inf, dtype=np.float32)
    if not np.any(road_mask):
        # No road pixels: every pixel is non-traversable (risk=inf).
        return risk, distance
    inv_distance = 1.0 / (distance[road_mask] + epsilon)
    low = float(inv_distance.min())
    high = float(inv_distance.max())
    if high > low:
        risk[road_mask] = 10.0 * (inv_distance - low) / (high - low)
    else:
        risk[road_mask] = 0.0
    return risk, distance


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------


def binary_cost(point: Point, mask: np.ndarray) -> float:
    """Unit cost for traversable pixels, inf for obstacles."""
    r, c = point
    return 0.0 if mask[r, c] else float("inf")


def probability_cost(
    point: Point, probability: np.ndarray, eps: float = 1e-9
) -> float:
    """Cost = -log(p) for soft probability maps."""
    r, c = point
    p = np.clip(float(probability[r, c]), eps, 1.0 - eps)
    return float(-math.log(p))


def clearance_cost(
    point: Point,
    probability: np.ndarray,
    distance: np.ndarray,
    lambda_b: float = 1.0,
    tau_d: float = 5.0,
    eps: float = 1e-9,
) -> float:
    """Cost combining probability and boundary distance.

    J = -log(p) + lambda_b * exp(-d / tau_d)
    """
    r, c = point
    p = np.clip(float(probability[r, c]), eps, 1.0 - eps)
    d = float(distance[r, c])
    return float(-math.log(p) + lambda_b * math.exp(-d / tau_d))


def unified_risk_cost(
    point: Point,
    probability: np.ndarray,
    distance: np.ndarray,
    lambda_r: float = 2.0,
    lambda_b: float = 1.0,
    tau_d: float = 5.0,
    eps: float = 1e-9,
) -> float:
    """Unified risk cost from the project plan.

    J = Δs + λ_r * (-log(q(x))) + λ_b * exp(-d(x)/τ_d)
    where q(x) is an uncertainty-discounted heuristic score.
    """
    r, c = point
    p = np.clip(float(probability[r, c]), eps, 1.0 - eps)
    d = float(distance[r, c])
    return float(1.0 + lambda_r * (-math.log(p)) + lambda_b * math.exp(-d / tau_d))


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------


def heuristic(a: Point, b: Point) -> float:
    """Euclidean heuristic for 8-connected grid (admissible)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def iter_neighbors(
    point: Point, traversable: np.ndarray
) -> Iterator[tuple[Point, float]]:
    """Yield (neighbor, step_cost) for traversable 8-connected neighbors."""
    row, col = point
    rows, cols = traversable.shape
    for dr, dc, step_cost in NEIGHBORS_8:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols and traversable[nr, nc]:
            yield (nr, nc), step_cost


def reconstruct_path(
    came_from: dict[Point, Point], current: Point
) -> list[Point]:
    """Trace back from goal to start through came_from pointers."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_search(
    name: str,
    traversable: np.ndarray,
    risk: np.ndarray,
    start: Point,
    goal: Point,
    heuristic_weight: float = 1.0,
    risk_weight: float = 0.0,
    point_cost: Callable[[Point], float] | None = None,
    max_expansions: int | None = None,
) -> SearchResult:
    """Generic A* search on a grid.

    Cost: f(n) = g(n) + heuristic_weight * h(n)
    where g(n) accumulates step_cost + risk_weight * normalized_risk.

    Args:
        name: identifier for this search
        traversable: binary (H, W) traversability mask
        risk: (H, W) risk values (inf for non-traversable)
        start, goal: (row, col) points
        heuristic_weight: multiplier for heuristic (1.0 = standard A*)
        risk_weight: multiplier for risk term (0.0 = ignore risk)
        point_cost: optional per-point cost function (overrides risk_weight)

    Returns:
        SearchResult with path, timing, and node count.
    """
    started_at = time.perf_counter()
    frontier: list[tuple[float, int, Point]] = []
    heapq.heappush(frontier, (0.0, 0, start))
    came_from: dict[Point, Point] = {}
    cost_so_far: dict[Point, float] = {start: 0.0}
    expanded_nodes = 0
    counter = 1

    while frontier:
        _, _, current = heapq.heappop(frontier)
        expanded_nodes += 1
        if max_expansions is not None and expanded_nodes > max_expansions:
            # Safety valve: avoid pathological full-grid searches on large maps.
            break

        if current == goal:
            elapsed = (time.perf_counter() - started_at) * 1000.0
            return SearchResult(
                name,
                reconstruct_path(came_from, current),
                elapsed,
                expanded_nodes,
                True,
            )

        for neighbor, step_cost in iter_neighbors(current, traversable):
            if point_cost is not None:
                extra = point_cost(neighbor)
            else:
                extra = risk_weight * float(risk[neighbor]) / 10.0

            if not np.isfinite(extra):
                extra = 1e6  # large penalty but not inf (allows fallback)

            new_cost = cost_so_far[current] + step_cost + extra
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic_weight * heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, counter, neighbor))
                counter += 1
                came_from[neighbor] = current

    elapsed = (time.perf_counter() - started_at) * 1000.0
    return SearchResult(name, [], elapsed, expanded_nodes, False)


# ---------------------------------------------------------------------------
# Path analysis
# ---------------------------------------------------------------------------


def path_length(path: list[Point]) -> float:
    """Compute total Euclidean path length."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        total += math.hypot(r2 - r1, c2 - c1)
    return total


def turn_count(path: list[Point]) -> int:
    """Count direction changes along a path."""
    if len(path) < 3:
        return 0
    turns = 0
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for a, b in zip(path[1:], path[2:]):
        direction = (b[0] - a[0], b[1] - a[1])
        if direction != prev_dir:
            turns += 1
        prev_dir = direction
    return turns


def choose_start_goal(
    road_component: np.ndarray,
    distance: np.ndarray,
    seed: int,
) -> tuple[Point, Point]:
    """Select start and goal points deep inside the road region.

    Prefers points near the road centerline (high distance to boundary)
    and maximizes pairwise Euclidean distance.
    """
    rng = np.random.default_rng(seed)
    max_distance = float(np.max(distance[road_component]))
    center_threshold = max(1.5, 0.65 * max_distance)
    candidates = np.argwhere(road_component & (distance >= center_threshold))
    if len(candidates) < 2:
        center_threshold = max(1.0, float(np.percentile(distance[road_component], 80)))
        candidates = np.argwhere(road_component & (distance >= center_threshold))
    if len(candidates) < 2:
        candidates = np.argwhere(road_component)
    if len(candidates) < 2:
        raise ValueError("not enough traversable pixels to select start and goal")

    sample_size = min(5000, len(candidates))
    sample = candidates[rng.choice(len(candidates), size=sample_size, replace=False)]
    first = sample[int(rng.integers(sample_size))]
    second = sample[int(np.argmax(np.sum((sample - first) ** 2, axis=1)))]
    third = sample[int(np.argmax(np.sum((sample - second) ** 2, axis=1)))]
    return (int(second[0]), int(second[1])), (int(third[0]), int(third[1]))


def validate_point(
    name: str, point: Point, road_component: np.ndarray
) -> None:
    """Raise if point is outside the traversable region."""
    rows, cols = road_component.shape
    row, col = point
    if row < 0 or row >= rows or col < 0 or col >= cols:
        raise ValueError(f"{name} {point} is outside the map")
    if not road_component[row, col]:
        raise ValueError(
            f"{name} {point} is not inside the traversable road region"
        )
