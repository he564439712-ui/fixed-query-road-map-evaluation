"""Path planning evaluation metrics.

Implements:
  - Success rate
  - Path length (and relative to oracle)
  - GT road path ratio / off-road ratio
  - Risk metrics (average, max)
  - Minimum road boundary distance
  - Planning time and expanded nodes
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.planning.grid import Point, SearchResult, path_length


def path_on_road_ratio(
    path: list[Point], road_mask: np.ndarray
) -> float:
    """Fraction of path points that lie on the GT road."""
    if not path:
        return 0.0
    on_road = sum(1 for r, c in path if road_mask[r, c])
    return on_road / len(path)


def off_road_ratio(
    path: list[Point], road_mask: np.ndarray
) -> float:
    """Fraction of path points that are off the GT road."""
    return 1.0 - path_on_road_ratio(path, road_mask)


def min_boundary_distance(
    path: list[Point], distance: np.ndarray
) -> float:
    """Minimum distance from any path point to the nearest road boundary."""
    if not path:
        return 0.0
    rows = np.array([p[0] for p in path], dtype=np.int32)
    cols = np.array([p[1] for p in path], dtype=np.int32)
    return float(np.min(distance[rows, cols]))


def path_risk_stats(
    path: list[Point], risk: np.ndarray
) -> tuple[float, float]:
    """Average and maximum risk along path."""
    if not path:
        return float("nan"), float("nan")
    rows = np.array([p[0] for p in path], dtype=np.int32)
    cols = np.array([p[1] for p in path], dtype=np.int32)
    path_risk = risk[rows, cols]
    return float(np.mean(path_risk)), float(np.max(path_risk))


def relative_detour(
    planned_length: float, oracle_length: float
) -> float:
    """Relative detour ratio: (planned - oracle) / oracle."""
    if oracle_length <= 0:
        return float("nan")
    return (planned_length - oracle_length) / oracle_length


def planning_metrics(
    result: SearchResult,
    ground_truth: np.ndarray,
    distance: np.ndarray | None = None,
    risk: np.ndarray | None = None,
    oracle_length: float | None = None,
) -> dict[str, Any]:
    """Compute all planning metrics for one query.

    Args:
        result: A* search result
        ground_truth: (H, W) binary GT road mask
        distance: (H, W) distance transform (optional)
        risk: (H, W) risk map (optional)
        oracle_length: GT oracle path length for detour calculation

    Returns dict of metric_name → value.
    """
    if not result.success:
        return {
            "method": result.name,
            "success": 0,
            "path_length": float("nan"),
            "expanded_nodes": result.expanded_nodes,
            "planning_time_ms": result.planning_time_ms,
            "on_road_ratio": float("nan"),
            "off_road_ratio": float("nan"),
            "average_risk": float("nan"),
            "max_risk": float("nan"),
            "min_boundary_distance": float("nan"),
            "relative_detour": float("nan"),
        }

    length = path_length(result.path)
    on_road = path_on_road_ratio(result.path, ground_truth)

    avg_risk = float("nan")
    max_risk = float("nan")
    if risk is not None:
        avg_risk, max_risk = path_risk_stats(result.path, risk)

    min_dist = float("nan")
    if distance is not None:
        min_dist = min_boundary_distance(result.path, distance)

    detour = float("nan")
    if oracle_length is not None:
        detour = relative_detour(length, oracle_length)

    return {
        "method": result.name,
        "success": 1,
        "path_length": length,
        "expanded_nodes": result.expanded_nodes,
        "planning_time_ms": result.planning_time_ms,
        "on_road_ratio": on_road,
        "off_road_ratio": 1.0 - on_road,
        "average_risk": avg_risk,
        "max_risk": max_risk,
        "min_boundary_distance": min_dist,
        "relative_detour": detour,
    }
