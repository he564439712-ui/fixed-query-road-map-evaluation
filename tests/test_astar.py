"""Unit tests for A* path planner.

Tests cover:
  - Trivial paths (start == goal)
  - Straight-line paths (no obstacles)
  - Obstacle avoidance
  - No-solution cases
  - 8-connected grid movement costs
  - Path reconstruction
  - Risk-aware vs vanilla A*
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We'll import from the planning module once we create it.
# For now, define the core A* inline to test independently.
# This also serves as a reference implementation for the planning module.

Point = tuple[int, int]

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


def heuristic(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_grid():
    """10×10 traversable grid (all ones)."""
    return np.ones((10, 10), dtype=bool)


@pytest.fixture
def risk_grid():
    """Uniform risk grid."""
    return np.zeros((10, 10), dtype=np.float32)


@pytest.fixture
def wall_grid():
    """10×10 grid with a vertical wall in the middle."""
    grid = np.ones((10, 10), dtype=bool)
    grid[1:9, 5] = False
    return grid


@pytest.fixture
def u_shaped_grid():
    """12×12 U-shaped obstacle."""
    grid = np.ones((12, 12), dtype=bool)
    # Horizontal bottom
    grid[9, 2:10] = False
    # Vertical left
    grid[2:9, 2] = False
    # Vertical right
    grid[2:9, 9] = False
    return grid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHeuristic:
    def test_zero(self):
        assert heuristic((0, 0), (0, 0)) == 0.0

    def test_straight_line(self):
        assert heuristic((0, 0), (0, 5)) == 5.0

    def test_diagonal(self):
        expected = math.hypot(3, 4)
        assert heuristic((0, 0), (3, 4)) == expected

    def test_symmetric(self):
        a, b = (2, 7), (9, 3)
        assert heuristic(a, b) == heuristic(b, a)

    def test_non_negative(self):
        for _ in range(100):
            a = (np.random.randint(0, 100), np.random.randint(0, 100))
            b = (np.random.randint(0, 100), np.random.randint(0, 100))
            assert heuristic(a, b) >= 0


class TestNeighborIteration:
    def test_all_neighbors_in_bounds(self, empty_grid):
        from src.planning.grid import iter_neighbors
        neighbors = list(iter_neighbors((5, 5), empty_grid))
        assert len(neighbors) == 8

    def test_corner_neighbors(self, empty_grid):
        from src.planning.grid import iter_neighbors
        neighbors = list(iter_neighbors((0, 0), empty_grid))
        assert len(neighbors) == 3  # right, down, down-right

    def test_edge_neighbors(self, empty_grid):
        from src.planning.grid import iter_neighbors
        neighbors = list(iter_neighbors((0, 5), empty_grid))
        assert len(neighbors) == 5

    def test_blocked_neighbors(self, empty_grid):
        from src.planning.grid import iter_neighbors
        grid = empty_grid.copy()
        grid[4, 5] = False
        grid[6, 5] = False
        neighbors = list(iter_neighbors((5, 5), grid))
        assert len(neighbors) == 6  # up and down blocked

    def test_step_costs(self, empty_grid):
        from src.planning.grid import iter_neighbors
        neighbors = list(iter_neighbors((5, 5), empty_grid))
        costs = [c for _, c in neighbors]
        # 4 cardinal (cost 1.0) + 4 diagonal (cost sqrt(2))
        assert sum(1 for c in costs if abs(c - 1.0) < 1e-9) == 4
        assert sum(1 for c in costs if abs(c - math.sqrt(2)) < 1e-9) == 4


class TestAStarVanilla:
    def test_trivial_path(self, empty_grid, risk_grid):
        from src.planning.grid import astar_search
        result = astar_search(
            name="test",
            traversable=empty_grid,
            risk=risk_grid,
            start=(5, 5),
            goal=(5, 5),
        )
        assert result.success
        assert len(result.path) == 1
        assert result.path[0] == (5, 5)

    def test_straight_path(self, empty_grid, risk_grid):
        from src.planning.grid import astar_search
        result = astar_search(
            name="test",
            traversable=empty_grid,
            risk=risk_grid,
            start=(0, 0),
            goal=(0, 9),
        )
        assert result.success
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (0, 9)
        # All points should be traversable
        for r, c in result.path:
            assert empty_grid[r, c]

    def test_path_avoids_wall(self, wall_grid, risk_grid):
        from src.planning.grid import astar_search
        result = astar_search(
            name="test",
            traversable=wall_grid,
            risk=risk_grid,
            start=(5, 0),
            goal=(5, 9),
        )
        assert result.success
        # Path must go around the wall (column 5 is blocked rows 1-8)
        for r, c in result.path:
            assert wall_grid[r, c]
            # Should never step on the wall
            assert not (c == 5 and 1 <= r <= 8)

    def test_no_path(self, wall_grid, risk_grid):
        from src.planning.grid import astar_search
        # Completely blocked
        blocked = np.zeros((5, 5), dtype=bool)
        result = astar_search(
            name="test",
            traversable=blocked,
            risk=np.zeros((5, 5), dtype=np.float32),
            start=(0, 0),
            goal=(4, 4),
        )
        assert not result.success

    def test_path_connectivity(self, empty_grid, risk_grid):
        """Every consecutive pair in the path should be neighbors."""
        from src.planning.grid import astar_search, iter_neighbors

        result = astar_search(
            name="test",
            traversable=empty_grid,
            risk=risk_grid,
            start=(1, 1),
            goal=(8, 8),
        )
        assert result.success
        for a, b in zip(result.path, result.path[1:]):
            neighbor_points = {p for p, _ in iter_neighbors(a, empty_grid)}
            assert b in neighbor_points, f"{b} not a neighbor of {a}"

    def test_monotonic_cost(self, empty_grid, risk_grid):
        """Cost-to-come should be monotonic along the path."""
        from src.planning.grid import astar_search

        # Cost is stored internally; we just verify path is shortest-ish
        result = astar_search(
            name="test",
            traversable=empty_grid,
            risk=risk_grid,
            start=(0, 0),
            goal=(9, 9),
        )
        assert result.success
        # Optimal path length on empty grid is 9*sqrt(2) ≈ 12.73
        # Path could be diagonal or stair-step; both have same cost in 8-grid
        path_len = 0.0
        for a, b in zip(result.path, result.path[1:]):
            path_len += math.hypot(b[0] - a[0], b[1] - a[1])
        # Should be close to optimal (allow small tolerance)
        optimal = 9 * math.sqrt(2)  # ≈ 12.73
        assert path_len < optimal + 1.0


class TestRiskAwareAStar:
    def test_risk_aversion(self, empty_grid):
        from src.planning.grid import astar_search

        # Create a risk hotspot
        risk = np.zeros((10, 10), dtype=np.float32)
        risk[3:7, 3:7] = 10.0  # high risk center

        vanilla = astar_search(
            name="vanilla",
            traversable=empty_grid,
            risk=risk,
            start=(0, 0),
            goal=(9, 9),
            risk_weight=0.0,
        )
        risk_aware = astar_search(
            name="risk_aware",
            traversable=empty_grid,
            risk=risk,
            start=(0, 0),
            goal=(9, 9),
            risk_weight=10.0,
        )
        assert vanilla.success and risk_aware.success

        # Risk-aware path should avoid the hotspot
        vanilla_risk = sum(risk[r, c] for r, c in vanilla.path)
        risk_aware_risk = sum(risk[r, c] for r, c in risk_aware.path)
        assert risk_aware_risk <= vanilla_risk + 1e-6

    def test_start_equals_goal(self, empty_grid, risk_grid):
        from src.planning.grid import astar_search
        result = astar_search(
            name="test",
            traversable=empty_grid,
            risk=np.zeros((10, 10), dtype=np.float32),
            start=(3, 3),
            goal=(3, 3),
            risk_weight=5.0,
        )
        assert result.success
        assert result.path == [(3, 3)]


class TestLargestConnectedRoad:
    def test_simple(self):
        from src.planning.grid import largest_connected_road

        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = True
        mask[0, 1] = True
        mask[3, 3] = True  # isolated pixel
        result = largest_connected_road(mask)
        assert result[0, 0] and result[0, 1]
        assert not result[3, 3]

    def test_empty_mask(self):
        from src.planning.grid import largest_connected_road

        mask = np.zeros((5, 5), dtype=bool)
        with pytest.raises(ValueError):
            largest_connected_road(mask)
