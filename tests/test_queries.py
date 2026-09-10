"""Unit tests for query generation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.queries import (
    _classify_distance,
    _euclidean,
    _snap_to_road,
    generate_queries_for_image,
    load_queries,
)


class TestEuclidean:
    def test_same_point(self):
        assert _euclidean((0, 0), (0, 0)) == 0.0

    def test_horizontal(self):
        assert _euclidean((5, 0), (5, 10)) == 10.0

    def test_diagonal(self):
        expected = np.hypot(3, 4)
        assert _euclidean((0, 0), (3, 4)) == expected


class TestSnapToRoad:
    @pytest.fixture
    def simple_mask(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 5:15] = True  # horizontal road at row 10
        return mask

    def test_snap_exact(self, simple_mask):
        result = _snap_to_road((10, 7), simple_mask, snap_radius=5)
        assert result == (10, 7)

    def test_snap_nearby(self, simple_mask):
        result = _snap_to_road((10, 4), simple_mask, snap_radius=5)
        assert result is not None
        r, c = result
        assert simple_mask[r, c]

    def test_snap_too_far(self, simple_mask):
        result = _snap_to_road((0, 0), simple_mask, snap_radius=2)
        assert result is None

    def test_snap_out_of_bounds(self, simple_mask):
        result = _snap_to_road((-5, -5), simple_mask, snap_radius=3)
        assert result is None


class TestClassifyDistance:
    def test_short(self):
        # short_cut = 0.3 * 100 = 30, so < 30 is short
        assert _classify_distance(25, 100, 0.3, 0.37) == "short"

    def test_medium(self):
        assert _classify_distance(50, 100, 0.3, 0.37) == "medium"

    def test_long(self):
        assert _classify_distance(80, 100, 0.3, 0.37) == "long"


class TestGenerateQueriesForImage:
    @pytest.fixture
    def road_mask(self):
        """Simple road mask: 100×100 with a cross shape."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, :] = True  # horizontal
        mask[:, 45:55] = True  # vertical
        return mask

    def test_generates_correct_count(self, road_mask):
        queries = generate_queries_for_image(
            road_mask, num_queries=20, seed=42
        )
        # May get fewer if sampling is difficult, but should be close
        assert len(queries) >= 10

    def test_all_points_traversable(self, road_mask):
        queries = generate_queries_for_image(
            road_mask, num_queries=20, seed=42
        )
        for q in queries:
            assert road_mask[q["start_row"], q["start_col"]]
            assert road_mask[q["goal_row"], q["goal_col"]]

    def test_start_not_equal_goal(self, road_mask):
        queries = generate_queries_for_image(
            road_mask, num_queries=20, seed=42
        )
        for q in queries:
            assert (q["start_row"], q["start_col"]) != (
                q["goal_row"],
                q["goal_col"],
            )

    def test_minimum_distance(self, road_mask):
        queries = generate_queries_for_image(
            road_mask, num_queries=20, seed=42
        )
        for q in queries:
            assert q["euclidean_distance"] >= 20.0

    def test_distance_class_present(self, road_mask):
        queries = generate_queries_for_image(
            road_mask, num_queries=20, seed=42
        )
        for q in queries:
            assert q["distance_class"] in ("short", "medium", "long")

    def test_reproducible(self, road_mask):
        q1 = generate_queries_for_image(road_mask, num_queries=20, seed=42)
        q2 = generate_queries_for_image(road_mask, num_queries=20, seed=42)
        for a, b in zip(q1, q2):
            assert a["start_row"] == b["start_row"]
            assert a["goal_col"] == b["goal_col"]

    def test_too_few_road_pixels(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[0, 0] = True  # only 1 road pixel
        queries = generate_queries_for_image(mask, num_queries=20)
        assert len(queries) == 0

    def test_different_seeds_different(self, road_mask):
        q1 = generate_queries_for_image(road_mask, num_queries=20, seed=42)
        q2 = generate_queries_for_image(road_mask, num_queries=20, seed=123)
        # Should differ in at least some queries
        starts1 = {(q["start_row"], q["start_col"]) for q in q1}
        starts2 = {(q["start_row"], q["start_col"]) for q in q2}
        # Very unlikely to have exact same set
        assert starts1 != starts2 or len(q1) == 0
