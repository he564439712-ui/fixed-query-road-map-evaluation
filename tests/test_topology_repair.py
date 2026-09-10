"""Unit tests for topology repair module.

Tests cover:
  - Skeletonization
  - Endpoint detection
  - Candidate pair generation
  - Cost computation
  - Bridge validation against GT
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestSkeletonization:
    def test_skeleton_binary(self):
        """Skeleton of a binary mask should be a thin line."""
        from skimage.morphology import skeletonize

        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 10:40] = True  # 10×30 rectangle
        skeleton = skeletonize(mask)
        # Skeleton should be thinner than original
        assert skeleton.sum() < mask.sum()
        # Skeleton should preserve connectivity
        assert skeleton.any()

    def test_skeleton_single_pixel(self):
        from skimage.morphology import skeletonize

        mask = np.ones((1, 1), dtype=bool)
        skeleton = skeletonize(mask)
        assert skeleton[0, 0]


class TestEndpointDetection:
    def count_endpoints(self, skeleton: np.ndarray) -> int:
        """Count degree-1 nodes in skeleton."""
        from scipy import ndimage

        # Count neighbors for each skeleton pixel
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = ndimage.convolve(
            skeleton.astype(np.uint8), kernel, mode="constant", cval=0
        )
        # Subtract self (center pixel)
        neighbor_count = neighbor_count - skeleton.astype(np.uint8)
        # Endpoints: exactly 1 neighbor
        endpoints = (skeleton) & (neighbor_count == 1)
        return int(endpoints.sum())

    def test_straight_line_endpoints(self):
        """A straight skeleton line should have 2 endpoints."""
        skeleton = np.zeros((30, 30), dtype=bool)
        skeleton[15, 5:25] = True
        assert self.count_endpoints(skeleton) == 2

    def test_cross_endpoints(self):
        """A cross-shaped skeleton should have 4 endpoints."""
        skeleton = np.zeros((30, 30), dtype=bool)
        skeleton[15, 5:25] = True  # horizontal
        skeleton[5:25, 15] = True  # vertical
        assert self.count_endpoints(skeleton) == 4

    def test_loop_no_endpoints(self):
        """A closed loop should have 0 endpoints."""
        skeleton = np.zeros((30, 30), dtype=bool)
        skeleton[10, 10:20] = True  # top
        skeleton[20, 10:20] = True  # bottom
        skeleton[10:21, 10] = True  # left
        skeleton[10:21, 20] = True  # right
        assert self.count_endpoints(skeleton) == 0

    def test_no_skeleton(self):
        skeleton = np.zeros((10, 10), dtype=bool)
        assert self.count_endpoints(skeleton) == 0


class TestCandidatePairs:
    def test_max_radius_constraint(self):
        """Candidate endpoint pairs must be within max_radius."""
        endpoints = [(0, 0), (0, 10), (0, 50)]
        max_radius = 20
        pairs = []
        for i, a in enumerate(endpoints):
            for j, b in enumerate(endpoints):
                if i >= j:
                    continue
                dist = np.hypot(a[0] - b[0], a[1] - b[1])
                if dist <= max_radius:
                    pairs.append((a, b, dist))
        # (0,0)-(0,10): dist 10 → included
        # (0,0)-(0,50): dist 50 → excluded
        # (0,10)-(0,50): dist 40 → excluded
        assert len(pairs) == 1

    def test_min_radius_constraint(self):
        """Endpoints too close should be excluded."""
        endpoints = [(0, 0), (0, 1)]
        min_radius = 2
        pairs = []
        for i, a in enumerate(endpoints):
            for j, b in enumerate(endpoints):
                if i >= j:
                    continue
                dist = np.hypot(a[0] - b[0], a[1] - b[1])
                if dist >= min_radius:
                    pairs.append((a, b))
        assert len(pairs) == 0


class TestDirectionConsistency:
    def compute_direction(self, skeleton: np.ndarray, point: tuple) -> tuple | None:
        """Estimate road direction at an endpoint using local skeleton."""
        r, c = point
        h, w = skeleton.shape
        # Look at the skeleton segment within a small window
        vectors = []
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and skeleton[nr, nc]:
                    if dr != 0 or dc != 0:
                        vectors.append((dr, dc))
        if not vectors:
            return None
        # Average direction
        avg_dr = np.mean([v[0] for v in vectors])
        avg_dc = np.mean([v[1] for v in vectors])
        norm = np.hypot(avg_dr, avg_dc)
        if norm < 1e-6:
            return None
        return (avg_dr / norm, avg_dc / norm)

    def test_same_direction_consistency(self):
        """Two endpoints pointing toward each other should be consistent."""
        # Endpoint at (10, 5) pointing right, at (10, 15) pointing left
        dir_a = (0.0, 1.0)  # right
        dir_b = (0.0, -1.0)  # left
        # Direction consistency: dot product of dir_a and vector from a to b
        gap_direction = (0.0, 1.0)  # from a to b
        dot_a = dir_a[0] * gap_direction[0] + dir_a[1] * gap_direction[1]
        dot_b = dir_b[0] * (-gap_direction[0]) + dir_b[1] * (-gap_direction[1])
        # Both should point toward each other → positive dot products
        assert dot_a > 0
        assert dot_b > 0

    def test_opposite_direction_inconsistent(self):
        """Two endpoints pointing away from each other should be inconsistent."""
        dir_a = (-0.6, -0.8)  # pointing away from b
        dir_b = (1.0, 0.0)  # pointing away from a
        gap_direction = (1.0, 0.0)  # from a to b
        dot_a = dir_a[0] * gap_direction[0] + dir_a[1] * gap_direction[1]
        # dir_a points opposite → negative dot
        assert dot_a < 0


class TestBridgeValidation:
    def test_correct_bridge_detection(self):
        """A bridge mostly on dilated GT should be correct."""
        # Simulated GT mask (dilated)
        gt_dilated = np.zeros((20, 20), dtype=bool)
        gt_dilated[8:12, :] = True  # horizontal road

        # Bridge line connecting two components
        bridge = [(10, 3), (10, 4), (10, 5), (10, 6), (10, 7)]

        on_gt = sum(1 for r, c in bridge if gt_dilated[r, c])
        ratio = on_gt / len(bridge)
        assert ratio > 0.5

    def test_false_bridge_detection(self):
        """A bridge mostly off GT should be incorrect."""
        gt_dilated = np.zeros((20, 20), dtype=bool)
        gt_dilated[8:12, :5] = True  # short road segment

        # Bridge going through non-road area
        bridge = [(10, 8), (10, 9), (10, 10), (10, 11), (10, 12)]

        on_gt = sum(1 for r, c in bridge if gt_dilated[r, c])
        ratio = on_gt / len(bridge)
        assert ratio < 0.5

    def test_bridge_requires_same_gt_component(self):
        from src.eval.topology import bridge_metrics
        from src.planning.topology_repair import Bridge, Endpoint

        ground_truth = np.zeros((30, 30), dtype=bool)
        ground_truth[10, 2:12] = True
        ground_truth[10, 18:28] = True
        start = Endpoint(10, 10, (0.0, 1.0), 2.0, 1)
        end = Endpoint(10, 20, (0.0, -1.0), 2.0, 2)
        curve = [(10, col) for col in range(10, 21)]
        bridge = Bridge(start, end, curve, 0.0, {})

        metrics = bridge_metrics([bridge], ground_truth, dilation_radius=4)

        assert metrics["endpoint_consistency"] == 0.0
        assert metrics["bridge_precision"] == 0.0


class TestGeodesicBridgeCurve:
    @staticmethod
    def _arched_confidence_case():
        """Return endpoints separated by a low-q gap with a supported detour."""
        from src.planning.topology_repair import Endpoint

        confidence = np.full((31, 31), 0.02, dtype=np.float32)
        # One-pixel inverted-U corridor: up, across, and down.  The direct
        # endpoint chord remains low confidence except at its two endpoints.
        for row in range(7, 16):
            confidence[row, 4] = 0.99
            confidence[row, 26] = 0.99
        confidence[7, 4:27] = 0.99
        ep_a = Endpoint(15, 4, (0.0, 1.0), 2.0, 1)
        ep_b = Endpoint(15, 26, (0.0, -1.0), 2.0, 2)
        return confidence, ep_a, ep_b

    def test_geodesic_prefers_supported_curved_corridor(self):
        from src.planning.topology_repair import find_bridge_curve

        confidence, ep_a, ep_b = self._arched_confidence_case()
        straight = find_bridge_curve(ep_a, ep_b, confidence, mode="straight")
        geodesic = find_bridge_curve(
            ep_a,
            ep_b,
            confidence,
            mode="geodesic",
            geodesic_margin=10,
            geodesic_risk_weight=3.0,
        )

        assert geodesic[0] == (ep_a.row, ep_a.col)
        assert geodesic[-1] == (ep_b.row, ep_b.col)
        assert any(row != ep_a.row for row, _ in geodesic)
        assert all(
            max(abs(r1 - r0), abs(c1 - c0)) == 1
            for (r0, c0), (r1, c1) in zip(geodesic, geodesic[1:])
        )
        straight_q = np.mean([confidence[r, c] for r, c in straight])
        geodesic_q = np.mean([confidence[r, c] for r, c in geodesic])
        assert geodesic_q > straight_q + 0.5

    def test_curve_mode_changes_geometry_not_acceptance(self):
        from src.planning.topology_repair import conservative_bridge_selection

        confidence, ep_a, ep_b = self._arched_confidence_case()
        labels = np.zeros(confidence.shape, dtype=np.int32)
        labels[:, :15] = 1
        labels[:, 16:] = 2
        candidate = [(ep_a, ep_b, 22.0)]

        straight = conservative_bridge_selection(
            candidate,
            confidence,
            labels.copy(),
            tau_gate=100.0,
            curve_mode="straight",
        )
        geodesic = conservative_bridge_selection(
            candidate,
            confidence,
            labels.copy(),
            tau_gate=100.0,
            curve_mode="geodesic",
            geodesic_margin=10,
            geodesic_risk_weight=3.0,
        )

        assert len(straight) == len(geodesic) == 1
        assert straight[0].start == geodesic[0].start
        assert straight[0].end == geodesic[0].end
        assert straight[0].cost == pytest.approx(geodesic[0].cost)
        assert straight[0].curve != geodesic[0].curve

    def test_unknown_curve_mode_is_rejected(self):
        from src.planning.topology_repair import find_bridge_curve

        confidence, ep_a, ep_b = self._arched_confidence_case()
        with pytest.raises(ValueError, match="Unknown bridge curve mode"):
            find_bridge_curve(ep_a, ep_b, confidence, mode="not-a-mode")


class TestGeometricRepairBaseline:
    def test_geometric_repair_reconnects_facing_segments(self):
        """The confidence-free baseline must repair a simple geometric gap."""
        from scipy import ndimage

        from src.planning.topology_repair import repair_topology_geometric

        mask = np.zeros((80, 80), dtype=bool)
        mask[40, 10:31] = True
        mask[40, 40:61] = True

        repaired, bridges, _ = repair_topology_geometric(
            mask, max_radius=20.0, tau_gate=1.0, bridge_width=0
        )
        labels, count = ndimage.label(
            repaired, structure=np.ones((3, 3), dtype=np.uint8)
        )

        assert bridges
        assert count == 1
        assert labels[40, 20] == labels[40, 50]

    def test_geometric_repair_does_not_depend_on_confidence(self):
        """The baseline API intentionally accepts no probability or std map."""
        from src.planning.topology_repair import repair_topology_geometric

        mask = np.zeros((20, 20), dtype=bool)
        repaired, bridges, skeleton = repair_topology_geometric(mask)

        assert not repaired.any()
        assert bridges == []
        assert not skeleton.any()
