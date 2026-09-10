"""Regression tests for bugs found in the experiment-code audit (2026-08-06)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tiling import stitch_tiles, tile_locations
from src.eval.segmentation import compute_apls
from src.planning.costs import make_matched_score_cost, make_unified_risk_cost
from src.planning.grid import build_risk_map
from src.planning.topology_repair import (
    Endpoint,
    conservative_bridge_selection,
    compute_connection_cost,
    compute_skeleton,
    estimate_direction,
)


class TestTilingBorder:
    def test_border_pixels_not_zeroed(self):
        """Blend mask must never be exactly 0 on the tile border, so stitched
        image border pixels (covered by a single tile) are not forced to 0."""
        h, w = 512, 512
        coords = tile_locations(h, w, 256, 64)
        tiles = [np.full((256, 256), 0.8, dtype=np.float32) for _ in coords]
        result = stitch_tiles(tiles, coords, h, w, 64)
        # Border frame must be > 0 everywhere (was 0 before the fix)
        assert result[0, :].min() > 0.0
        assert result[:, 0].min() > 0.0
        assert result[h - 1, :].min() > 0.0
        assert result[:, w - 1].min() > 0.0
        # Interior reconstructs the constant value
        assert abs(result[64, 64] - 0.8) < 1e-3


class TestBuildRiskMapEmpty:
    def test_empty_mask_does_not_crash(self):
        """build_risk_map on an all-False mask must return all-inf risk,
        not raise (was ValueError on zero-size array)."""
        mask = np.zeros((10, 10), dtype=bool)
        risk, distance = build_risk_map(mask)
        assert risk.shape == mask.shape
        assert np.all(np.isinf(risk))

    def test_normal_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 2:8] = True
        risk, distance = build_risk_map(mask)
        assert risk.shape == mask.shape
        assert np.isfinite(risk[mask]).all()
        assert np.isinf(risk[~mask]).all()


class TestDirectionPointsOutward:
    def test_left_endpoint_points_left(self):
        """A horizontal road's left endpoint must point LEFT (toward the gap),
        i.e. OUT of the road body (was inverted before the fix)."""
        skeleton = np.zeros((30, 30), dtype=bool)
        skeleton[15, 5:25] = True  # horizontal road, body to the right of (15,5)
        direction = estimate_direction(skeleton, (15, 5))
        assert direction is not None
        # Column component must point left (negative)
        assert direction[1] < 0
        assert abs(direction[0]) < 0.1  # roughly horizontal

    def test_right_endpoint_points_right(self):
        skeleton = np.zeros((30, 30), dtype=bool)
        skeleton[15, 5:25] = True
        direction = estimate_direction(skeleton, (15, 24))
        assert direction is not None
        assert direction[1] > 0


class TestComponentLabelsLive:
    def test_merged_endpoints_get_zero_gain(self):
        """Connectivity-gain term must read CURRENT component labels from the
        array (updated by prior bridges), not stale Endpoint fields."""
        from dataclasses import dataclass

        from src.planning.topology_repair import Endpoint

        # Two endpoints in components 1 and 2, originally different.
        ep_a = Endpoint(row=0, col=0, direction=(0.0, 1.0), width=4.0, component_id=1)
        ep_b = Endpoint(row=0, col=10, direction=(0.0, -1.0), width=4.0, component_id=2)

        labels = np.ones((20, 20), dtype=np.int32) * 2
        labels[:, :] = 1  # simulate that a prior bridge already merged them
        confidence = np.full((20, 20), 0.9, dtype=np.float32)

        _, comps = compute_connection_cost(
            ep_a, ep_b, 10.0, confidence, labels
        )
        # Since live labels are now the same component, gain must be 0.
        assert comps["connectivity_gain"] == 0.0

        # And with distinct live labels, gain = -w_g.
        labels2 = np.zeros((20, 20), dtype=np.int32)
        labels2[:, :] = 1
        labels2[:, 10:] = 2
        _, comps2 = compute_connection_cost(
            ep_a, ep_b, 10.0, confidence, labels2
        )
        assert comps2["connectivity_gain"] == -1.0

    def test_chain_merge_uses_live_component_ids(self):
        ep_a = Endpoint(0, 4, (0.0, 1.0), 2.0, 1)
        ep_b = Endpoint(0, 5, (0.0, -1.0), 2.0, 2)
        ep_c = Endpoint(0, 10, (0.0, -1.0), 2.0, 3)
        labels = np.zeros((2, 15), dtype=np.int32)
        labels[:, :5] = 1
        labels[:, 5:10] = 2
        labels[:, 10:] = 3
        confidence = np.full(labels.shape, 0.99, dtype=np.float32)

        bridges = conservative_bridge_selection(
            [(ep_a, ep_b, 1.0), (ep_b, ep_c, 5.0)],
            confidence,
            labels,
            tau_gate=100.0,
        )

        assert len(bridges) == 2
        assert labels[ep_a.row, ep_a.col] == labels[ep_c.row, ep_c.col]


class TestHardMinimumQGate:
    @staticmethod
    def _candidate():
        ep_a = Endpoint(5, 2, (0.0, 1.0), 2.0, 1)
        ep_b = Endpoint(5, 8, (0.0, -1.0), 2.0, 2)
        labels = np.zeros((11, 11), dtype=np.int32)
        labels[:, :5] = 1
        labels[:, 6:] = 2
        return ep_a, ep_b, labels

    def test_rejects_bridge_with_one_low_q_pixel(self):
        ep_a, ep_b, labels = self._candidate()
        confidence = np.full(labels.shape, 0.9, dtype=np.float32)
        confidence[5, 5] = 0.1

        bridges = conservative_bridge_selection(
            [(ep_a, ep_b, 6.0)], confidence, labels,
            tau_gate=100.0, min_q=0.2,
        )

        assert bridges == []

    def test_accepts_bridge_when_curve_meets_minimum(self):
        ep_a, ep_b, labels = self._candidate()
        confidence = np.full(labels.shape, 0.9, dtype=np.float32)
        confidence[5, 5] = 0.2

        bridges = conservative_bridge_selection(
            [(ep_a, ep_b, 6.0)], confidence, labels,
            tau_gate=100.0, min_q=0.2,
        )

        assert len(bridges) == 1
        assert bridges[0].cost_components["curve_min_q"] == pytest.approx(0.2)

    def test_default_preserves_soft_gate_behavior(self):
        ep_a, ep_b, labels = self._candidate()
        confidence = np.full(labels.shape, 0.9, dtype=np.float32)
        confidence[5, 5] = 0.0

        bridges = conservative_bridge_selection(
            [(ep_a, ep_b, 6.0)], confidence, labels, tau_gate=100.0,
        )

        assert len(bridges) == 1


class TestSpatialBlur:
    def test_blur_does_not_mix_rgb_channels(self):
        from src.data.corruptions import gaussian_blur

        image = np.zeros((3, 9, 9), dtype=np.float32)
        image[0] = 1.0

        blurred = gaussian_blur(image, sigma=2.0)

        assert np.allclose(blurred[0], 1.0)
        assert np.allclose(blurred[1:], 0.0)


class TestCorruptionSeeds:
    def test_seed_is_stable_and_image_specific(self):
        from src.data.corruptions import deterministic_corruption_seed

        seed_a = deterministic_corruption_seed("image-a", "noise_light")
        seed_a_repeat = deterministic_corruption_seed("image-a", "noise_light")
        seed_b = deterministic_corruption_seed("image-b", "noise_light")

        assert seed_a == seed_a_repeat
        assert seed_a != seed_b


class TestApls:
    def test_perfect_identical(self):
        """APLS of a prediction identical to GT should be ~1.0."""
        gt = np.zeros((40, 40), dtype=bool)
        gt[20, 5:35] = True  # single horizontal road
        pred = gt.copy()
        assert compute_apls(pred, gt) > 0.99

    def test_disconnected_prediction_lower(self):
        """A prediction that breaks a road into two halves must score below a
        connected one (paths become unroutable)."""
        gt = np.zeros((40, 40), dtype=bool)
        gt[20, 5:35] = True

        broken = gt.copy()
        broken[20, 15:25] = False  # cut in the middle -> two components
        connected_score = compute_apls(gt, gt)
        broken_score = compute_apls(broken, gt)
        assert broken_score < connected_score

    def test_empty_gt(self):
        assert compute_apls(np.zeros((10, 10), dtype=bool), np.zeros((10, 10), dtype=bool)) == 1.0

class TestMatchedScoreCost:
    def test_same_score_gets_same_transform(self):
        score = np.asarray([[0.2, 0.5], [0.8, 0.99]], dtype=np.float32)
        distance = np.asarray([[0.0, 1.0], [3.0, 8.0]], dtype=np.float32)
        p_cost = make_matched_score_cost(score, distance, use_boundary=True)
        q_cost = make_matched_score_cost(score.copy(), distance, use_boundary=True)
        for idx in np.ndindex(score.shape):
            assert p_cost(idx) == pytest.approx(q_cost(idx))

    def test_boundary_toggle_changes_only_boundary_term(self):
        score = np.full((2, 2), 0.7, dtype=np.float32)
        distance = np.asarray([[0.0, 2.0], [5.0, 10.0]], dtype=np.float32)
        off = make_matched_score_cost(score, distance, lambda_b=1.5, tau_d=5.0,
                                      use_boundary=False)
        on = make_matched_score_cost(score, distance, lambda_b=1.5, tau_d=5.0,
                                     use_boundary=True)
        for idx in np.ndindex(score.shape):
            expected = 1.5 * np.exp(-float(distance[idx]) / 5.0)
            assert on(idx) - off(idx) == pytest.approx(expected, rel=1e-6)

    def test_legacy_unified_cost_is_factorial_full_cost(self):
        score = np.asarray([[0.1, 0.6], [0.9, 0.99]], dtype=np.float32)
        distance = np.asarray([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32)
        legacy = make_unified_risk_cost(score, distance, 2.0, 1.0, 5.0, 1.0)
        matched = make_matched_score_cost(score, distance, 2.0, 1.0, 5.0, 1.0,
                                          use_boundary=True)
        for idx in np.ndindex(score.shape):
            assert legacy(idx) == pytest.approx(matched(idx))
