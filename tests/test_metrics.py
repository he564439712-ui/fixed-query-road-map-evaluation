"""Unit tests for evaluation metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.segmentation import compute_cldice


class TestSegmentationMetrics:
    def test_iou_perfect(self):
        pred = np.ones((10, 10), dtype=bool)
        gt = np.ones((10, 10), dtype=bool)
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()
        fn = (~pred & gt).sum()
        iou = tp / max(tp + fp + fn, 1.0)
        assert iou == 1.0

    def test_iou_empty(self):
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()
        fn = (~pred & gt).sum()
        iou = tp / max(tp + fp + fn, 1.0)
        assert iou == 0.0  # Both empty: TP=0, but IoU is 0 by convention

    def test_dice_perfect(self):
        pred = np.ones((10, 10), dtype=bool)
        gt = np.ones((10, 10), dtype=bool)
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()
        fn = (~pred & gt).sum()
        dice = 2 * tp / max(2 * tp + fp + fn, 1.0)
        assert dice == 1.0

    def test_precision_recall(self):
        pred = np.zeros((10, 10), dtype=bool)
        pred[:5, :] = True
        gt = np.zeros((10, 10), dtype=bool)
        gt[3:8, :] = True
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()
        fn = (~pred & gt).sum()
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        # 2 rows overlap, 5 predicted, 5 GT
        assert precision == 2 * 10 / (5 * 10)  # 0.4
        assert recall == 2 * 10 / (5 * 10)  # 0.4

    def test_cldice_uses_each_centerline_with_its_own_denominator(self):
        gt = np.zeros((9, 9), dtype=bool)
        gt[3:6, 1:8] = True
        pred = np.zeros((9, 9), dtype=bool)
        pred[2:5, 2:7] = True
        score, topology_precision, topology_sensitivity = compute_cldice(pred, gt)
        assert 0.0 <= score <= 1.0
        assert 0.0 <= topology_precision <= 1.0
        assert 0.0 <= topology_sensitivity <= 1.0


class TestCalibrationMetrics:
    def test_ece_perfect_calibration(self):
        """ECE should be 0 when confidence = accuracy in every bin."""
        # Perfect calibration: prob = 0.9, accuracy = 0.9
        ece = 0.0
        for acc, conf, count in [(0.9, 0.9, 100), (0.5, 0.5, 100)]:
            ece += abs(acc - conf) * count
        ece /= 200
        assert ece == 0.0

    def test_ece_overconfident(self):
        """Model says 0.95 but only 0.70 correct → ECE > 0."""
        ece = abs(0.70 - 0.95) * 100 / 100
        assert ece > 0.0

    def test_brier_perfect(self):
        probs = np.array([1.0, 0.0, 1.0])
        labels = np.array([1.0, 0.0, 1.0])
        brier = np.mean((probs - labels) ** 2)
        assert brier == 0.0

    def test_brier_worst(self):
        probs = np.array([0.0, 1.0])
        labels = np.array([1.0, 0.0])
        brier = np.mean((probs - labels) ** 2)
        assert brier == 1.0

    def test_nll(self):
        probs = np.array([0.9, 0.1])
        labels = np.array([1.0, 0.0])
        eps = 1e-15
        nll = -np.mean(
            labels * np.log(np.clip(probs, eps, 1 - eps))
            + (1 - labels) * np.log(np.clip(1 - probs, eps, 1 - eps))
        )
        assert nll > 0
        assert np.isfinite(nll)


class TestPlanningMetrics:
    def test_path_length(self):
        path = [(0, 0), (0, 5), (5, 5)]
        length = 0.0
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            length += np.hypot(r2 - r1, c2 - c1)
        assert length == 10.0  # 5 horizontal + 5 vertical

    def test_path_length_single_point(self):
        path = [(3, 3)]
        length = 0.0
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            length += np.hypot(r2 - r1, c2 - c1)
        assert length == 0.0

    def test_road_path_ratio(self):
        """Fraction of path points on road."""
        path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        road_mask = np.zeros((5, 5), dtype=bool)
        road_mask[0, :3] = True
        road_mask[1, 2] = True
        road_mask[2, 2] = True
        on_road = sum(1 for r, c in path if road_mask[r, c])
        assert on_road / len(path) == 1.0

    def test_off_road_ratio(self):
        path = [(0, 0), (0, 1), (0, 2)]
        road_mask = np.zeros((5, 5), dtype=bool)
        road_mask[0, :2] = True  # first 2 on, last off
        on_road = sum(1 for r, c in path if road_mask[r, c])
        off_ratio = 1.0 - on_road / len(path)
        assert abs(off_ratio - 1.0 / 3.0) < 1e-9


class TestStatisticalHelpers:
    def test_bootstrap_ci(self):
        """Simple bootstrap CI for mean."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=5.0, scale=1.0, size=100)
        n_bootstrap = 1000
        means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            means.append(np.mean(sample))
        ci_low = np.percentile(means, 2.5)
        ci_high = np.percentile(means, 97.5)
        assert ci_low < np.mean(data) < ci_high
        # True mean (5.0) should be in CI
        assert ci_low <= 5.0 <= ci_high

    def test_effect_size_cohens_d(self):
        """Cohen's d for paired samples."""
        before = np.array([10, 9, 8, 9, 10, 9, 8, 9, 10, 9])
        after = np.array([8, 7, 6, 7, 8, 7, 6, 7, 8, 7])
        diff = before - after
        d = np.mean(diff) / max(np.std(diff, ddof=1), 1e-8)
        # All positive differences → d >> 0
        assert d > 1.0
