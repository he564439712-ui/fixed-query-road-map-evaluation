"""Calibration evaluation metrics.

Implements: ECE, Brier score, NLL, reliability diagram data,
uncertainty-error detection AUROC/AUPRC.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Expected Calibration Error (ECE).

    Partitions predictions into equal-width bins and measures
    |accuracy(bin) - confidence(bin)| for each bin.

    Args:
        probabilities: (N,) predicted probabilities
        labels: (N,) binary ground truth
        num_bins: number of bins

    Returns:
        ece: scalar ECE value
        bin_confidences: mean confidence per bin
        bin_accuracies: mean accuracy per bin
        bin_counts: number of samples per bin
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_confidences = np.zeros(num_bins)
    bin_accuracies = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for i in range(num_bins):
        mask = (probabilities >= bin_boundaries[i]) & (
            probabilities < bin_boundaries[i + 1]
        )
        # Include right boundary in last bin
        if i == num_bins - 1:
            mask = (probabilities >= bin_boundaries[i]) & (
                probabilities <= bin_boundaries[i + 1]
            )

        bin_counts[i] = int(mask.sum())
        if bin_counts[i] > 0:
            bin_confidences[i] = float(np.mean(probabilities[mask]))
            bin_accuracies[i] = float(np.mean(labels[mask]))

    ece = float(
        np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences))
        / max(np.sum(bin_counts), 1)
    )
    return ece, bin_confidences, bin_accuracies, bin_counts


def brier_score(
    probabilities: np.ndarray, labels: np.ndarray
) -> float:
    """Brier score: MSE between predicted probability and binary label."""
    return float(np.mean((probabilities - labels) ** 2))


def negative_log_likelihood(
    probabilities: np.ndarray,
    labels: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    """Binary cross-entropy / negative log-likelihood."""
    p = np.clip(probabilities, epsilon, 1.0 - epsilon)
    nll = -(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))
    return float(np.mean(nll))


def uncertainty_error_detection(
    uncertainty: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float]:
    """How well does uncertainty identify prediction errors?

    Args:
        uncertainty: (N,) per-pixel uncertainty values
        errors: (N,) binary error mask (1 = misclassified)

    Returns:
        auroc: AUROC for error detection
        auprc: AUPRC for error detection
    """
    if len(np.unique(errors)) < 2:
        return 0.5, 0.0

    try:
        auroc = float(roc_auc_score(errors, uncertainty))
    except ValueError:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(errors, uncertainty))
    except ValueError:
        auprc = 0.0

    return auroc, auprc


def calibration_metrics(
    probability: np.ndarray,
    ground_truth: np.ndarray,
    uncertainty: np.ndarray | None = None,
    num_bins: int = 15,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all calibration metrics for one image.

    Args:
        probability: (H, W) predicted probability
        ground_truth: (H, W) binary mask
        uncertainty: (H, W) uncertainty map (entropy or epistemic std)
        threshold: error threshold for binary prediction

    Returns dict of metric_name → value.
    """
    p_flat = probability.flatten().astype(np.float64)
    gt_flat = ground_truth.flatten().astype(np.float64)

    ece, _, _, _ = expected_calibration_error(p_flat, gt_flat, num_bins)

    metrics = {
        "ece": ece,
        "brier": brier_score(p_flat, gt_flat),
        "nll": negative_log_likelihood(p_flat, gt_flat),
    }

    if uncertainty is not None:
        pred_binary = probability >= threshold
        errors = (pred_binary != ground_truth.astype(bool)).flatten()
        u_flat = uncertainty.flatten()
        auroc, auprc = uncertainty_error_detection(u_flat, errors)
        metrics["auroc_error"] = auroc
        metrics["auprc_error"] = auprc

    return metrics
