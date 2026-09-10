"""Statistical analysis utilities.

Implements:
  - Per-image aggregation (20 queries/image → 1 image stat)
  - Cluster bootstrap 95% CI
  - Paired Wilcoxon test
  - Holm-Bonferroni correction
  - Cohen's d effect size
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def aggregate_per_image(
    query_metrics: list[dict],
    image_key: str = "image_id",
) -> dict[str, dict[str, float]]:
    """Aggregate per-query metrics to per-image means.

    Each test image has 20 queries. We first average within each
    image to handle the intra-image correlation.

    Returns {image_id: {metric_name: mean_value}}.
    """
    from collections import defaultdict

    image_data: dict[str, list[dict]] = defaultdict(list)
    for row in query_metrics:
        image_data[row[image_key]].append(row)

    result = {}
    numeric_keys = [
        k for k in next(iter(query_metrics)).keys()
        if k not in (image_key, "method", "distance_class")
        and isinstance(next(iter(query_metrics))[k], (int, float))
    ]

    for image_id, rows in image_data.items():
        means = {}
        for key in numeric_keys:
            values = [r[key] for r in rows if not np.isnan(float(r[key]))]
            if values:
                means[key] = float(np.mean(values))
            else:
                means[key] = float("nan")
        result[image_id] = means

    return result


def cluster_bootstrap_ci(
    per_image_values: dict[str, list[float]],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Cluster bootstrap CI for per-image aggregated metrics.

    Resamples images (not queries) to preserve within-image structure.

    Args:
        per_image_values: {metric_name: [value_per_image]}
        n_bootstrap: number of bootstrap resamples
        alpha: significance level (0.05 → 95% CI)

    Returns:
        {metric_name: (mean, ci_lower, ci_upper)}
    """
    rng = np.random.default_rng(seed)
    results = {}

    for metric_name, values in per_image_values.items():
        values_arr = np.array(values, dtype=np.float64)
        # Remove NaN
        valid = values_arr[~np.isnan(values_arr)]
        if len(valid) == 0:
            results[metric_name] = (float("nan"), float("nan"), float("nan"))
            continue

        means = np.zeros(n_bootstrap)
        n = len(valid)
        for i in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            means[i] = np.mean(valid[idx])

        ci_lower = float(np.percentile(means, 100 * alpha / 2))
        ci_upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
        results[metric_name] = (float(np.mean(valid)), ci_lower, ci_upper)

    return results


def paired_wilcoxon(
    method_a: list[float],
    method_b: list[float],
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test.

    Args:
        method_a, method_b: paired per-image metric values
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        (statistic, p_value)
    """
    a = np.array(method_a, dtype=np.float64)
    b = np.array(method_b, dtype=np.float64)
    # Drop pairs where either is NaN
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 5:
        return float("nan"), float("nan")

    try:
        stat, p = stats.wilcoxon(a[mask], b[mask], alternative=alternative)
        return float(stat), float(p)
    except ValueError:
        return float("nan"), float("nan")


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple comparisons.

    Returns adjusted p-values.
    """
    n = len(p_values)
    # Sort by p-value but track original indices
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])

    adjusted = [0.0] * n
    for rank, (p, original_idx) in enumerate(indexed):
        adjusted[original_idx] = min(1.0, p * (n - rank))

    # Ensure monotonicity
    for i in range(n - 1):
        rank_i = indexed[i][1]
        rank_j = indexed[i + 1][1]
        adjusted[rank_j] = max(adjusted[rank_j], adjusted[rank_i])

    return adjusted


def cohens_d(
    a: list[float], b: list[float], paired: bool = True
) -> float:
    """Cohen's d effect size.

    For paired: d = mean(diff) / std(diff).
    For independent: d = (mean(a) - mean(b)) / pooled_std.
    """
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    mask = ~np.isnan(a_arr) & ~np.isnan(b_arr)
    a_valid = a_arr[mask]
    b_valid = b_arr[mask]

    if len(a_valid) < 3:
        return float("nan")

    if paired:
        diff = a_valid - b_valid
        std_diff = float(np.std(diff, ddof=1))
        if std_diff < 1e-8:
            return 0.0
        return float(np.mean(diff) / std_diff)
    else:
        pooled_std = np.sqrt(
            (np.var(a_valid, ddof=1) + np.var(b_valid, ddof=1)) / 2.0
        )
        if pooled_std < 1e-8:
            return 0.0
        return float((np.mean(a_valid) - np.mean(b_valid)) / pooled_std)


def report_significance(
    per_image_baseline: list[float],
    per_image_proposed: list[float],
    metric_name: str,
    alpha: float = 0.05,
) -> dict:
    """Full significance report for one metric comparison.

    Returns dict with: mean_diff, ci_lower, ci_upper, p_value,
    adjusted_p, cohens_d, significant (bool).
    """
    _, p_value = paired_wilcoxon(
        per_image_baseline, per_image_proposed, alternative="two-sided"
    )

    d = cohens_d(per_image_proposed, per_image_baseline, paired=True)

    # Bootstrap CI of the difference
    rng = np.random.default_rng(42)
    a = np.array(per_image_baseline, dtype=np.float64)
    b = np.array(per_image_proposed, dtype=np.float64)
    diff = b - a
    valid_diff = diff[~np.isnan(diff)]
    if len(valid_diff) == 0:
        return {
            "metric": metric_name,
            "mean_diff": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": p_value,
            "cohens_d": d,
            "significant": False,
        }

    n = len(valid_diff)
    boot_means = np.zeros(10000)
    for i in range(10000):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(valid_diff[idx])

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    mean_diff = float(np.mean(valid_diff))

    # Significant if CI doesn't cross 0
    significant = ci_low > 0 or ci_high < 0

    return {
        "metric": metric_name,
        "mean_diff": mean_diff,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "p_value": p_value,
        "cohens_d": d,
        "significant": significant,
    }
