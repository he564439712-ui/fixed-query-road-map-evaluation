"""Probability calibration via temperature scaling.

Temperature scaling (Guo et al., 2017):
  p_cal(x) = σ(logit(x) / T)

where T is optimized on a held-out calibration set to minimize NLL.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class TemperatureScaler:
    """Temperature scaling for binary segmentation.

    Optimizes a single scalar temperature T on logits from the
    calibration set.
    """

    def __init__(self, init_temperature: float = 1.0):
        self.temperature = torch.nn.Parameter(
            torch.tensor(init_temperature, dtype=torch.float32)
        )

    def calibrate(
        self,
        logits_list: list[np.ndarray],
        labels_list: list[np.ndarray],
        learning_rate: float = 0.01,
        max_iter: int = 100,
        device: str = "cpu",
    ) -> float:
        """Optimize temperature on calibration set.

        Args:
            logits_list: list of (H, W) logit arrays (pre-sigmoid)
            labels_list: list of (H, W) binary label arrays
            learning_rate: optimizer step size
            max_iter: max optimization iterations
            device: torch device

        Returns:
            optimized temperature value
        """
        # Stack all pixels
        all_logits = np.concatenate([l.flatten() for l in logits_list])
        all_labels = np.concatenate([l.flatten() for l in labels_list])

        logits_t = torch.from_numpy(all_logits).float().to(device)
        labels_t = torch.from_numpy(all_labels).float().to(device)

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=learning_rate, max_iter=max_iter
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled_logits = logits_t / self.temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        return float(self.temperature.item())

    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits, returning calibrated probs."""
        with torch.no_grad():
            logits_t = torch.from_numpy(logits).float()
            scaled = logits_t / self.temperature
            return torch.sigmoid(scaled).numpy().astype(np.float32)

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling, returning calibrated logits."""
        return logits / float(self.temperature.item())


def deep_ensemble_statistics(
    prob_maps: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ensemble mean and epistemic uncertainty.

    Args:
        prob_maps: list of (H, W) probability maps from each model

    Returns:
        mean_prob: (H, W) mean probability
        epistemic_std: (H, W) standard deviation
    """
    stacked = np.stack(prob_maps, axis=0)  # (M, H, W)
    mean_prob = np.mean(stacked, axis=0)
    epistemic_std = np.std(stacked, axis=0)
    return mean_prob.astype(np.float32), epistemic_std.astype(np.float32)


def conservative_lower_bound(
    calibrated_prob: np.ndarray,
    epistemic_std: np.ndarray,
    kappa: float = 1.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute q(x) = clip(p_cal - κ*σ_epi, ε, 1-ε)."""
    raw = calibrated_prob - kappa * epistemic_std
    return np.clip(raw, epsilon, 1.0 - epsilon).astype(np.float32)
