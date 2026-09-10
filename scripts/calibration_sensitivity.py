#!/usr/bin/env python
"""Report pooled and road-candidate ECE for clean and corrupted predictions.

The primary tables use mean image-level ECE over all pixels. This diagnostic
exposes how much that number is influenced by the dominant, near-zero
background predictions and reports ECE on nontrivial road candidates.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_mask
from src.data.splits import build_splits, resolve_image_paths
from src.eval.calibration import expected_calibration_error


ROOT = Path(__file__).resolve().parent.parent
PREDICTION_DIRS = {
    "clean": ROOT / "artifacts" / "predictions" / "test",
    "blur_medium": ROOT / "artifacts" / "predictions" / "corruptions" / "test" / "blur_medium",
}


def main() -> None:
    image_ids = build_splits(ROOT)["test_ids"]
    labels = np.concatenate([
        load_mask(resolve_image_paths(ROOT, image_id)[1]).ravel()
        for image_id in image_ids
    ]).astype(np.float64)
    print(f"road prevalence: {labels.mean():.6f}")
    for condition, prediction_dir in PREDICTION_DIRS.items():
        probabilities = np.concatenate([
            np.load(prediction_dir / f"{image_id}_prob.npy").ravel()
            for image_id in image_ids
        ]).astype(np.float64)
        pooled, *_ = expected_calibration_error(probabilities, labels, 15)
        print(f"{condition}: pooled ECE={pooled:.6f}")
        for threshold in (0.01, 0.05, 0.10):
            selected = probabilities >= threshold
            ece, *_ = expected_calibration_error(
                probabilities[selected], labels[selected], 15
            )
            print(
                f"  p>={threshold:.2f}: ECE={ece:.6f}, "
                f"pixels={selected.mean() * 100:.3f}%"
            )


if __name__ == "__main__":
    main()
