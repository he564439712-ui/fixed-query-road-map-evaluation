from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.data.deepglobe import (
    DeepGlobeRoadPatchDataset,
    build_deepglobe_splits,
    discover_deepglobe_ids,
)


def _write_pair(root: Path, image_id: str) -> None:
    data_dir = root / "data" / "train"
    data_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[6:10, :] = 255
    Image.fromarray(image).save(data_dir / f"{image_id}_sat.jpg")
    Image.fromarray(mask).save(data_dir / f"{image_id}_mask.png")


def test_deepglobe_split_is_complete_disjoint_and_deterministic(tmp_path: Path):
    for index in range(20):
        _write_pair(tmp_path, f"tile_{index:03d}")

    first = build_deepglobe_splits(tmp_path, seed=7)
    second = build_deepglobe_splits(tmp_path, seed=7)
    changed = build_deepglobe_splits(tmp_path, seed=8)

    assert first == second
    assert first != changed
    split_sets = [set(ids) for ids in first.values()]
    assert set.union(*split_sets) == set(discover_deepglobe_ids(tmp_path))
    assert sum(len(ids) for ids in split_sets) == len(set.union(*split_sets))


def test_deepglobe_patch_dataset_resolves_flat_filenames(tmp_path: Path):
    _write_pair(tmp_path, "123")
    dataset = DeepGlobeRoadPatchDataset(
        image_ids=["123"],
        root=tmp_path,
        patch_size=8,
        patches_per_image=1,
        augment=False,
    )

    image, mask = dataset[0]

    assert tuple(image.shape) == (3, 8, 8)
    assert tuple(mask.shape) == (1, 8, 8)
    assert set(mask.unique().tolist()).issubset({0.0, 1.0})
