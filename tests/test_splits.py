"""Unit tests for data splits."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import (
    build_splits,
    count_per_split,
    deterministic_split,
    get_split_ids,
    load_metadata,
    resolve_image_paths,
    verify_no_overlap,
)


@pytest.fixture
def root():
    return Path(__file__).resolve().parent.parent


class TestDeterministicSplit:
    def test_empty_holdout(self):
        items = ["a", "b", "c"]
        kept, held = deterministic_split(items, 0, seed=42)
        assert kept == items
        assert held == []

    def test_full_holdout(self):
        items = ["a", "b", "c"]
        kept, held = deterministic_split(items, 5, seed=42)
        assert kept == []
        assert held == items

    def test_reproducible(self):
        items = [f"img_{i}" for i in range(100)]
        kept1, held1 = deterministic_split(items, 20, seed=42)
        kept2, held2 = deterministic_split(items, 20, seed=42)
        assert kept1 == kept2
        assert held1 == held2

    def test_different_seeds_different(self):
        items = [f"img_{i}" for i in range(100)]
        _, held1 = deterministic_split(items, 20, seed=42)
        _, held2 = deterministic_split(items, 20, seed=123)
        # Very unlikely to be identical (hash-based)
        assert held1 != held2

    def test_no_overlap(self):
        items = [f"img_{i}" for i in range(100)]
        kept, held = deterministic_split(items, 30, seed=42)
        assert set(kept).isdisjoint(set(held))
        assert len(kept) + len(held) == len(items)


class TestMetadataLoading:
    def test_load_metadata(self, root):
        metadata = load_metadata(root)
        assert len(metadata) == 1171
        required_cols = {"image_id", "split", "tiff_image_path", "tif_label_path"}
        for row in metadata:
            assert required_cols.issubset(set(row.keys()))

    def test_get_split_ids(self, root):
        metadata = load_metadata(root)
        train_ids = get_split_ids(metadata, "train")
        val_ids = get_split_ids(metadata, "val")
        test_ids = get_split_ids(metadata, "test")
        assert len(train_ids) == 1108
        assert len(val_ids) == 14
        assert len(test_ids) == 49


class TestBuildSplits:
    def test_counts(self, root):
        splits = build_splits(root)
        counts = count_per_split(splits)
        assert counts["seg_train"] == 998
        assert counts["calibration"] == 110
        assert counts["val"] == 14
        assert counts["test"] == 49
        assert counts["total"] == 1171

    def test_no_overlap(self, root):
        splits = build_splits(root)
        errors = verify_no_overlap(splits)
        assert len(errors) == 0, f"Split errors: {errors}"

    def test_no_leakage_between_sets(self, root):
        splits = build_splits(root)
        seg_set = set(splits["seg_train_ids"])
        cal_set = set(splits["calibration_ids"])
        val_set = set(splits["val_ids"])
        test_set = set(splits["test_ids"])

        assert seg_set.isdisjoint(cal_set)
        assert seg_set.isdisjoint(val_set)
        assert seg_set.isdisjoint(test_set)
        assert cal_set.isdisjoint(val_set)
        assert cal_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_total_images(self, root):
        splits = build_splits(root)
        all_ids = (
            set(splits["seg_train_ids"])
            | set(splits["calibration_ids"])
            | set(splits["val_ids"])
            | set(splits["test_ids"])
        )
        assert len(all_ids) == 1171

    def test_calibration_not_in_train(self, root):
        splits = build_splits(root)
        cal_set = set(splits["calibration_ids"])
        train_set = set(splits["seg_train_ids"])
        assert cal_set.isdisjoint(train_set)

    def test_test_set_locked(self, root):
        """Test set should never change regardless of calibration_seed."""
        splits1 = build_splits(root, calibration_seed=42)
        splits2 = build_splits(root, calibration_seed=999)
        assert splits1["test_ids"] == splits2["test_ids"]
        assert splits1["val_ids"] == splits2["val_ids"]


class TestResolvePaths:
    def test_resolve_train_image(self, root):
        img_path, mask_path = resolve_image_paths(root, "10078660_15")
        assert img_path == root / "tiff" / "train" / "10078660_15.tiff"
        assert mask_path == root / "tiff" / "train_labels" / "10078660_15.tif"

    def test_resolve_test_image(self, root):
        img_path, mask_path = resolve_image_paths(root, "24478825_15")
        assert img_path == root / "tiff" / "test" / "24478825_15.tiff"
        assert mask_path == root / "tiff" / "test_labels" / "24478825_15.tif"

    def test_resolve_nonexistent(self, root):
        with pytest.raises(ValueError):
            resolve_image_paths(root, "nonexistent_id")
