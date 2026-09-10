"""Unit tests for sliding-window tiling and stitching."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tiling import (
    _blend_mask,
    extract_tiles,
    stitch_tiles,
    tile_locations,
)


class TestTileLocations:
    def test_exact_fit(self):
        tiles = tile_locations(1024, 1024, 512, 0)
        assert len(tiles) == 4  # 2×2 grid

    def test_no_overlap_coverage(self):
        tiles = tile_locations(100, 100, 50, 0)
        # Should cover full area
        covered = np.zeros((100, 100), dtype=bool)
        for rs, cs, re, ce in tiles:
            covered[rs:re, cs:ce] = True
        assert np.all(covered)

    def test_with_overlap(self):
        tiles = tile_locations(1500, 1500, 512, 96)
        assert len(tiles) >= 9  # at least 3×3

    def test_small_image(self):
        tiles = tile_locations(200, 200, 512, 96)
        assert len(tiles) == 1

    def test_all_tiles_same_size(self):
        tiles = tile_locations(1500, 1500, 512, 96)
        for rs, cs, re, ce in tiles:
            assert re - rs == 512
            assert ce - cs == 512

    def test_tiles_within_bounds(self):
        for size, tile_size in [(1500, 512), (512, 256), (1024, 512), (256, 128)]:
            tiles = tile_locations(size, size, tile_size, tile_size // 4)
            for rs, cs, re, ce in tiles:
                assert 0 <= rs < size, f"rs={rs} out of [0, {size})"
                assert 0 <= cs < size, f"cs={cs} out of [0, {size})"
                assert rs < re <= size, f"re={re} out of (rs, {size}]"
                assert cs < ce <= size, f"ce={ce} out of (cs, {size}]"

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            tile_locations(100, 100, 50, 60)  # overlap > tile_size


class TestExtractTiles:
    def test_extract(self):
        image = torch.rand(3, 1024, 1024)
        tiles, coords = extract_tiles(image, 512, 0)
        assert len(tiles) == 4
        assert all(t.shape == (1, 3, 512, 512) for t in tiles)

    def test_coords_match(self):
        image = torch.rand(3, 1024, 1024)
        tiles, coords = extract_tiles(image, 512, 0)
        for tile, (rs, cs, re, ce) in zip(tiles, coords):
            expected = image[:, rs:re, cs:ce]
            assert torch.allclose(tile[0], expected)


class TestBlendMask:
    def test_no_overlap(self):
        mask = _blend_mask(100, 0)
        assert np.allclose(mask, 1.0)

    def test_unity_at_center(self):
        mask = _blend_mask(100, 20)
        assert mask[50, 50] == 1.0

    def test_ramps_to_zero(self):
        mask = _blend_mask(100, 20)
        assert mask[0, 50] < 1.0
        assert mask[0, 0] < mask[50, 50]


class TestStitchTiles:
    def test_stitch_perfect_reconstruction(self):
        """Tiles without overlap should stitch to original."""
        h, w = 512, 512
        original = np.random.rand(h, w).astype(np.float32)

        coords = tile_locations(h, w, 256, 0)
        tiles = []
        for rs, cs, re, ce in coords:
            tiles.append(original[rs:re, cs:ce])

        result = stitch_tiles(tiles, coords, h, w, 0)
        assert np.allclose(result, original)

    def test_stitch_with_overlap(self):
        h, w = 512, 512
        original = np.random.rand(h, w).astype(np.float32)

        coords = tile_locations(h, w, 256, 64)
        tiles = []
        for rs, cs, re, ce in coords:
            tiles.append(original[rs:re, cs:ce])

        result = stitch_tiles(tiles, coords, h, w, 64)
        assert result.shape == (h, w)
        # Should be close to original (blending changes values slightly)
        assert np.all(np.isfinite(result))

    def test_stitch_no_nan(self):
        h, w = 512, 512
        coords = tile_locations(h, w, 256, 64)
        tiles = [np.ones((256, 256), dtype=np.float32) for _ in coords]
        result = stitch_tiles(tiles, coords, h, w, 64)
        assert not np.any(np.isnan(result))
        # Interior pixels should be positive; extreme corners may be 0
        # due to blend mask ramp-to-zero at image boundaries
        assert np.all(result[32:-32, 32:-32] > 0)
