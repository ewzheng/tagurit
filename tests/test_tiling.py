"""Tests for frame tiling: grid coverage, overlap tests, decoding, and cropping."""

import cv2
import numpy as np
import pytest

from tagurit.model.tiling import Tile, crop_tiles, decode_image, tile_grid


def test_grid_tiles_exact_multiples_without_overlap() -> None:
    tiles = tile_grid(1024, 512, 256)
    assert [(t.left, t.top) for t in tiles] == [
        (0, 0), (256, 0), (512, 0), (768, 0),
        (0, 256), (256, 256), (512, 256), (768, 256),
    ]  # fmt: skip


def test_grid_shifts_last_row_and_column_flush_with_edges() -> None:
    tiles = tile_grid(3840, 2160, 512)
    lefts = sorted({t.left for t in tiles})
    tops = sorted({t.top for t in tiles})
    assert lefts[-1] == 3840 - 512 and len(lefts) == 8
    assert tops[-1] == 2160 - 512 and len(tops) == 5
    assert all(t.left + t.size <= 3840 and t.top + t.size <= 2160 for t in tiles)


def test_grid_covers_every_pixel() -> None:
    covered = np.zeros((1080, 1920), dtype=bool)
    for t in tile_grid(1920, 1080, 512):
        covered[t.top : t.top + t.size, t.left : t.left + t.size] = True
    assert covered.all()


def test_grid_rejects_frames_smaller_than_a_tile() -> None:
    with pytest.raises(ValueError):
        tile_grid(500, 1000, 512)
    with pytest.raises(ValueError):
        tile_grid(1000, 1000, 0)


def test_overlap_respects_edges_and_margin() -> None:
    tile = Tile(left=100, top=100, size=50)
    assert tile.overlaps(140, 140, 20, 20)
    assert not tile.overlaps(150, 100, 10, 10)  # touches the right edge only
    assert not tile.overlaps(160, 100, 10, 10, margin=9)
    assert tile.overlaps(160, 100, 10, 10, margin=11)
    with pytest.raises(ValueError):
        tile.overlaps(0, 0, 1, 1, margin=-1)


def test_decode_returns_rgb() -> None:
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # pure red in OpenCV's BGR order
    ok, png = cv2.imencode(".png", bgr)
    assert ok
    rgb = decode_image(png.tobytes())
    assert rgb.shape == (8, 8, 3)
    assert (rgb[..., 0] == 255).all() and (rgb[..., 2] == 0).all()


def test_decode_rejects_bad_bytes() -> None:
    with pytest.raises(ValueError):
        decode_image(b"")
    with pytest.raises(ValueError):
        decode_image(b"not an image")


def test_crop_returns_views_in_order() -> None:
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    a, b = crop_tiles(image, [Tile(0, 0, 2), Tile(4, 2, 2)])
    assert (a == image[0:2, 0:2]).all() and (b == image[2:4, 4:6]).all()
    assert np.shares_memory(a, image)


def test_crop_rejects_tiles_outside_the_frame() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_tiles(image, [Tile(5, 0, 2)])
    with pytest.raises(ValueError):
        crop_tiles(image, [Tile(-1, 0, 2)])
