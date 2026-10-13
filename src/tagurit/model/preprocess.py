"""
Turn batches of RGB tiles into normalised tensors for an image backbone.

Shared by every torch scorer so they resize and validate tiles the same
way and differ only in input size and normalisation constants. torch is
imported inside the function, so importing this module stays cheap.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tiles_to_tensor(
    tiles: Sequence[np.ndarray],
    size: int,
    mean: Sequence[float],
    std: Sequence[float],
    device: str,
) -> torch.Tensor:
    """
    Resize square tiles to ``size`` and normalise them per channel.

    Downscaling uses area interpolation, which averages rather than skips
    pixels, so small targets fade instead of vanishing. A tile that is not
    a square uint8 RGB array RAISES ValueError.

    Parameters:
        - tiles (Sequence[np.ndarray]): uint8 RGB square tiles
        - size (int): output side length
        - mean (Sequence[float]): per-channel mean, on a 0 to 1 scale
        - std (Sequence[float]): per-channel standard deviation, on a 0 to 1 scale
        - device (str): torch device for the result

    Return: float32 tensor of shape (len(tiles), 3, size, size)
    """
    import torch

    resized = []
    for tile in tiles:
        if tile.dtype != np.uint8 or tile.ndim != 3 or tile.shape[2] != 3:
            raise ValueError(f"expected a uint8 RGB tile, got {tile.dtype} {tile.shape}")
        if tile.shape[0] != tile.shape[1]:
            raise ValueError(f"expected a square tile, got {tile.shape[1]}x{tile.shape[0]}")
        if tile.shape[0] != size:
            shrinking = tile.shape[0] > size
            interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
            tile = cv2.resize(tile, (size, size), interpolation=interpolation)
        resized.append(tile)

    batch = torch.from_numpy(np.stack(resized)).to(device)
    batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (batch - mean_t) / std_t
