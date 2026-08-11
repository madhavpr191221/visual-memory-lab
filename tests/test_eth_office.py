"""Tests for ETH Office preparation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visual_memory_lab.eth_office import _decode_image, evenly_spaced_indices


def test_evenly_spaced_indices_cover_sequence_boundaries() -> None:
    assert evenly_spaced_indices(800, 8) == [0, 114, 228, 342, 457, 571, 685, 799]
    assert evenly_spaced_indices(3, 8) == [0, 1, 2]


def test_decode_bgr_image_to_rgb() -> None:
    message = SimpleNamespace(
        width=2,
        height=1,
        step=6,
        encoding="bgr8",
        data=np.asarray([0, 0, 255, 0, 255, 0], dtype=np.uint8),
    )
    image = _decode_image(message)
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert image.getpixel((1, 0)) == (0, 255, 0)

