from pathlib import Path

import numpy as np
from PIL import Image

from visual_memory_lab.rgbd_evidence import _mask_color_points, robust_extent


def test_robust_extent_ignores_extreme_outlier() -> None:
    points = np.vstack([np.zeros((20, 3)), np.asarray([[100.0, 100.0, 100.0]])])
    minimum, maximum = robust_extent(points)
    assert maximum[0] < 100.0
    assert minimum == [0.0, 0.0, 0.0]


def test_mask_color_linking_returns_matching_point_subset(tmp_path: Path) -> None:
    image = Image.new("RGB", (2, 1), (0, 0, 0))
    image.putpixel((0, 0), (240, 16, 16))
    mask_path = tmp_path / "mask.png"
    Image.new("L", (2, 1), 0).save(mask_path)
    mask = Image.open(mask_path)
    mask.putpixel((0, 0), 255)
    mask.save(mask_path)
    points = np.asarray([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
    colors = np.asarray([[239, 17, 18], [16, 240, 16]], dtype=np.uint8)
    selected = _mask_color_points(image, mask_path, points, colors)
    assert selected.tolist() == [[1.0, 2.0, 3.0]]
