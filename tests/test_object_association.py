from pathlib import Path

import numpy as np
from PIL import Image

from visual_memory_lab.object_association import _fallback_embeddings, _pair_score


def test_pair_score_marks_large_displacement_as_possible_candidate_only() -> None:
    a = {"mask_area_fraction": 0.1, "score": 0.8, "mask_score": 0.8, "point_count": 300, "centroid_world_m": [0.0, 0.0, 0.0]}
    b = {"mask_area_fraction": 0.1, "score": 0.8, "mask_score": 0.8, "point_count": 300, "centroid_world_m": [1.0, 0.0, 0.0]}
    result = _pair_score(a, b, 0.95)
    assert result["association_score"] > 0.5
    assert result["centroid_distance_m"] == 1.0


def test_fallback_embeddings_are_normalized_and_deterministic() -> None:
    images = [Image.new("RGB", (10, 10), (20, 40, 80)), Image.new("RGB", (10, 10), (80, 40, 20))]
    first = _fallback_embeddings(images)
    second = _fallback_embeddings(images)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(2))
