"""Tests for deterministic 3D change geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from visual_memory_lab.change_detection import (
    VoxelCloud,
    bidirectional_residuals,
    connected_components,
    evaluate_eth_change,
    load_voxel_cloud,
)


def _cloud(points: list[list[float]], voxel_size: float = 0.02) -> VoxelCloud:
    array = np.asarray(points, dtype=np.float64)
    return VoxelCloud(
        points=array,
        colors=np.zeros_like(array),
        normals=np.tile([0.0, 0.0, 1.0], (len(array), 1)),
        keys=np.floor(array / voxel_size).astype(np.int64),
        source_counts=np.ones(len(array), dtype=np.int64),
    )


def _write_ply(path: Path, points: list[list[float]]) -> None:
    vertex = np.empty(
        len(points),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("normal_x", "f4"), ("normal_y", "f4"), ("normal_z", "f4"),
        ],
    )
    array = np.asarray(points, dtype=np.float32)
    for index, name in enumerate(("x", "y", "z")):
        vertex[name] = array[:, index]
    vertex["red"] = 100
    vertex["green"] = 120
    vertex["blue"] = 140
    vertex["normal_x"] = 0
    vertex["normal_y"] = 0
    vertex["normal_z"] = 1
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def test_bidirectional_residuals_separate_added_and_removed_points() -> None:
    earlier = _cloud([[0, 0, 0], [0.02, 0, 0], [1, 0, 0]])
    current = _cloud([[0, 0, 0], [0.02, 0, 0], [2, 0, 0]])
    residuals = bidirectional_residuals(earlier, current)
    assert residuals["current_distances"].tolist() == [0.0, 0.0, 1.0]
    assert residuals["earlier_distances"].tolist() == [0.0, 0.0, 0.98]


def test_connected_components_filters_isolated_noise() -> None:
    keys = np.asarray([[0, 0, 0], [1, 0, 0], [10, 10, 10]], dtype=np.int64)
    components = connected_components(keys, min_voxels=2)
    assert len(components) == 1
    assert components[0].tolist() == [0, 1]


def test_voxel_loader_aggregates_points(tmp_path: Path) -> None:
    path = tmp_path / "mesh.ply"
    _write_ply(path, [[0.001, 0, 0], [0.009, 0, 0], [0.03, 0, 0]])
    cloud = load_voxel_cloud(path, voxel_size=0.02)
    assert len(cloud.points) == 2
    assert cloud.source_counts.tolist() == [2, 1]


def test_full_evaluation_writes_six_pairs(tmp_path: Path) -> None:
    mesh_paths = []
    base = [[x * 0.02, y * 0.02, 0.0] for x in range(6) for y in range(6)]
    for index in range(4):
        path = tmp_path / f"observation_{index}.ply"
        moved = base + [[1.0 + index * 0.1 + x * 0.02, y * 0.02, 0.0] for x in range(5) for y in range(5)]
        _write_ply(path, moved)
        mesh_paths.append(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "observations": [
                    {"logical_order": index, "mesh": {"path": str(path)}}
                    for index, path in enumerate(mesh_paths)
                ]
            }
        ),
        encoding="utf-8",
    )
    result = evaluate_eth_change(
        manifest_path=manifest,
        output=tmp_path / "result",
        min_cluster_voxels=2,
    )
    assert result["pair_count"] == 6
    assert len((tmp_path / "result" / "pairs.jsonl").read_text().splitlines()) == 6
