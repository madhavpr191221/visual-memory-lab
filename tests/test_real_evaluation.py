"""Tests for pose-grounded and semantic-zone retrieval metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visual_memory_lab.evaluation import (
    evaluate_pose_retrieval,
    evaluate_text_zones,
    rotation_errors_deg,
)
from visual_memory_lab.memory import MemoryIndex


def _record(observation_id: str, x: float, angle_deg: float, *, step: int = 0) -> dict[str, object]:
    angle = np.radians(angle_deg)
    pose = np.eye(4)
    pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    pose[0, 3] = x
    return {
        "observation_id": observation_id,
        "episode_id": observation_id.split(":")[0],
        "sequence_id": observation_id.split(":")[0],
        "step": step,
        "image_path": "unused.png",
        "camera_pose": {"convention": "camera_to_world", "matrix": pose.tolist()},
    }


def _index(tmp_path: Path, name: str, records: list[dict[str, object]], embeddings: np.ndarray) -> MemoryIndex:
    manifest = {"model": {"id": "test/clip", "revision": "1"}}
    return MemoryIndex(
        root=tmp_path / name,
        manifest=manifest,
        records=records,
        embeddings=embeddings.astype(np.float32),
        source=tmp_path,
        image_root=tmp_path,
    )


class TextEncoder:
    model_id = "test/clip"
    model_revision = "1"

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        rows = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))
        return rows


def test_rotation_errors_and_pose_hit_metrics(tmp_path: Path) -> None:
    rotations = np.stack([np.eye(3), np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])])
    assert np.allclose(rotation_errors_deg(np.eye(3), rotations), [0.0, 90.0])

    memory = _index(
        tmp_path,
        "memory",
        [_record("train:0", 0.0, 0.0), _record("train:1", 2.0, 0.0)],
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    queries = _index(
        tmp_path,
        "queries",
        [_record("test:0", 0.1, 10.0)],
        np.array([[1.0, 0.0]]),
    )

    metrics, rows = evaluate_pose_retrieval(memory, queries)

    assert metrics["strict"]["coverage"] == 1.0
    assert metrics["strict"]["hit_at_1"]["covered_rate"] == 1.0
    assert rows[0]["top1_translation_error_m"] == 0.1
    assert np.isclose(rows[0]["top1_rotation_error_deg"], 10.0)


def test_text_zone_metrics_use_frozen_assignments(tmp_path: Path) -> None:
    memory = _index(
        tmp_path,
        "memory",
        [_record("train:0", 0.0, 0.0), _record("train:1", 1.0, 0.0)],
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    zone = {
        "slug": "monitor-desk",
        "prompts": {
            "name": "monitor desk",
            "landmarks": "a desk with several monitors",
            "technician_question": "Where is the desk with several monitors?",
        },
    }
    artifact = tmp_path / "zones.json"
    artifact.write_text(
        json.dumps({"zones": [zone], "assignments": {"train:0": "monitor-desk", "train:1": "unassigned"}})
    )

    metrics, rows = evaluate_text_zones(memory, artifact, TextEncoder())

    assert metrics["prompt_count"] == 3
    assert metrics["assignment_coverage"] == 0.5
    assert metrics["macro_hit_at_1"] == 1.0
    assert all(row["precision_at_1"] == 1.0 for row in rows)
