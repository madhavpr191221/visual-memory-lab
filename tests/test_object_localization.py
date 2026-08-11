"""Tests for the frozen Phase 6B1 object-localization pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import visual_memory_lab.object_localization as localization


class FakeDetector:
    provenance = {"model_id": "fake-detector", "device": "cpu", "revision": "test"}

    def detect(self, image: Image.Image) -> list[dict[str, object]]:
        del image
        return [
            {"phrase": "office chair", "score": 0.91, "box_xyxy": [2, 2, 12, 10]},
            {"phrase": "desk chair", "score": 0.72, "box_xyxy": [3, 2, 12, 10]},
            {"phrase": "cardboard box", "score": 0.64, "box_xyxy": [15, 3, 22, 12]},
            {"phrase": "lamp", "score": 0.99, "box_xyxy": [0, 0, 4, 4]},
        ]

    def close(self) -> None:
        pass


class FakeSegmenter:
    provenance = {"model_id": "fake-segmenter", "device": "cpu", "revision": "test"}

    def segment(
        self, image: Image.Image, boxes_xyxy: list[list[float]]
    ) -> list[tuple[np.ndarray, float]]:
        predictions = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = (int(value) for value in box)
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            predictions.append((mask, 0.88))
        return predictions

    def close(self) -> None:
        pass


def _fake_frames(*, root: Path, work: Path, keyframes_per_observation: int) -> list[dict[str, object]]:
    del root, keyframes_per_observation
    frames: list[dict[str, object]] = []
    for observation in range(4):
        relative = Path("frames") / f"observation-{observation}" / "frame-000000.jpg"
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 16), (230, 230, 230)).save(path)
        frames.append(
            {
                "frame_id": f"eth-office:{observation}:000000",
                "observation": observation,
                "message_index": 0,
                "timestamp_ns": observation,
                "pose": {
                    "frame": "T_G_C",
                    "translation_m": [float(observation), 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "image_path": relative.as_posix(),
                "width": 24,
                "height": 16,
            }
        )
    return frames


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_normalization_maps_classes_clamps_boxes_and_removes_duplicates() -> None:
    detections, rejected = localization.normalize_detections(
        [
            {"phrase": "office chair", "score": 0.9, "box_xyxy": [-1, 1, 9, 9]},
            {"phrase": "desk chair", "score": 0.7, "box_xyxy": [0, 1, 9, 9]},
            {"phrase": "trash bin", "score": 0.8, "box_xyxy": [10, 2, 15, 8]},
            {"phrase": "plant", "score": 0.99, "box_xyxy": [1, 1, 3, 3]},
        ],
        width=16,
        height=10,
    )
    assert [item["canonical_class"] for item in detections] == ["chair", "waste_bin"]
    assert detections[0]["box_xyxy"] == [0.0, 1.0, 9.0, 9.0]
    assert rejected == 2
    assert localization.box_iou([0, 0, 4, 4], [2, 0, 6, 4]) == 1 / 3


def test_pose_keyframes_are_temporally_spread_and_deterministic() -> None:
    poses = [
        {
            "translation_m": [float(index), 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        for index in range(12)
    ]
    assert localization.select_pose_keyframes(poses, 4) == [1, 5, 8, 11]
    assert localization.select_pose_keyframes(poses, 4) == localization.select_pose_keyframes(poses, 4)


def test_localization_writes_complete_atomic_artifact(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "office"
    dataset.mkdir()
    fake_bags = [dataset / f"observation_{index}.bag" for index in range(4)]
    monkeypatch.setattr(localization, "_observation_paths", lambda root: fake_bags)
    monkeypatch.setattr(localization, "_prepare_frames", _fake_frames)
    output = tmp_path / "object-localization"

    summary = localization.localize_eth_objects(
        dataset_root=dataset,
        output=output,
        keyframes_per_observation=1,
        device="cpu",
        detector_factory=FakeDetector,
        segmenter_factory=FakeSegmenter,
    )

    assert summary.frame_count == 4
    assert summary.detection_count == 8
    assert summary.device == "cpu"
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["class_counts"] == {"box": 4, "chair": 4}
    assert run["frames_per_observation"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert "ground-truth" in run["claim_boundary"]
    frames = _records(output / "frames.jsonl")
    detections = _records(output / "detections.jsonl")
    assert all((output / str(item["overlay_path"])).is_file() for item in frames)
    assert all((output / str(item["mask_path"])).is_file() for item in detections)
    assert all(item["mask_area_fraction"] > 0 for item in detections)
    assert not (tmp_path / ".object-localization-work").exists()
