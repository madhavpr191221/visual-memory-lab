"""Tests for the real-image 7-Scenes adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visual_memory_lab.memory import MemoryIndex, build_index
from visual_memory_lab.seven_scenes import prepare_office_dataset


class TinyEncoder:
    model_id = "test/tiny"
    model_revision = "1"
    embedding_dim = 2
    processor_config = {"test": True}

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        values = np.arange(1, len(image_paths) + 1, dtype=np.float32)
        rows = np.stack([np.ones_like(values), values], axis=1)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def make_office_fixture(root: Path, *, frames: int = 2) -> Path:
    office = root / "office"
    for sequence in range(1, 11):
        sequence_root = office / f"seq-{sequence:02d}"
        sequence_root.mkdir(parents=True)
        for frame in range(frames):
            stem = f"frame-{frame:06d}"
            Image.new("RGB", (8, 6), color=(sequence, frame, 20)).save(
                sequence_root / f"{stem}.color.png"
            )
            pose = np.eye(4)
            pose[0, 3] = sequence / 10
            pose[2, 3] = frame / 10
            np.savetxt(sequence_root / f"{stem}.pose.txt", pose)
    return office


def test_prepare_office_writes_official_split_without_copying_images(tmp_path: Path) -> None:
    office = make_office_fixture(tmp_path)
    output = tmp_path / "prepared"

    summary = prepare_office_dataset(
        dataset_root=office, output=output, expected_frames=2
    )

    assert summary.train_count == 12
    assert summary.test_count == 8
    train_manifest = json.loads((output / "train" / "run.json").read_text())
    assert train_manifest["sequences"] == [
        "seq-01",
        "seq-03",
        "seq-04",
        "seq-05",
        "seq-08",
        "seq-10",
    ]
    assert train_manifest["image"] == {"height": 6, "mode": "RGB", "width": 8}
    assert not list(output.rglob("*.png"))

    index_root = tmp_path / "index"
    build_index(
        source=output / "train",
        output=index_root,
        encoder=TinyEncoder(),
        batch_size=20,
    )
    index = MemoryIndex.load(index_root)
    result = index.search(index.embeddings[0], top_k=1)[0]
    assert result.image_path.is_file()
    assert result.observation["sequence_id"] == "seq-01"
    assert result.nearby_actions == ()


def test_prepare_office_rejects_missing_pose_and_nonempty_output(tmp_path: Path) -> None:
    office = make_office_fixture(tmp_path)
    missing = office / "seq-04" / "frame-000001.pose.txt"
    missing.unlink()
    with pytest.raises(ValueError, match="exactly 2 RGB and pose files"):
        prepare_office_dataset(
            dataset_root=office,
            output=tmp_path / "prepared",
            expected_frames=2,
        )

    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep").write_text("safe")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_office_dataset(
            dataset_root=office,
            output=output,
            expected_frames=2,
        )
