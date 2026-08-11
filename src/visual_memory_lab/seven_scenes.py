"""Prepare the public 7-Scenes office RGB sequences as memory manifests."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

TRAIN_SEQUENCES = (1, 3, 4, 5, 8, 10)
TEST_SEQUENCES = (2, 6, 7, 9)
DATASET_ID = "7-scenes-office"
MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PreparationSummary:
    output: Path
    train_count: int
    test_count: int
    depth_count: int


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def load_pose(path: Path) -> np.ndarray:
    """Load and validate one camera-to-world 4x4 pose matrix."""

    try:
        pose = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read pose matrix {path}: {error}") from error
    if pose.shape != (4, 4):
        raise ValueError(f"expected a 4x4 pose matrix at {path}, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"pose matrix contains non-finite values: {path}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        raise ValueError(f"pose matrix has an invalid homogeneous row: {path}")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=2e-3
    ):
        raise ValueError(f"pose matrix has an invalid rotation: {path}")
    return pose


def _sequence_records(
    dataset_root: Path,
    sequence_number: int,
    split: str,
    *,
    expected_frames: int,
) -> tuple[list[dict[str, object]], tuple[int, int], int]:
    sequence_id = f"seq-{sequence_number:02d}"
    sequence_root = dataset_root / sequence_id
    if not sequence_root.is_dir():
        raise ValueError(f"missing 7-Scenes sequence directory: {sequence_root}")

    records: list[dict[str, object]] = []
    expected_size: tuple[int, int] | None = None
    color_count = len(list(sequence_root.glob("frame-*.color.png")))
    pose_count = len(list(sequence_root.glob("frame-*.pose.txt")))
    depth_count = len(list(sequence_root.glob("frame-*.depth.png")))
    if color_count != expected_frames or pose_count != expected_frames:
        raise ValueError(
            f"{sequence_id} must contain exactly {expected_frames} RGB and pose files; "
            f"found {color_count} RGB and {pose_count} pose files"
        )
    for frame in range(expected_frames):
        stem = f"frame-{frame:06d}"
        image_path = sequence_root / f"{stem}.color.png"
        pose_path = sequence_root / f"{stem}.pose.txt"
        if not image_path.is_file():
            raise ValueError(f"missing RGB frame: {image_path}")
        if not pose_path.is_file():
            raise ValueError(f"missing pose file: {pose_path}")
        try:
            with Image.open(image_path) as image:
                image.load()
                if image.mode != "RGB":
                    raise ValueError(f"expected RGB image at {image_path}, got {image.mode}")
                if expected_size is None:
                    expected_size = image.size
                elif image.size != expected_size:
                    raise ValueError(
                        f"inconsistent image size at {image_path}: {image.size} != {expected_size}"
                    )
        except OSError as error:
            raise ValueError(f"could not read RGB frame {image_path}: {error}") from error

        pose = load_pose(pose_path)
        records.append(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "observation_id": f"office:{sequence_id}:{frame:06d}",
                "episode_id": sequence_id,
                "sequence_id": sequence_id,
                "step": frame,
                "split": split,
                "image_path": image_path.relative_to(dataset_root).as_posix(),
                "camera_pose": {
                    "convention": "camera_to_world",
                    "matrix": pose.tolist(),
                    "translation_m": pose[:3, 3].tolist(),
                },
            }
        )
    assert expected_size is not None
    return records, expected_size, depth_count


def prepare_office_dataset(
    *,
    dataset_root: Path,
    output: Path,
    expected_frames: int = 1000,
) -> PreparationSummary:
    """Validate Office RGB/pose data and write official train/test manifests."""

    if expected_frames < 1:
        raise ValueError("expected_frames must be at least 1")
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"7-Scenes office directory does not exist: {dataset_root}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output path is not empty: {output}")

    split_records: dict[str, list[dict[str, object]]] = {"train": [], "test": []}
    image_size: tuple[int, int] | None = None
    depth_count = 0
    for split, sequence_numbers in (("train", TRAIN_SEQUENCES), ("test", TEST_SEQUENCES)):
        for sequence_number in sequence_numbers:
            records, sequence_size, sequence_depth_count = _sequence_records(
                dataset_root,
                sequence_number,
                split,
                expected_frames=expected_frames,
            )
            if image_size is None:
                image_size = sequence_size
            elif sequence_size != image_size:
                raise ValueError(
                    f"inconsistent sequence image size: {sequence_size} != {image_size}"
                )
            split_records[split].extend(records)
            depth_count += sequence_depth_count
    assert image_size is not None

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for split, records in split_records.items():
            split_root = temporary / split
            split_root.mkdir()
            image_root = Path(os.path.relpath(dataset_root, split_root)).as_posix()
            _write_json(
                split_root / "run.json",
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "dataset_id": DATASET_ID,
                    "split": split,
                    "sequences": [
                        f"seq-{value:02d}"
                        for value in (TRAIN_SEQUENCES if split == "train" else TEST_SEQUENCES)
                    ],
                    "observation_count": len(records),
                    "image_root": image_root,
                    "image": {
                        "width": image_size[0],
                        "height": image_size[1],
                        "mode": "RGB",
                    },
                    "pose": {
                        "convention": "camera_to_world",
                        "matrix_shape": [4, 4],
                        "translation_unit": "metre",
                    },
                },
            )
            _write_jsonl(split_root / "observations.jsonl", records)
        _write_json(
            temporary / "summary.json",
            {
                "dataset_id": DATASET_ID,
                "dataset_root": str(dataset_root),
                "train_count": len(split_records["train"]),
                "test_count": len(split_records["test"]),
                "depth_files_present": depth_count,
                "depth_used": False,
            },
        )
        if output.exists():
            output.rmdir()
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return PreparationSummary(
        output=output,
        train_count=len(split_records["train"]),
        test_count=len(split_records["test"]),
        depth_count=depth_count,
    )
