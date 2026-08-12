"""Build RGB-D evidence records for frozen ETH Office detections.

The ETH bags provide registered RGB-coloured point clouds rather than a plain
depth image. We therefore use the point cloud's RGB field to associate points
with a segmentation mask, then keep the world-frame points supplied by the
dataset. This is deliberately labelled as approximate evidence, not an object
reconstruction or identity tracker.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def robust_extent(points: np.ndarray) -> tuple[list[float], list[float]]:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must be a non-empty N x 3 array")
    return (
        np.percentile(points, 5, axis=0).astype(float).tolist(),
        np.percentile(points, 95, axis=0).astype(float).tolist(),
    )


def _cloud_points(message: Any) -> tuple[np.ndarray, np.ndarray]:
    if message.point_step < 20 or message.width < 1:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    data = bytes(message.data)
    points: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    for index in range(int(message.width) * max(int(message.height), 1)):
        offset = index * int(message.point_step)
        x, y, z = struct.unpack_from("<fff", data, offset)
        packed = struct.unpack_from("<I", data, offset + 16)[0]
        if not np.isfinite([x, y, z]).all():
            continue
        points.append((x, y, z))
        colors.append(((packed >> 16) & 255, (packed >> 8) & 255, packed & 255))
    return np.asarray(points, dtype=np.float64), np.asarray(colors, dtype=np.uint8)


def _mask_color_points(image: Image.Image, mask_path: Path, points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=bool)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("mask and RGB image dimensions do not match")
    pixels = rgb[mask]
    if len(pixels) == 0 or len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    # The bag stores RGB-coloured registered points. Quantized colour bins make
    # the association deterministic and robust to JPEG rounding.
    sampled = pixels[:: max(1, len(pixels) // 2048)]
    sampled = sampled.astype(np.uint32, copy=False)
    color_values = colors.astype(np.uint32, copy=False)
    sampled_q = np.minimum((sampled + 8) // 16, 15)
    color_q = np.minimum((color_values + 8) // 16, 15)
    sampled_bins = sampled_q[:, 0] * 256 + sampled_q[:, 1] * 16 + sampled_q[:, 2]
    color_bins = color_q[:, 0] * 256 + color_q[:, 1] * 16 + color_q[:, 2]
    keep = np.isin(color_bins, np.unique(sampled_bins))
    return points[keep]


def build_rgbd_evidence(*, dataset_root: Path, localization: Path, output: Path) -> dict[str, object]:
    root = dataset_root.resolve()
    localization = localization.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output path is not empty: {output.resolve()}")
    output.mkdir(parents=True, exist_ok=True)
    frames = _jsonl(localization / "frames.jsonl")
    detections = _jsonl(localization / "detections.jsonl")
    by_frame: dict[str, list[dict[str, object]]] = {}
    for detection in detections:
        by_frame.setdefault(str(detection["frame_id"]), []).append(detection)
    records: list[dict[str, object]] = []
    for observation in range(4):
        bag = root / "rosbag" / f"observation_{observation}.bag"
        selected = {int(frame["message_index"]): frame for frame in frames if int(frame["observation"]) == observation}
        with AnyReader([bag]) as reader:
            connections = [c for c in reader.connections if c.topic in {"color_image", "point_cloud_G"}]
            images: dict[int, Image.Image] = {}
            clouds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            indices = {0: 0, 1: 0}
            for connection, _, raw in reader.messages(connections=connections):
                topic_kind = 0 if connection.topic == "color_image" else 1
                index = indices[topic_kind]
                indices[topic_kind] += 1
                if index not in selected:
                    continue
                message = reader.deserialize(raw, connection.msgtype)
                if topic_kind == 0:
                    width, height, step = int(message.width), int(message.height), int(message.step)
                    pixels = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(height, step)[:, : width * 3].reshape(height, width, 3)
                    if str(message.encoding).lower() == "bgr8":
                        pixels = pixels[:, :, ::-1]
                    images[index] = Image.fromarray(np.ascontiguousarray(pixels), mode="RGB")
                else:
                    clouds[index] = _cloud_points(message)
        for frame in [item for item in frames if int(item["observation"]) == observation]:
            index = int(frame["message_index"])
            if index not in images or index not in clouds:
                continue
            points, colors = clouds[index]
            image = images[index]
            for detection in by_frame.get(str(frame["frame_id"]), []):
                mask_path = localization / str(detection["mask_path"])
                selected_points = _mask_color_points(image, mask_path, points, colors)
                if len(selected_points):
                    minimum, maximum = robust_extent(selected_points)
                    centroid = np.median(selected_points, axis=0).astype(float).tolist()
                else:
                    minimum = maximum = [None, None, None]
                    centroid = [None, None, None]
                records.append({
                    "detection_id": detection["detection_id"],
                    "frame_id": frame["frame_id"],
                    "observation": observation,
                    "message_index": index,
                    "canonical_class": detection["canonical_class"],
                    "point_count": int(len(selected_points)),
                    "centroid_world_m": centroid,
                    "extent_world_m": {"minimum": minimum, "maximum": maximum},
                    "evidence_method": "registered_rgb_coloured_point_cloud",
                    "claim_boundary": "Approximate visible geometry; no persistent object identity or movement claim.",
                })
    summary = {
        "phase": "6.1.2",
        "dataset": "ETH ASL Change Detection: Office",
        "frame_count": len(frames),
        "detection_count": len(detections),
        "evidence_count": len(records),
        "nonempty_evidence_count": sum(int(item["point_count"]) > 0 for item in records),
        "claim_boundary": "Recorded RGB-coloured point clouds provide approximate visible geometry. This phase does not establish object identity or movement.",
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "evidence.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")
    return summary
