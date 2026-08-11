"""Preparation and visual audit for the ETH change-detection Office dataset."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from rosbags.highlevel import AnyReader


def _require_empty_directory(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError(f"output path is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_indices(count: int, sample_count: int) -> list[int]:
    """Return deterministic, unique indices spanning the full sequence."""

    if count < 1 or sample_count < 1:
        raise ValueError("count and sample_count must be positive")
    actual = min(count, sample_count)
    return np.rint(np.linspace(0, count - 1, actual)).astype(int).tolist()


def _decode_image(message: Any) -> Image.Image:
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(height, step)
    if encoding in {"rgb8", "bgr8"}:
        pixels = raw[:, : width * 3].reshape(height, width, 3)
        if encoding == "bgr8":
            pixels = pixels[:, :, ::-1]
        return Image.fromarray(np.ascontiguousarray(pixels), mode="RGB")
    if encoding in {"mono8", "8uc1"}:
        return Image.fromarray(raw[:, :width].copy(), mode="L").convert("RGB")
    raise ValueError(f"unsupported color_image encoding: {message.encoding}")


def _contact_sheet(paths: list[Path], output: Path, *, columns: int = 4) -> None:
    if not paths:
        raise ValueError("cannot create an empty contact sheet")
    thumbnails: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((360, 220))
            thumbnails.append(thumb.copy())
    cell_w = max(image.width for image in thumbnails)
    cell_h = max(image.height for image in thumbnails) + 28
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (path, image) in enumerate(zip(paths, thumbnails, strict=True)):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + image.height + 4), path.stem, fill="black")
    sheet.save(output, format="JPEG", quality=90)


def _mesh_summary(path: Path) -> dict[str, object]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue", "normal_x", "normal_y", "normal_z"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"mesh {path.name} is missing properties: {', '.join(missing)}")
    xyz = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(np.float64)
    face_count = int(ply["face"].count) if "face" in ply else 0
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "vertex_count": int(len(vertex)),
        "face_count": face_count,
        "bounds_m": {"minimum": xyz.min(axis=0).tolist(), "maximum": xyz.max(axis=0).tolist()},
    }


def _inspect_bag(
    path: Path,
    frame_dir: Path,
    *,
    rgb_samples: int,
    vlm_samples: int,
) -> dict[str, object]:
    with AnyReader([path]) as reader:
        topics = {
            name: {"message_type": info.msgtype, "message_count": info.msgcount}
            for name, info in sorted(reader.topics.items())
        }
        image_connections = [connection for connection in reader.connections if connection.topic == "color_image"]
        if not image_connections:
            raise ValueError(f"bag has no color_image topic: {path}")
        image_count = sum(connection.msgcount for connection in image_connections)
        selected = evenly_spaced_indices(image_count, rgb_samples)
        selected_set = set(selected)
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames: list[dict[str, object]] = []
        image_index = 0
        for connection, timestamp, rawdata in reader.messages(connections=image_connections):
            if image_index in selected_set:
                message = reader.deserialize(rawdata, connection.msgtype)
                image = _decode_image(message)
                frame_path = frame_dir / f"frame-{image_index:06d}.jpg"
                image.save(frame_path, format="JPEG", quality=92)
                frames.append(
                    {
                        "message_index": image_index,
                        "timestamp_ns": int(timestamp),
                        "path": str(frame_path.resolve()),
                        "width": image.width,
                        "height": image.height,
                        "encoding": str(message.encoding),
                    }
                )
            image_index += 1
        if len(frames) != len(selected):
            raise ValueError(f"expected {len(selected)} sampled frames from {path.name}, read {len(frames)}")

        human_paths = [Path(str(item["path"])) for item in frames]
        human_sheet = frame_dir.parent / "contact-sheet.jpg"
        _contact_sheet(human_paths, human_sheet, columns=4)
        vlm_positions = evenly_spaced_indices(len(frames), min(vlm_samples, len(frames)))
        vlm_paths = [human_paths[index] for index in vlm_positions]
        vlm_sheet = frame_dir.parent / "vlm-contact-sheet.jpg"
        _contact_sheet(vlm_paths, vlm_sheet, columns=4)

        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "start_time_ns": int(reader.start_time),
            "end_time_ns": int(reader.end_time),
            "duration_ns": int(reader.duration),
            "message_count": int(reader.message_count),
            "topics": topics,
            "rgb_frames": frames,
            "contact_sheet": str(human_sheet.resolve()),
            "vlm_contact_sheet": str(vlm_sheet.resolve()),
            "vlm_message_indices": [int(frames[index]["message_index"]) for index in vlm_positions],
        }


def _write_gallery(manifest: dict[str, object], output: Path) -> None:
    sections: list[str] = []
    for observation in manifest["observations"]:  # type: ignore[index]
        assert isinstance(observation, dict)
        bag = observation["bag"]
        assert isinstance(bag, dict)
        cards = []
        for frame in bag["rgb_frames"]:  # type: ignore[index]
            assert isinstance(frame, dict)
            relative = Path(str(frame["path"])).relative_to(output).as_posix()
            cards.append(
                f'<figure><a href="{html.escape(relative)}"><img loading="lazy" src="{html.escape(relative)}"></a>'
                f'<figcaption>Frame {int(frame["message_index"]):06d}</figcaption></figure>'
            )
        sheet = Path(str(bag["contact_sheet"])).relative_to(output).as_posix()
        vlm_sheet = Path(str(bag["vlm_contact_sheet"])).relative_to(output).as_posix()
        sections.append(
            f'<section><h2>{html.escape(str(observation["observation_id"]))}</h2>'
            f'<p><a href="{sheet}">Human contact sheet</a> · '
            f'<a href="{vlm_sheet}">8-frame VLM contact sheet</a></p>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ETH Office visual audit</title><style>
body{{font:16px/1.5 system-ui;margin:0 auto;max-width:1500px;padding:32px;background:#f5f2e9;color:#20231f}}
h1,h2{{font-family:Georgia,serif}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
figure{{margin:0;background:white;border:1px solid #d8d2c4;border-radius:10px;overflow:hidden}}
img{{display:block;width:100%;height:180px;object-fit:cover}} figcaption{{padding:8px 12px}}
</style></head><body><h1>ETH Office visual audit</h1>
<p>Twenty-four evenly spaced RGB frames per observation. The eight-frame VLM subset is kept separate.</p>
{"".join(sections)}</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def prepare_eth_office(
    *,
    dataset_root: Path,
    output: Path,
    rgb_samples: int = 24,
    vlm_samples: int = 8,
) -> dict[str, object]:
    """Validate ETH Office and produce a browsable, reproducible visual audit."""

    if rgb_samples < 1 or vlm_samples < 1:
        raise ValueError("RGB sample counts must be positive")
    if vlm_samples > rgb_samples:
        raise ValueError("VLM samples cannot exceed human RGB samples")
    root = dataset_root.resolve()
    mesh_dir = root / "complete_mesh"
    bag_dir = root / "rosbag"
    if not mesh_dir.is_dir() or not bag_dir.is_dir():
        raise ValueError("ETH Office root must contain complete_mesh and rosbag directories")
    mesh_paths = sorted(mesh_dir.glob("observation_*.ply"))
    bag_paths = sorted(bag_dir.glob("observation_*.bag"))
    expected = [f"observation_{index}" for index in range(4)]
    if [path.stem for path in mesh_paths] != expected or [path.stem for path in bag_paths] != expected:
        raise ValueError("ETH Office requires observation_0 through observation_3 meshes and bags")

    resolved_output = _require_empty_directory(output)
    observations: list[dict[str, object]] = []
    for index, (mesh_path, bag_path) in enumerate(zip(mesh_paths, bag_paths, strict=True)):
        observation_output = resolved_output / f"observation-{index}"
        observation_output.mkdir()
        observations.append(
            {
                "observation_id": f"eth-office:{index}",
                "logical_order": index,
                "mesh": _mesh_summary(mesh_path),
                "bag": _inspect_bag(
                    bag_path,
                    observation_output / "frames",
                    rgb_samples=rgb_samples,
                    vlm_samples=vlm_samples,
                ),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": "ETH ASL Change Detection: Office",
        "dataset_root": str(root),
        "logical_order_note": "Observation indices are logical order, not calendar timestamps.",
        "rgb_samples_per_observation": rgb_samples,
        "vlm_samples_per_observation": vlm_samples,
        "observations": observations,
    }
    (resolved_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_gallery(manifest, resolved_output)
    return manifest
