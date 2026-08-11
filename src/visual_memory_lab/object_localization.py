"""Frozen RGB object-localization baseline for the ETH Office observations."""

from __future__ import annotations

import gc
import json
import math
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rosbags.highlevel import AnyReader

DETECTOR_MODEL = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_MODEL = "facebook/sam2.1-hiera-small"
TEXT_PROMPT = (
    "office chair. desk chair. waste bin. trash bin. wastebasket. "
    "cardboard box. storage box."
)
CLASS_COLORS = {
    "chair": (31, 111, 235),
    "waste_bin": (14, 145, 92),
    "box": (224, 124, 30),
}


class Detector(Protocol):
    provenance: dict[str, object]

    def detect(self, image: Image.Image) -> list[dict[str, object]]: ...

    def close(self) -> None: ...


class Segmenter(Protocol):
    provenance: dict[str, object]

    def segment(
        self, image: Image.Image, boxes_xyxy: Sequence[Sequence[float]]
    ) -> list[tuple[np.ndarray, float]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LocalizationSummary:
    output: Path
    frame_count: int
    detection_count: int
    device: str


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch build cannot use it; "
            "sync the project with the cuda extra"
        )
    return requested


def _model_revision(model: object) -> str | None:
    config = getattr(model, "config", None)
    value = getattr(config, "_commit_hash", None)
    return str(value) if value else None


class HuggingFaceGroundingDino:
    """Grounding DINO adapter with a stable, small output contract."""

    def __init__(
        self,
        *,
        model_id: str = DETECTOR_MODEL,
        device: str = "auto",
        prompt: str = TEXT_PROMPT,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = _resolve_device(device)
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device).eval()
        self._torch = torch
        self.provenance = {
            "model_id": model_id,
            "revision": _model_revision(self.model),
            "framework": "transformers",
            "device": self.device,
            "dtype": "bfloat16-autocast" if self.device == "cuda" else "float32",
        }

    def detect(self, image: Image.Image) -> list[dict[str, object]]:
        torch = self._torch
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(
            self.device
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else _NullContext()
        )
        with torch.inference_mode(), autocast:
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        # Current Transformers exposes the grounded phrases as ``text_labels``.
        # ``labels`` is retained only as a compatibility fallback for older
        # processor versions and may now contain integer token IDs.
        labels = result.get("text_labels", result["labels"])
        # Transformers 5.15 currently returns ``[""]`` for labels when no box
        # clears the thresholds. Treat that representation as an empty result.
        if not boxes:
            return []
        if len(labels) != len(boxes):
            raise ValueError(
                "Grounding DINO returned inconsistent box and phrase counts: "
                f"{len(boxes)} boxes and {len(labels)} phrases"
            )
        return [
            {
                "phrase": str(label),
                "score": float(score),
                "box_xyxy": [float(value) for value in box],
            }
            for box, score, label in zip(boxes, scores, labels, strict=True)
        ]

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


class HuggingFaceSam2:
    """SAM 2 image adapter using Grounding DINO boxes as prompts."""

    def __init__(
        self,
        *,
        model_id: str = SEGMENTER_MODEL,
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.device = _resolve_device(device)
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()
        self._torch = torch
        self.provenance = {
            "model_id": model_id,
            "revision": _model_revision(self.model),
            "framework": "transformers",
            "device": self.device,
            "dtype": "bfloat16-autocast" if self.device == "cuda" else "float32",
        }

    def segment(
        self, image: Image.Image, boxes_xyxy: Sequence[Sequence[float]]
    ) -> list[tuple[np.ndarray, float]]:
        if not boxes_xyxy:
            return []
        torch = self._torch
        boxes = [[list(map(float, box)) for box in boxes_xyxy]]
        inputs = self.processor(
            images=image,
            input_boxes=boxes,
            return_tensors="pt",
        ).to(self.device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else _NullContext()
        )
        with torch.inference_mode(), autocast:
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu()
        )[0]
        scores = outputs.iou_scores.detach().float().cpu()[0]
        results: list[tuple[np.ndarray, float]] = []
        for index in range(len(boxes_xyxy)):
            mask = np.asarray(masks[index]).squeeze().astype(bool)
            score = float(np.asarray(scores[index]).reshape(-1)[0])
            results.append((mask, score))
        return results

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


def canonical_class(phrase: str) -> str | None:
    normalized = phrase.strip().lower().replace("_", " ")
    if "chair" in normalized:
        return "chair"
    if any(value in normalized for value in ("waste bin", "trash bin", "wastebasket", "bin")):
        return "waste_bin"
    if "box" in normalized:
        return "box"
    return None


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _clamp_box(box: Sequence[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = map(float, box)
    values = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def normalize_detections(
    raw: Sequence[dict[str, object]],
    *,
    width: int,
    height: int,
    nms_iou: float = 0.50,
    max_detections: int = 20,
) -> tuple[list[dict[str, object]], int]:
    candidates: list[dict[str, object]] = []
    rejected = 0
    for item in raw:
        canonical = canonical_class(str(item.get("phrase", "")))
        box = item.get("box_xyxy")
        if canonical is None or not isinstance(box, Sequence):
            rejected += 1
            continue
        clamped = _clamp_box(box, width, height)
        if clamped is None:
            rejected += 1
            continue
        candidates.append(
            {
                "canonical_class": canonical,
                "phrase": str(item["phrase"]),
                "score": float(item["score"]),
                "box_xyxy": clamped,
            }
        )
    kept: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: -float(item["score"])):
        duplicate = any(
            candidate["canonical_class"] == item["canonical_class"]
            and box_iou(candidate["box_xyxy"], item["box_xyxy"]) > nms_iou  # type: ignore[arg-type]
            for item in kept
        )
        if duplicate:
            rejected += 1
        else:
            kept.append(candidate)
        if len(kept) == max_detections:
            rejected += max(0, len(candidates) - len(kept) - rejected)
            break
    return kept, rejected


def quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


def select_pose_keyframes(poses: Sequence[dict[str, object]], count: int) -> list[int]:
    """Select one pose-diverse frame from each of ``count`` temporal windows."""

    if count < 1:
        raise ValueError("keyframe count must be positive")
    if not poses:
        raise ValueError("pose sequence cannot be empty")
    if len(poses) <= count:
        return list(range(len(poses)))
    boundaries = np.linspace(0, len(poses), count + 1).astype(int)
    selected: list[int] = []
    for window in range(count):
        start, stop = int(boundaries[window]), int(boundaries[window + 1])
        candidates = list(range(start, max(start + 1, stop)))
        if not selected:
            selected.append(candidates[len(candidates) // 2])
            continue
        previous = poses[selected[-1]]
        previous_t = np.asarray(previous["translation_m"], dtype=np.float64)
        previous_q = previous["quaternion_xyzw"]

        def distance(index: int) -> tuple[float, int]:
            pose = poses[index]
            translation = np.asarray(pose["translation_m"], dtype=np.float64)
            translation_term = float(np.linalg.norm(translation - previous_t)) / 0.10
            rotation_term = quaternion_angle(previous_q, pose["quaternion_xyzw"]) / math.radians(10)
            return max(translation_term, rotation_term), -index

        selected.append(max(candidates, key=distance))
    return selected


def _read_poses(path: Path) -> list[dict[str, object]]:
    with AnyReader([path]) as reader:
        connections = [item for item in reader.connections if item.topic == "T_G_C"]
        if not connections:
            raise ValueError(f"bag has no T_G_C topic: {path}")
        poses: list[dict[str, object]] = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            translation = message.transform.translation
            rotation = message.transform.rotation
            poses.append(
                {
                    "timestamp_ns": int(timestamp),
                    "translation_m": [float(translation.x), float(translation.y), float(translation.z)],
                    "quaternion_xyzw": [
                        float(rotation.x),
                        float(rotation.y),
                        float(rotation.z),
                        float(rotation.w),
                    ],
                }
            )
    return poses


def _decode_image(message: Any) -> Image.Image:
    width, height, step = int(message.width), int(message.height), int(message.step)
    encoding = str(message.encoding).lower()
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(height, step)
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported color_image encoding: {message.encoding}")
    pixels = raw[:, : width * 3].reshape(height, width, 3)
    if encoding == "bgr8":
        pixels = pixels[:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(pixels), mode="RGB")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )


def _mask_overlay(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[mask, :3] = color
    rgba[mask, 3] = 112
    return Image.fromarray(rgba, mode="RGBA")


def _render_overlay(
    image_path: Path,
    detections: Sequence[dict[str, object]],
    work: Path,
    output: Path,
) -> None:
    with Image.open(image_path) as source:
        base = source.convert("RGBA")
    for detection in detections:
        mask_path = work / str(detection["mask_path"])
        with Image.open(mask_path) as mask_image:
            base.alpha_composite(mask_image.convert("RGBA"))
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default()
    for detection in detections:
        color = CLASS_COLORS[str(detection["canonical_class"])]
        box = [float(item) for item in detection["box_xyxy"]]  # type: ignore[union-attr]
        draw.rectangle(box, outline=color + (255,), width=4)
        label = f"{str(detection['canonical_class']).replace('_', ' ')} {float(detection['score']):.2f}"
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_box, fill=color + (235,))
        draw.text((box[0], box[1]), label, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(output, format="JPEG", quality=92)


def _observation_paths(root: Path) -> list[Path]:
    bag_dir = root.resolve() / "rosbag"
    paths = sorted(bag_dir.glob("observation_*.bag"))
    expected = [f"observation_{index}" for index in range(4)]
    if [path.stem for path in paths] != expected:
        raise ValueError("ETH Office requires observation_0 through observation_3 bags")
    return paths


def _prepare_frames(
    *, root: Path, work: Path, keyframes_per_observation: int
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for observation_index, bag_path in enumerate(_observation_paths(root)):
        poses = _read_poses(bag_path)
        selected = select_pose_keyframes(poses, keyframes_per_observation)
        selected_set = set(selected)
        image_dir = work / "frames" / f"observation-{observation_index}"
        image_dir.mkdir(parents=True, exist_ok=True)
        with AnyReader([bag_path]) as reader:
            connections = [item for item in reader.connections if item.topic == "color_image"]
            if not connections:
                raise ValueError(f"bag has no color_image topic: {bag_path}")
            image_index = 0
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                if image_index in selected_set:
                    pose = poses[image_index]
                    if int(pose["timestamp_ns"]) != int(timestamp):
                        raise ValueError(
                            f"RGB and T_G_C timestamps differ at {bag_path.name}:{image_index}"
                        )
                    relative = Path("frames") / f"observation-{observation_index}" / f"frame-{image_index:06d}.jpg"
                    image_path = work / relative
                    if not image_path.is_file():
                        image = _decode_image(reader.deserialize(rawdata, connection.msgtype))
                        image.save(image_path, format="JPEG", quality=94)
                    with Image.open(image_path) as image:
                        width, height = image.size
                    frame_id = f"eth-office:{observation_index}:{image_index:06d}"
                    records.append(
                        {
                            "frame_id": frame_id,
                            "observation": observation_index,
                            "message_index": image_index,
                            "timestamp_ns": int(timestamp),
                            "pose": {
                                "frame": "T_G_C",
                                "translation_m": pose["translation_m"],
                                "quaternion_xyzw": pose["quaternion_xyzw"],
                            },
                            "image_path": relative.as_posix(),
                            "width": width,
                            "height": height,
                        }
                    )
                image_index += 1
        if image_index != len(poses):
            raise ValueError(f"RGB and T_G_C message counts differ in {bag_path.name}")
    records.sort(key=lambda item: (int(item["observation"]), int(item["message_index"])))
    return records


def _record_path(work: Path, frame: dict[str, object]) -> Path:
    return work / "records" / f"observation-{int(frame['observation'])}-frame-{int(frame['message_index']):06d}.json"


def localize_eth_objects(
    *,
    dataset_root: Path,
    output: Path,
    keyframes_per_observation: int = 96,
    device: str = "auto",
    detector_model: str = DETECTOR_MODEL,
    segmenter_model: str = SEGMENTER_MODEL,
    prompt: str = TEXT_PROMPT,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    nms_iou: float = 0.50,
    max_detections: int = 20,
    detector_factory: Callable[[], Detector] | None = None,
    segmenter_factory: Callable[[], Segmenter] | None = None,
) -> LocalizationSummary:
    """Create a resumable, then atomically finalized, localization artifact."""

    if keyframes_per_observation < 1:
        raise ValueError("keyframes per observation must be positive")
    if not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1:
        raise ValueError("detection thresholds must lie in [0, 1]")
    if not 0 <= nms_iou <= 1:
        raise ValueError("NMS IoU must lie in [0, 1]")
    root = dataset_root.resolve()
    _observation_paths(root)
    resolved = output.resolve()
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError(f"output path is not empty: {resolved}")
    if resolved.is_dir():
        resolved.rmdir()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    work = resolved.parent / f".{resolved.name}-work"
    config = {
        "schema_version": 1,
        "dataset_root": str(root),
        "keyframes_per_observation": keyframes_per_observation,
        "detector_model": detector_model,
        "segmenter_model": segmenter_model,
        "prompt": prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "nms_iou": nms_iou,
        "max_detections": max_detections,
    }
    config_path = work / "config.json"
    if config_path.is_file() and _read_json(config_path) != config:
        raise ValueError(f"resumable work directory has a different configuration: {work}")
    work.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, config)

    frames_path = work / "keyframes.json"
    if frames_path.is_file():
        frames = _read_json(frames_path).get("frames")
        if not isinstance(frames, list):
            raise ValueError("keyframe cache is malformed")
    else:
        frames = _prepare_frames(
            root=root,
            work=work,
            keyframes_per_observation=keyframes_per_observation,
        )
        _write_json(frames_path, {"frames": frames})

    missing_detection = [
        frame
        for frame in frames
        if not _record_path(work, frame).is_file()
        or _read_json(_record_path(work, frame)).get("stage") not in {"detected", "segmented"}
    ]
    detector_provenance: dict[str, object] = {
        "model_id": detector_model,
        "device": _resolve_device(device),
    }
    detector_provenance_path = work / "detector-provenance.json"
    if detector_provenance_path.is_file():
        detector_provenance = _read_json(detector_provenance_path)
    if missing_detection:
        factory = detector_factory or (
            lambda: HuggingFaceGroundingDino(
                model_id=detector_model,
                device=device,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        )
        detector = factory()
        detector_provenance = detector.provenance
        _write_json(detector_provenance_path, detector_provenance)
        try:
            for frame in missing_detection:
                started = time.perf_counter()
                image_path = work / str(frame["image_path"])
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                raw = detector.detect(image)
                detections, rejected = normalize_detections(
                    raw,
                    width=int(frame["width"]),
                    height=int(frame["height"]),
                    nms_iou=nms_iou,
                    max_detections=max_detections,
                )
                for index, detection in enumerate(detections):
                    detection["detection_id"] = f"{frame['frame_id']}:det-{index:02d}"
                    box = detection["box_xyxy"]
                    detection["box_normalized"] = [
                        float(box[0]) / int(frame["width"]),
                        float(box[1]) / int(frame["height"]),
                        float(box[2]) / int(frame["width"]),
                        float(box[3]) / int(frame["height"]),
                    ]
                _write_json(
                    _record_path(work, frame),
                    {
                        "stage": "detected",
                        "frame": frame,
                        "detections": detections,
                        "rejected_detection_count": rejected,
                        "detection_seconds": time.perf_counter() - started,
                    },
                )
        finally:
            detector.close()

    incomplete_segmentation = [
        frame
        for frame in frames
        if _read_json(_record_path(work, frame)).get("stage") != "segmented"
    ]
    segmenter_provenance: dict[str, object] = {
        "model_id": segmenter_model,
        "device": _resolve_device(device),
    }
    segmenter_provenance_path = work / "segmenter-provenance.json"
    if segmenter_provenance_path.is_file():
        segmenter_provenance = _read_json(segmenter_provenance_path)
    if incomplete_segmentation:
        factory = segmenter_factory or (
            lambda: HuggingFaceSam2(model_id=segmenter_model, device=device)
        )
        segmenter = factory()
        segmenter_provenance = segmenter.provenance
        _write_json(segmenter_provenance_path, segmenter_provenance)
        try:
            for frame in incomplete_segmentation:
                record_path = _record_path(work, frame)
                record = _read_json(record_path)
                detections = record["detections"]
                assert isinstance(detections, list)
                image_path = work / str(frame["image_path"])
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                started = time.perf_counter()
                predictions = segmenter.segment(
                    image,
                    [item["box_xyxy"] for item in detections],
                )
                if len(predictions) != len(detections):
                    raise ValueError("segmenter returned a different number of masks than boxes")
                for detection, (mask, mask_score) in zip(detections, predictions, strict=True):
                    if mask.shape != (image.height, image.width):
                        raise ValueError(
                            f"mask shape {mask.shape} does not match image {(image.height, image.width)}"
                        )
                    slug = str(detection["detection_id"]).replace(":", "-")
                    relative = Path("masks") / f"{slug}.png"
                    color = CLASS_COLORS[str(detection["canonical_class"])]
                    (work / relative).parent.mkdir(parents=True, exist_ok=True)
                    _mask_overlay(mask, color).save(work / relative, format="PNG")
                    area_fraction = float(mask.mean())
                    warnings: list[str] = []
                    if area_fraction < 0.001:
                        warnings.append("tiny_mask")
                    if area_fraction > 0.60:
                        warnings.append("large_mask")
                    detection["mask_path"] = relative.as_posix()
                    detection["mask_score"] = mask_score
                    detection["mask_area_fraction"] = area_fraction
                    detection["warnings"] = warnings
                overlay_relative = Path("overlays") / (
                    f"observation-{int(frame['observation'])}-frame-{int(frame['message_index']):06d}.jpg"
                )
                _render_overlay(image_path, detections, work, work / overlay_relative)
                record["stage"] = "segmented"
                record["detections"] = detections
                record["segmentation_seconds"] = time.perf_counter() - started
                record["overlay_path"] = overlay_relative.as_posix()
                _write_json(record_path, record)
        finally:
            segmenter.close()

    frame_records: list[dict[str, object]] = []
    detections: list[dict[str, object]] = []
    rejected_total = 0
    detection_seconds = 0.0
    segmentation_seconds = 0.0
    for frame in frames:
        record = _read_json(_record_path(work, frame))
        if record.get("stage") != "segmented":
            raise ValueError(f"frame did not finish segmentation: {frame['frame_id']}")
        frame_detections = record["detections"]
        assert isinstance(frame_detections, list)
        ids = [str(item["detection_id"]) for item in frame_detections]
        frame_records.append(
            {
                **frame,
                "overlay_path": record["overlay_path"],
                "detection_ids": ids,
                "detection_count": len(ids),
            }
        )
        for item in frame_detections:
            detections.append({**item, "frame_id": frame["frame_id"]})
        rejected_total += int(record.get("rejected_detection_count", 0))
        detection_seconds += float(record.get("detection_seconds", 0.0))
        segmentation_seconds += float(record.get("segmentation_seconds", 0.0))

    class_counts = Counter(str(item["canonical_class"]) for item in detections)
    observation_counts = Counter(int(item["observation"]) for item in frame_records)
    frames_with_detections = sum(bool(item["detection_ids"]) for item in frame_records)
    device_used = str(detector_provenance.get("device", _resolve_device(device)))
    run = {
        **config,
        "python": sys.version.split()[0],
        "detector": detector_provenance,
        "segmenter": segmenter_provenance,
        "frame_count": len(frame_records),
        "frames_per_observation": {
            str(index): observation_counts[index] for index in sorted(observation_counts)
        },
        "detection_count": len(detections),
        "frames_with_detections": frames_with_detections,
        "empty_frame_count": len(frame_records) - frames_with_detections,
        "class_counts": dict(sorted(class_counts.items())),
        "rejected_detection_count": rejected_total,
        "detection_seconds": detection_seconds,
        "segmentation_seconds": segmentation_seconds,
        "claim_boundary": (
            "Detections and masks are frozen-model predictions, not ground-truth annotations. "
            "A missing detection does not prove that an object was absent."
        ),
    }
    _write_json(work / "run.json", run)
    _jsonl(work / "frames.jsonl", frame_records)
    _jsonl(work / "detections.jsonl", detections)
    shutil.rmtree(work / "records")
    frames_path.unlink(missing_ok=True)
    config_path.unlink(missing_ok=True)
    detector_provenance_path.unlink(missing_ok=True)
    segmenter_provenance_path.unlink(missing_ok=True)
    work.replace(resolved)
    return LocalizationSummary(
        output=resolved,
        frame_count=len(frame_records),
        detection_count=len(detections),
        device=device_used,
    )
