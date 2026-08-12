"""Cautious cross-visit association for frozen ETH object detections."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from visual_memory_lab.encoder import ClipEncoder

PAIR_PROMPT_VERSION = "phase6.1.3-association-v1"
CLASSES = ("chair", "waste_bin", "box")


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _crop(frame: dict[str, object], detection: dict[str, object], root: Path) -> Image.Image:
    image = Image.open(root / str(frame["image_path"])).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in detection["box_xyxy"]]
    pad_x, pad_y = (x2 - x1) * 0.08, (y2 - y1) * 0.08
    box = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)), min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y)))
    return image.crop(box)


def _quality(row: dict[str, object]) -> float:
    points = min(1.0, math.log1p(max(0, int(row.get("point_count", 0)))) / math.log1p(500))
    return float(row["score"]) * float(row["mask_score"]) * (0.5 + 0.5 * points)


def _pair_score(a: dict[str, object], b: dict[str, object], appearance: float) -> dict[str, float]:
    a_area = max(float(a.get("mask_area_fraction", 0.001)), 0.001)
    b_area = max(float(b.get("mask_area_fraction", 0.001)), 0.001)
    shape = math.exp(-abs(math.log(a_area / b_area)))
    evidence = min(1.0, (_quality(a) + _quality(b)) / 2.0)
    ac = a.get("centroid_world_m")
    bc = b.get("centroid_world_m")
    if isinstance(ac, list) and isinstance(bc, list) and all(v is not None for v in ac + bc):
        distance = float(np.linalg.norm(np.asarray(ac, dtype=float) - np.asarray(bc, dtype=float)))
        position = math.exp(-distance / 2.0)
    else:
        distance, position = None, 0.5
    score = 0.55 * max(0.0, min(1.0, (appearance + 1.0) / 2.0)) + 0.15 * shape + 0.15 * evidence + 0.15 * position
    return {"appearance_similarity": float(appearance), "shape_score": float(shape), "evidence_score": float(evidence), "position_score": float(position), "association_score": float(score), "centroid_distance_m": distance}


def _fallback_embeddings(crops: list[Image.Image]) -> np.ndarray:
    """Offline-safe colour/texture descriptor when CLIP weights are unavailable."""
    vectors: list[np.ndarray] = []
    for crop in crops:
        image = np.asarray(crop.resize((32, 32)).convert("RGB"), dtype=np.float32) / 255.0
        hist = np.concatenate([np.histogram(image[:, :, channel], bins=16, range=(0, 1), density=True)[0] for channel in range(3)])
        gray = image.mean(axis=2)
        descriptor = np.concatenate([hist, gray.mean(axis=(0, 1), keepdims=False).reshape(1), gray.std(axis=(0, 1), keepdims=False).reshape(1)])
        vectors.append(descriptor.astype(np.float32))
    result = np.asarray(vectors, dtype=np.float32)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-8)


def associate_eth_objects(*, localization: Path, rgbd_evidence: Path, output: Path, device: str = "auto", top_per_group: int = 200) -> dict[str, object]:
    load_dotenv()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output path is not empty: {output.resolve()}")
    output.mkdir(parents=True, exist_ok=True)
    root = localization.resolve()
    frames = {str(item["frame_id"]): item for item in _jsonl(root / "frames.jsonl")}
    detections = _jsonl(root / "detections.jsonl")
    evidence = {str(item["detection_id"]): item for item in _jsonl(rgbd_evidence.resolve() / "evidence.jsonl")}
    crops: list[Image.Image] = []
    usable: list[dict[str, object]] = []
    for detection in detections:
        frame = frames.get(str(detection["frame_id"]))
        if frame is None or str(detection["canonical_class"]) not in CLASSES:
            continue
        row = {**detection, **evidence.get(str(detection["detection_id"]), {})}
        crops.append(_crop(frame, detection, root))
        usable.append(row)
    appearance_model = "clip-vit-b32"
    try:
        encoder = ClipEncoder(device=device)
        embeddings = np.concatenate([encoder.encode_pil_images(crops[index:index + 64]) for index in range(0, len(crops), 64)], axis=0)
        resolved_device = str(encoder.device)
    except Exception:
        appearance_model = "offline-rgb-histogram-fallback"
        embeddings = _fallback_embeddings(crops)
        resolved_device = "cpu-fallback"
    np.save(output / "crop_embeddings.npy", embeddings)
    index_by_id = {str(row["detection_id"]): index for index, row in enumerate(usable)}
    pairs: list[dict[str, object]] = []
    for earlier in range(3):
        for later in range(earlier + 1, 4):
            for object_class in CLASSES:
                old = [row for row in usable if int(row["observation"]) == earlier and row["canonical_class"] == object_class]
                new = [row for row in usable if int(row["observation"]) == later and row["canonical_class"] == object_class]
                group: list[dict[str, object]] = []
                for a in old:
                    for b in new:
                        appearance = float(np.dot(embeddings[index_by_id[str(a["detection_id"])]], embeddings[index_by_id[str(b["detection_id"])] ]))
                        scores = _pair_score(a, b, appearance)
                        group.append({"pair_id": f"{a['detection_id']}__{b['detection_id']}", "earlier_observation": earlier, "later_observation": later, "object_class": object_class, "earlier_detection_id": a["detection_id"], "later_detection_id": b["detection_id"], **scores, "association_status": "likely_same" if scores["association_score"] >= 0.70 else "possible_match" if scores["association_score"] >= 0.50 else "uncertain", "movement_status": "possible_movement" if scores["association_score"] >= 0.70 and scores["centroid_distance_m"] is not None and scores["centroid_distance_m"] > 0.5 else "not_established", "claim_boundary": "Candidate association only; no persistent identity or definitive movement claim."})
                pairs.extend(sorted(group, key=lambda item: float(item["association_score"]), reverse=True)[:top_per_group])
    run = {"phase": "6.1.3", "prompt_version": PAIR_PROMPT_VERSION, "pair_count": len(pairs), "detection_count": len(usable), "top_per_group": top_per_group, "classes": list(CLASSES), "claim_boundary": "Matches are ranked candidates. Possible movement is not definitive.", "device": resolved_device, "appearance_model": appearance_model}
    (output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "associations.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in pairs), encoding="utf-8")
    return run
