"""Learned temporal retrieval artifacts for the Charades video slice.

The implementation deliberately keeps the first learned system explicit:
windows are sampled deterministically, CLIP produces frame/text features, and a
small temporal head learns one vector per window.  The annotation retriever in
``charades.py`` remains the safe fallback until learned artifacts are present.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from visual_memory_lab.charades import load_manifest, search_windows
from visual_memory_lab.encoder import MODEL_ID, MODEL_REVISION, resolve_device
from visual_memory_lab.temporal import ThreeHeadTemporalModel, TemporalWindowEncoder, symmetric_contrastive_loss, three_head_loss

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _record_action_names(record: dict[str, object]) -> set[str]:
    return {
        str(action.get("name", "")).strip()
        for action in record.get("actions", [])
        if isinstance(action, dict) and str(action.get("name", "")).strip()
    }


class VideoActionResolver:
    """Resolve natural-language questions to exact recording action labels."""

    def __init__(self, *, model: str, cache_dir: Path, client: object | None = None) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self._client = client

    def resolve(self, question: str, action_names: list[str], candidate_names: list[str] | None = None) -> dict[str, object]:
        labels = sorted(set(action_names))
        if not labels:
            return {"matched_action_names": [], "unsupported": True, "reason": "recording has no annotated actions", "cached": False}
        prompt = (
            "Map the user question to zero or more exact action names from the supplied list. "
            "Return JSON only with matched_action_names, unsupported, and reason. Never invent "
            "labels. Return unsupported=true when no supplied label answers the question.\n\n"
            f"Question: {question.strip()}\nCLIP shortlist: {json.dumps(candidate_names or labels, ensure_ascii=False)}\n"
            f"Available action names: {json.dumps(labels, ensure_ascii=False)}"
        )
        digest = hashlib.sha256((self.model + "\n" + prompt).encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"action-resolve-{digest}.json"
        if cache_path.is_file():
            return {**json.loads(cache_path.read_text(encoding="utf-8")), "cached": True}
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        response = self._client.responses.create(
            model=self.model,
            input=prompt,
            text={"format": {"type": "json_object"}},
            store=False,
        )
        parsed = json.loads(str(getattr(response, "output_text", "")).strip())
        matched = [str(value) for value in parsed.get("matched_action_names", []) if str(value) in labels]
        result = {
            "matched_action_names": sorted(set(matched)),
            "unsupported": not bool(matched),
            "reason": str(parsed.get("reason", "")),
            "model": str(getattr(response, "model", self.model)),
            "cached": False,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result


def _ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output path is not empty: {path.resolve()}")
    path.mkdir(parents=True, exist_ok=True)


def sample_window_timestamps(
    start_s: float, end_s: float, frames_per_window: int = 16
) -> list[float]:
    """Return evenly spaced centre timestamps for one temporal window."""

    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")
    if frames_per_window < 1:
        raise ValueError("frames_per_window must be positive")
    duration = end_s - start_s
    return [
        round(start_s + (index + 0.5) * duration / frames_per_window, 6)
        for index in range(frames_per_window)
    ]


def window_text(window: dict[str, object]) -> str:
    """Create an auditable, window-level text target from Charades metadata."""

    actions = [
        str(item.get("name", "")).strip().lower()
        for item in window.get("actions", [])
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    ]
    objects = [str(item).strip().lower() for item in window.get("objects", []) if str(item).strip()]
    if actions:
        return "A person is " + " and ".join(actions) + "."
    description = str(window.get("description", "")).strip()
    if description:
        return description
    if objects:
        return "A video showing " + ", ".join(objects) + "."
    return "A video showing an office activity."


def build_frame_manifest(
    windows_manifest: Path,
    output: Path,
    *,
    frames_per_window: int = 16,
) -> dict[str, object]:
    """Write deterministic frame timestamps without duplicating MP4 files."""

    _ensure_empty(output)
    records: list[dict[str, object]] = []
    for window in load_manifest(windows_manifest):
        start_s = float(window["start_s"])
        end_s = float(window["end_s"])
        records.append(
            {
                "window_id": window["window_id"],
                "video_id": window["video_id"],
                "video_path": window["video_path"],
                "split": window["split"],
                "start_s": start_s,
                "end_s": end_s,
                "timestamps_s": sample_window_timestamps(start_s, end_s, frames_per_window),
                "text": window_text(window),
                "actions": window.get("actions", []),
                "objects": window.get("objects", []),
                "description": window.get("description", ""),
            }
        )
    manifest = output / "frames.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "window_count": len(records),
        "frames_per_window": frames_per_window,
        "manifest": str(manifest.resolve()),
        "sampling": "window-centres-evenly-spaced",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_frame_manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class LearnedVideoModel(nn.Module):
    """CLIP plus a trainable temporal head.

    CLIP is frozen by default. ``finetune_vision_blocks`` unfreezes only the
    final vision transformer blocks, making the expensive choice explicit.
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        finetune_vision_blocks: int = 0,
        temporal_layers: int = 2,
    ) -> None:
        super().__init__()
        from transformers import AutoProcessor, CLIPModel

        self.device = resolve_device(device)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.clip = CLIPModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.embedding_dim = int(self.clip.config.projection_dim)
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
        if finetune_vision_blocks:
            layers = self.clip.vision_model.encoder.layers
            if finetune_vision_blocks > len(layers):
                raise ValueError("finetune_vision_blocks exceeds the CLIP vision depth")
            for layer in layers[-finetune_vision_blocks:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        self.temporal = TemporalWindowEncoder(
            self.embedding_dim,
            output_dim=self.embedding_dim,
            max_frames=32,
            layers=temporal_layers,
        )
        self.to(self.device)

    @staticmethod
    def _normalise(features: Tensor) -> Tensor:
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _feature_tensor(output: object) -> Tensor:
        if isinstance(output, torch.Tensor):
            return output
        pooled_output = getattr(output, "pooler_output", None)
        if isinstance(pooled_output, torch.Tensor):
            return pooled_output
        raise TypeError("CLIP feature extraction did not return a pooled tensor")

    def encode_text(self, texts: list[str]) -> Tensor:
        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        features = self._feature_tensor(self.clip.get_text_features(**inputs))
        return self._normalise(features)

    def encode_frames(self, frames: Tensor) -> Tensor:
        """Encode [batch, frames, channels, height, width] RGB tensors."""

        batch, count, channels, height, width = frames.shape
        flat = frames.reshape(batch * count, channels, height, width)
        features = self._feature_tensor(self.clip.get_image_features(pixel_values=flat))
        features = self._normalise(features).reshape(batch, count, -1)
        return self.temporal(features)


def _decode_window(window: dict[str, object], frames_per_window: int) -> Tensor:
    """Decode a window with PyAV and return CLIP-normalized RGB tensors."""

    import av

    start_s = float(window["start_s"])
    end_s = float(window["end_s"])
    decoded: list[Tensor] = []
    with av.open(str(window["video_path"])) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int(start_s / float(stream.time_base))), stream=stream, backward=True)
        for frame in container.decode(stream):
            timestamp = float(frame.time or 0.0)
            if timestamp < start_s:
                continue
            if timestamp > end_s:
                break
            array = frame.to_rgb().to_ndarray()
            decoded.append(torch.from_numpy(array).permute(2, 0, 1))
    if not decoded:
        raise ValueError(f"video window contains no decodable frames: {window['window_id']}")
    video = torch.stack(decoded)
    indices = torch.linspace(0, video.shape[0] - 1, frames_per_window).round().long()
    selected = video[indices].float() / 255.0
    processor_size = 224
    selected = torch.nn.functional.interpolate(selected, size=(processor_size, processor_size), mode="bilinear")
    return (selected - _CLIP_MEAN) / _CLIP_STD


def _decode_video_targets(video_path: str, targets: list[float]) -> dict[float, Tensor]:
    """Decode one video once and retain the frames nearest requested times."""

    import av

    wanted = sorted(set(float(value) for value in targets))
    if not wanted:
        return {}
    result: dict[float, Tensor] = {}
    target_index = 0
    previous_time: float | None = None
    previous_frame: object | None = None
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            current_time = float(frame.time or 0.0)
            while target_index < len(wanted) and current_time >= wanted[target_index]:
                target = wanted[target_index]
                if previous_frame is not None and previous_time is not None and abs(previous_time - target) <= abs(current_time - target):
                    chosen = torch.from_numpy(previous_frame.to_rgb().to_ndarray()).permute(2, 0, 1)
                else:
                    chosen = torch.from_numpy(frame.to_rgb().to_ndarray()).permute(2, 0, 1)
                result[target] = chosen
                target_index += 1
            if target_index >= len(wanted):
                break
            previous_time = current_time
            previous_frame = frame
    if len(result) != len(wanted):
        raise ValueError(f"video did not contain all requested timestamps: {video_path}")
    return result


def _prepare_frames(frames: list[Tensor]) -> Tensor:
    selected = torch.stack(frames).float() / 255.0
    selected = torch.nn.functional.interpolate(selected, size=(224, 224), mode="bilinear")
    return (selected - _CLIP_MEAN) / _CLIP_STD


def build_embedding_cache(
    frame_manifest: Path,
    output: Path,
    *,
    device: str = "auto",
    batch_size: int = 16,
    max_videos: int | None = None,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, object]:
    """Decode windows and cache frozen CLIP frame/text embeddings.

    Each video is written as an atomic chunk so interrupted runs can resume.
    Video decoding is parallelised conservatively while CLIP inference stays
    batched on the selected device.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if max_videos is not None and max_videos < 1:
        raise ValueError("max_videos must be positive when provided")
    if resume:
        output.mkdir(parents=True, exist_ok=True)
    else:
        _ensure_empty(output)
    chunks = output / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    model = LearnedVideoModel(device=device)
    model.eval()
    records = load_frame_manifest(frame_manifest)
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["video_path"]), []).append(record)
    grouped_items = list(grouped.items())[:max_videos]
    config = {
        "manifest": str(frame_manifest.resolve()),
        "manifest_record_count": len(records),
        "video_count": len(grouped_items),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "frames_per_window": len(records[0]["timestamps_s"]) if records else 0,
        "batch_size": batch_size,
    }
    config_path = output / "config.json"
    if resume and config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in config}
        if comparable != config:
            raise ValueError("resume configuration does not match the existing cache")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    started = time.perf_counter()
    inference_seconds = 0.0
    failures = 0

    def existing_chunk(index: int) -> bool:
        return (chunks / f"chunk-{index:05d}.npz").exists() and (chunks / f"chunk-{index:05d}.jsonl").exists()

    def decode(item: tuple[str, list[dict[str, object]]]) -> tuple[str, list[dict[str, object]], dict[float, Tensor] | None]:
        video_path, video_records = item
        try:
            targets = [float(timestamp) for record in video_records for timestamp in record["timestamps_s"]]
            return video_path, video_records, _decode_video_targets(video_path, targets)
        except (OSError, RuntimeError, ValueError):
            return video_path, video_records, None

    with torch.inference_mode():
        for group_start in range(0, len(grouped_items), workers):
            group = grouped_items[group_start : group_start + workers]
            pending = [
                (group_start + offset, item)
                for offset, item in enumerate(group)
                if not existing_chunk(group_start + offset)
            ]
            pending_items = [item for _, item in pending]
            with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending_items)))) as pool:
                decoded_items = list(pool.map(decode, pending_items)) if pending_items else []
            for (chunk_index, _), (video_path, video_records, decoded) in zip(pending, decoded_items):
                if decoded is None:
                    failures += 1
                    continue
                frame_vectors: list[np.ndarray] = []
                text_vectors: list[np.ndarray] = []
                usable: list[dict[str, object]] = []
                inference_started = time.perf_counter()
                for start in range(0, len(video_records), batch_size):
                    batch_records = video_records[start : start + batch_size]
                    batch_frames = [
                        _prepare_frames([decoded[float(timestamp)] for timestamp in record["timestamps_s"]])
                        for record in batch_records
                    ]
                    frames_tensor = torch.stack(batch_frames).to(model.device)
                    batch, count, channels, height, width = frames_tensor.shape
                    flat = frames_tensor.reshape(batch * count, channels, height, width)
                    autocast = (
                        torch.autocast(device_type="cuda", dtype=torch.float16)
                        if model.device.type == "cuda"
                        else nullcontext()
                    )
                    with autocast:
                        frame_features = model._feature_tensor(model.clip.get_image_features(pixel_values=flat))
                        text_features = model.encode_text([str(record["text"]) for record in batch_records])
                    frame_features = model._normalise(frame_features).float().cpu().numpy().reshape(batch, count, -1)
                    text_features = text_features.float().cpu().numpy()
                    frame_vectors.extend(item.astype(np.float32) for item in frame_features)
                    text_vectors.extend(item.astype(np.float32) for item in text_features)
                    usable.extend(batch_records)
                inference_seconds += time.perf_counter() - inference_started
                stem = chunks / f"chunk-{chunk_index:05d}"
                temp = chunks / f".chunk-{chunk_index:05d}.npz"
                np.savez_compressed(temp, frame_embeddings=np.stack(frame_vectors), text_embeddings=np.stack(text_vectors))
                temp.replace(stem.with_suffix(".npz"))
                stem.with_suffix(".jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in usable), encoding="utf-8")

    chunk_paths = sorted(chunks.glob("chunk-*.npz"))
    frame_vectors = [np.load(path)["frame_embeddings"] for path in chunk_paths]
    text_vectors = [np.load(path)["text_embeddings"] for path in chunk_paths]
    usable = []
    for path in chunk_paths:
        usable.extend(load_frame_manifest(path.with_suffix(".jsonl")))
    if not usable:
        raise ValueError("no video windows could be decoded")
    np.save(output / "frame_embeddings.npy", np.concatenate(frame_vectors, axis=0))
    np.save(output / "text_embeddings.npy", np.concatenate(text_vectors, axis=0))
    (output / "records.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in usable), encoding="utf-8")
    elapsed = time.perf_counter() - started
    summary = {
        "window_count": len(usable),
        "video_count": len(grouped_items),
        "failed_videos": failures,
        "frames_per_window": int(frame_vectors[0].shape[1]),
        "embedding_dim": int(frame_vectors[0].shape[2]),
        "device": str(model.device),
        "batch_size": batch_size,
        "workers": workers,
        "resumed": resume,
        "elapsed_seconds": round(elapsed, 3),
        "windows_per_second": round(len(usable) / elapsed, 3) if elapsed else 0.0,
        "inference_seconds": round(inference_seconds, 3),
        "output": str(output.resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_temporal_from_cache(
    cache: Path,
    output: Path,
    *,
    device: str = "auto",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    split: str = "train",
    action_weight: float = 1.0,
    boundary_weight: float = 2.0,
) -> dict[str, object]:
    """Train retrieval, action, and boundary heads against cached features."""

    _ensure_empty(output)
    resolved = resolve_device(device)
    frames = torch.from_numpy(np.load(cache / "frame_embeddings.npy")).float()
    texts = torch.from_numpy(np.load(cache / "text_embeddings.npy")).float()
    records = load_frame_manifest(cache / "records.jsonl")
    selected = torch.tensor(
        [index for index, record in enumerate(records) if str(record.get("split")) == split],
        dtype=torch.long,
    )
    if not len(selected):
        raise ValueError(f"cache contains no records for split {split!r}")
    frames = frames[selected]
    texts = texts[selected]
    if len(frames) < 2:
        raise ValueError("at least two windows are required for contrastive training")
    action_names = sorted({
        str(action.get("action_id"))
        for record in records
        for action in record.get("actions", [])
        if isinstance(action, dict) and action.get("action_id")
    })
    if not action_names:
        raise ValueError("cache records contain no action labels")
    action_index = {name: index for index, name in enumerate(action_names)}
    action_targets = torch.zeros((len(records), len(action_names)), dtype=torch.float32)
    boundary_targets = torch.zeros((len(records), 2), dtype=torch.float32)
    boundary_mask = torch.zeros(len(records), dtype=torch.bool)
    for row_index, record in enumerate(records):
        window_start = float(record.get("start_s", 0.0))
        window_end = float(record.get("end_s", 0.0))
        window_length = max(window_end - window_start, 1e-6)
        best_overlap = 0.0
        for action in record.get("actions", []):
            if not isinstance(action, dict):
                continue
            action_id = str(action.get("action_id", ""))
            if action_id in action_index:
                action_targets[row_index, action_index[action_id]] = 1.0
            action_start = float(action.get("start_s", 0.0))
            action_end = float(action.get("end_s", 0.0))
            overlap = max(0.0, min(window_end, action_end) - max(window_start, action_start))
            if overlap > best_overlap:
                best_overlap = overlap
                boundary_targets[row_index] = torch.tensor([
                    max(0.0, min(1.0, (action_start - window_start) / window_length)),
                    max(0.0, min(1.0, (action_end - window_start) / window_length)),
                ])
                boundary_mask[row_index] = True
    action_targets = action_targets[selected]
    boundary_targets = boundary_targets[selected]
    boundary_mask = boundary_mask[selected]
    model = ThreeHeadTemporalModel(
        frames.shape[-1],
        len(action_names),
        output_dim=texts.shape[-1],
        max_frames=frames.shape[1],
    ).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    generator = torch.Generator().manual_seed(42)
    for epoch in range(epochs):
        order = torch.randperm(len(frames), generator=generator)
        model.train()
        losses: list[float] = []
        loss_parts: dict[str, list[float]] = {"retrieval": [], "action": [], "boundary": []}
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if len(indices) < 2:
                continue
            outputs = model(frames[indices].to(resolved))
            text = texts[indices].to(resolved)
            loss, parts = three_head_loss(
                outputs,
                text,
                action_targets[indices].to(resolved),
                boundary_targets[indices].to(resolved),
                boundary_mask[indices].to(resolved),
                action_weight=action_weight,
                boundary_weight=boundary_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                loss_parts[key].append(float(value.detach().cpu()))
        epoch_summary = {"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else float("nan"), **{f"{key}_loss": float(np.mean(value)) if value else float("nan") for key, value in loss_parts.items()}}
        history.append(epoch_summary)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={epoch_summary['loss']:.4f} | "
            f"retrieval={epoch_summary['retrieval_loss']:.4f} | "
            f"action={epoch_summary['action_loss']:.4f} | "
            f"boundary={epoch_summary['boundary_loss']:.4f}",
            flush=True,
        )
    torch.save({"state_dict": model.state_dict(), "input_dim": frames.shape[-1], "output_dim": texts.shape[-1], "max_frames": int(frames.shape[1]), "action_names": action_names, "model_type": "three_head_temporal_v1"}, output / "temporal_multitask.pt")
    (output / "history.jsonl").write_text("".join(json.dumps(item) + "\n" for item in history), encoding="utf-8")
    summary = {"epochs": epochs, "window_count": len(frames), "split": split, "device": str(resolved), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()), "action_count": len(action_names), "boundary_labeled_windows": int(boundary_mask.sum()), "action_weight": action_weight, "boundary_weight": boundary_weight, "model_type": "three_head_temporal_v1", "output": str(output.resolve())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_index_from_cache(
    cache: Path,
    checkpoint: Path | None,
    output: Path,
    *,
    device: str = "auto",
    split: str = "train",
) -> dict[str, object]:
    """Encode cached frame sequences with a trained head and write a flat index."""

    _ensure_empty(output)
    frames = torch.from_numpy(np.load(cache / "frame_embeddings.npy")).float()
    records = [json.loads(line) for line in (cache / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [
        index for index, record in enumerate(records)
        if split == "all" or str(record.get("split")) == split
    ]
    if not selected:
        raise ValueError(f"cache contains no records for split {split!r}")
    frames = frames[selected]
    records = [records[index] for index in selected]
    payload = None
    head: torch.nn.Module
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=resolve_device(device), weights_only=True)
    if payload is not None and payload.get("model_type") == "three_head_temporal_v1":
        head = ThreeHeadTemporalModel(frames.shape[-1], len(payload["action_names"]), output_dim=frames.shape[-1], max_frames=frames.shape[1]).to(resolve_device(device))
    else:
        head = TemporalWindowEncoder(frames.shape[-1], output_dim=frames.shape[-1], max_frames=frames.shape[1]).to(resolve_device(device))
    if payload is not None:
        head.load_state_dict(payload["state_dict"])
    head.eval()
    with torch.inference_mode():
        if isinstance(head, ThreeHeadTemporalModel):
            outputs = head(frames.to(next(head.parameters()).device))
            vectors = outputs["retrieval"].cpu().numpy()
            action_scores = torch.sigmoid(outputs["action_logits"]).cpu().numpy()
            boundary = torch.sigmoid(outputs["boundary_logits"]).cpu().numpy()
        else:
            vectors = head(frames.to(head.position.device)).cpu().numpy()
            action_scores = None
            boundary = None
    if action_scores is not None and payload is not None:
        for index, record in enumerate(records):
            start = float(record.get("start_s", 0.0))
            end = float(record.get("end_s", 0.0))
            length = end - start
            record["predicted_action_scores"] = {name: round(float(action_scores[index, column]), 5) for column, name in enumerate(payload["action_names"])}
            record["predicted_start_s"] = round(start + float(boundary[index, 0]) * length, 4)
            record["predicted_end_s"] = round(start + float(boundary[index, 1]) * length, 4)
    summary = write_index(vectors, records, output)
    summary["split"] = split
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def diversify_video_results(
    results: list[dict[str, object]],
    *,
    top_k: int,
    max_per_video: int = 2,
    overlap_threshold: float = 0.25,
) -> list[dict[str, object]]:
    """Keep high-scoring, temporally distinct evidence windows.

    Retrieval deliberately over-fetches candidates because neighbouring windows
    can describe the same event. This pass is presentation logic: it does not
    alter the stored scores or vectors.
    """

    if top_k < 1:
        return []
    selected: list[dict[str, object]] = []
    per_video: dict[str, int] = {}
    for candidate in results:
        video_id = str(candidate.get("video_id", ""))
        count = per_video.get(video_id, 0)
        interval = (float(candidate.get("start_s", 0.0)), float(candidate.get("end_s", 0.0)))
        same_video_overlap = any(
            str(item.get("video_id", "")) == video_id
            and interval_iou(
                interval,
                (float(item.get("start_s", 0.0)), float(item.get("end_s", 0.0))),
            ) >= overlap_threshold
            for item in selected
        )
        if count >= max_per_video or same_video_overlap:
            continue
        selected.append(candidate)
        per_video[video_id] = count + 1
        if len(selected) >= top_k:
            return selected
    return selected[:top_k]


def group_video_events(
    results: list[dict[str, object]],
    *,
    top_k: int = 3,
    overlap_threshold: float = 0.25,
) -> list[dict[str, object]]:
    """Turn ranked retrieval windows into distinct, reviewable events.

    Charades windows overlap on purpose, so several high-scoring windows often
    describe the same short action.  We keep the best window as the anchor and
    absorb nearby windows from the same recording.  The source window IDs stay
    attached so the result remains auditable.
    """
    if top_k < 1:
        return []
    groups: list[dict[str, object]] = []
    for candidate in results:
        video_id = str(candidate.get("video_id", ""))
        context_interval = (float(candidate.get("start_s", 0.0)), float(candidate.get("end_s", 0.0)))
        predicted_start = float(candidate.get("predicted_start_s", context_interval[0]))
        predicted_end = float(candidate.get("predicted_end_s", context_interval[1]))
        interval = (predicted_start, predicted_end) if predicted_end > predicted_start else context_interval
        action_ids = {
            str(action.get("action_id"))
            for action in candidate.get("actions", [])
            if isinstance(action, dict) and action.get("action_id")
        }
        match: dict[str, object] | None = None
        for group in groups:
            if str(group.get("video_id", "")) != video_id:
                continue
            group_interval = (float(group["start_s"]), float(group["end_s"]))
            group_action_ids = set(group.get("_action_ids", []))
            if interval_iou(interval, group_interval) >= overlap_threshold or action_ids & group_action_ids:
                match = group
                break
        if match is None:
            match = dict(candidate)
            match["start_s"], match["end_s"] = interval
            match["context_start_s"], match["context_end_s"] = context_interval
            match["evidence_window_ids"] = [str(candidate.get("window_id", ""))]
            match["_action_ids"] = sorted(action_ids)
            groups.append(match)
            continue

        match["start_s"] = min(float(match["start_s"]), interval[0])
        match["end_s"] = max(float(match["end_s"]), interval[1])
        match["context_start_s"] = min(float(match.get("context_start_s", match["start_s"])), context_interval[0])
        match["context_end_s"] = max(float(match.get("context_end_s", match["end_s"])), context_interval[1])
        match["score"] = max(float(match.get("score", 0.0)), float(candidate.get("score", 0.0)))
        evidence_ids = list(match.get("evidence_window_ids", []))
        window_id = str(candidate.get("window_id", ""))
        if window_id and window_id not in evidence_ids:
            evidence_ids.append(window_id)
        match["evidence_window_ids"] = evidence_ids
        match["_action_ids"] = sorted(set(match.get("_action_ids", [])) | action_ids)
        actions = list(match.get("actions", []))
        known_actions = {(str(a.get("action_id")), str(a.get("name"))) for a in actions if isinstance(a, dict)}
        for action in candidate.get("actions", []):
            if not isinstance(action, dict):
                continue
            key = (str(action.get("action_id")), str(action.get("name")))
            if key not in known_actions:
                actions.append(action)
                known_actions.add(key)
        match["actions"] = actions
        match["objects"] = sorted(set(match.get("objects", [])) | set(candidate.get("objects", [])))

    for index, group in enumerate(groups, start=1):
        group["event_id"] = f"{group.get('video_id', 'video')}:{float(group['start_s']):.3f}-{float(group['end_s']):.3f}"
        group.pop("_action_ids", None)
    groups.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return groups[:top_k]


@dataclass(frozen=True)
class LearnedVideoIndex:
    vectors: np.ndarray
    records: list[dict[str, object]]
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION

    def save(self, output: Path) -> None:
        _ensure_empty(output)
        np.save(output / "vectors.npy", self.vectors.astype(np.float32))
        (output / "records.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in self.records),
            encoding="utf-8",
        )
        (output / "metadata.json").write_text(
            json.dumps({"model_id": self.model_id, "model_revision": self.model_revision, "count": len(self.records)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "LearnedVideoIndex":
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (path / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        vectors = np.load(path / "vectors.npy")
        return cls(vectors=vectors, records=records, model_id=str(metadata["model_id"]), model_revision=str(metadata["model_revision"]))

    def search(
        self,
        query: np.ndarray,
        top_k: int = 8,
        *,
        diversify: bool = True,
        video_id: str | None = None,
        action_names: set[str] | None = None,
    ) -> list[dict[str, object]]:
        query = query / max(float(np.linalg.norm(query)), 1e-8)
        scores = self.vectors @ query
        eligible = [
            index for index, record in enumerate(self.records)
            if (video_id is None or str(record.get("video_id", "")) == video_id)
            and (action_names is None or _record_action_names(record) & action_names)
        ]
        if not eligible:
            return []
        candidate_k = min(len(eligible), max(top_k * 4, top_k)) if diversify else min(len(eligible), top_k)
        ranked = sorted(eligible, key=lambda index: float(scores[index]), reverse=True)
        order = ranked[:candidate_k]
        candidates = [{**self.records[int(index)], "score": round(float(scores[index]), 4), "retrieval_mode": "learned_temporal_clip"} for index in order]
        return diversify_video_results(candidates, top_k=top_k) if diversify else candidates


class LearnedVideoRetriever:
    """Lazy text-to-window adapter used by the API."""

    def __init__(self, index: LearnedVideoIndex, *, device: str = "auto", action_resolver: VideoActionResolver | None = None) -> None:
        from visual_memory_lab.encoder import ClipEncoder

        self.index = index
        self.encoder = ClipEncoder(device=device)
        self._action_vectors: dict[str, np.ndarray] = {}
        self.action_resolver = action_resolver

    @staticmethod
    def _tokens(value: str) -> set[str]:
        stop = {"a", "an", "the", "person", "someone", "some", "when", "did", "what", "is", "are", "was", "were", "to", "of", "on", "in", "with", "from", "then"}
        tokens = {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in stop and len(token) > 2}
        expanded = set(tokens)
        for token in tokens:
            if token.endswith("ing") and len(token) > 5:
                stem = token[:-3]
                expanded.add(stem)
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    expanded.add(stem[:-1])
            elif token.endswith("ed") and len(token) > 4:
                expanded.add(token[:-2])
        return expanded

    def _action_matches(self, query: str, records: list[dict[str, object]]) -> tuple[set[str], dict[str, object]]:
        """Infer supported action labels from the selected recording.

        The vocabulary comes from the recording itself; no fixed list of actions
        is embedded in the application. Exact token overlap handles clear cases,
        while CLIP similarity supplies a small synonym bridge (for example,
        ``pick up a bag`` versus ``taking a bag from somewhere``).
        """
        labels = sorted({name for record in records for name in _record_action_names(record) if name})
        if not labels:
            return set(), {"status": "unsupported", "matched_actions": [], "reason": "recording has no annotated actions"}
        query_tokens = self._tokens(query)
        missing = [label for label in labels if label not in self._action_vectors]
        if missing:
            vectors = self.encoder.encode_texts([f"A person is {label.lower()}." for label in missing])
            self._action_vectors.update({label: vector for label, vector in zip(missing, vectors, strict=True)})
        query_vector = self.encoder.encode_texts([query])[0]
        semantic_scores = sorted(
            ((float(np.dot(query_vector, vector) / max(np.linalg.norm(query_vector) * np.linalg.norm(vector), 1e-8)), label) for label, vector in self._action_vectors.items()),
            reverse=True,
        )
        shortlist = [label for _, label in semantic_scores[:5]]
        if self.action_resolver is not None:
            try:
                resolved = self.action_resolver.resolve(query, labels, shortlist)
                matched = set(str(value) for value in resolved.get("matched_action_names", [])) & set(labels)
                if matched:
                    return matched, {"status": "supported", "matched_actions": sorted(matched), "reason": str(resolved.get("reason", "")), "resolver": "llm", "cached": bool(resolved.get("cached", False))}
                return set(), {"status": "unsupported", "matched_actions": [], "reason": str(resolved.get("reason", "no recorded action supports this question")), "resolver": "llm", "cached": bool(resolved.get("cached", False))}
            except Exception:
                # The deterministic path keeps local development and tests usable
                # when the external resolver is unavailable.
                pass
        lexical: list[tuple[float, str]] = []
        for label in labels:
            label_tokens = self._tokens(label)
            overlap = len(query_tokens & label_tokens)
            coverage = overlap / max(1, len(query_tokens))
            lexical.append((coverage, label))
        lexical.sort(reverse=True)
        best_coverage, best_label = lexical[0]
        # A shared object/action term is enough when it is specific to one label.
        # Otherwise use the learned CLIP text space to bridge synonyms.
        if best_coverage > 0:
            tied = {label for coverage, label in lexical if coverage == best_coverage and coverage > 0}
            if len(tied) == 1 or best_coverage >= 0.5:
                return tied, {"status": "supported", "matched_actions": sorted(tied), "reason": "recording action vocabulary matched the question"}
        scores = sorted(
            ((float(np.dot(query_vector, vector) / max(np.linalg.norm(query_vector) * np.linalg.norm(vector), 1e-8)), label) for label, vector in self._action_vectors.items()),
            reverse=True,
        )
        if scores and scores[0][0] >= 0.25 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.015):
            return {scores[0][1]}, {"status": "supported", "matched_actions": [scores[0][1]], "reason": "question matched a recording action in the learned text space"}
        return set(), {"status": "unsupported", "matched_actions": [], "reason": "no recorded action supports this question"}

    def search(self, query: str, *, top_k: int = 8, video_id: str | None = None) -> list[dict[str, object]]:
        results, _ = self.search_with_metadata(query, top_k=top_k, video_id=video_id)
        return results

    def search_with_metadata(self, query: str, *, top_k: int = 8, video_id: str | None = None) -> tuple[list[dict[str, object]], dict[str, object]]:
        query_vector = self.encoder.encode_texts([query])[0]
        eligible = [record for record in self.index.records if video_id is None or str(record.get("video_id", "")) == video_id]
        action_names, metadata = self._action_matches(query, eligible)
        if metadata["status"] != "supported":
            return [], metadata
        results = self.index.search(query_vector, top_k=top_k, video_id=video_id, action_names=action_names)
        if not results:
            # The learned index is intentionally built from the training split.
            # A user can still select a held-out recording in the catalogue, so
            # do not turn a known annotation into a misleading "unsupported"
            # answer merely because that recording has no learned vector yet.
            # Return the matching catalogue windows as a transparent fallback;
            # the UI still labels the result as annotation-backed evidence.
            fallback = search_windows(
                [record for record in eligible if _record_action_names(record) & action_names],
                query,
                top_k=top_k,
            )
            if fallback:
                for item in fallback:
                    item["retrieval_mode"] = "annotation_fallback_unindexed_recording"
                return fallback, {
                    "status": "supported",
                    "matched_actions": sorted(action_names),
                    "reason": "learned vectors are unavailable for this recording; matched annotated windows are shown",
                    "fallback": True,
                }
            metadata = {"status": "unsupported", "matched_actions": sorted(action_names), "reason": "matched action has no retrievable evidence window"}
        return results, metadata


def write_index(vectors: np.ndarray, records: list[dict[str, object]], output: Path) -> dict[str, object]:
    if vectors.ndim != 2 or len(vectors) != len(records):
        raise ValueError("vectors and records must have matching two-dimensional shapes")
    index = LearnedVideoIndex(vectors=vectors, records=records)
    index.save(output)
    return {"count": len(records), "embedding_dim": int(vectors.shape[1]), "output": str(output.resolve())}


def interval_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Intersection-over-union for two one-dimensional time intervals."""

    first_start, first_end = first
    second_start, second_end = second
    intersection = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


def boundary_error(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Mean absolute start/end error in seconds."""

    return (abs(first[0] - second[0]) + abs(first[1] - second[1])) / 2.0


def normalized_boundary_error(
    predicted: tuple[float, float], target: tuple[float, float], duration_s: float
) -> float:
    """Mean start/end error expressed as a fraction of recording duration."""

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    return boundary_error(predicted, target) / duration_s


def _action_ids(record: dict[str, object]) -> set[str]:
    return {
        str(action.get("action_id"))
        for action in record.get("actions", [])
        if isinstance(action, dict) and action.get("action_id")
    }


def _action_intervals(record: dict[str, object]) -> list[tuple[float, float]]:
    return [
        (float(action.get("start_s", 0.0)), float(action.get("end_s", 0.0)))
        for action in record.get("actions", [])
        if isinstance(action, dict) and float(action.get("end_s", 0.0)) > float(action.get("start_s", 0.0))
    ]


def evaluate_index(
    index_path: Path,
    test_manifest: Path,
    output: Path,
    *,
    device: str = "auto",
    top_ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, object]:
    """Evaluate learned text-to-window retrieval against official action spans."""

    _ensure_empty(output)
    index = LearnedVideoIndex.load(index_path)
    queries = [record for record in load_frame_manifest(test_manifest) if str(record.get("split")) == "test"]
    if not queries:
        raise ValueError("test manifest contains no test windows")
    from visual_memory_lab.encoder import ClipEncoder

    encoder = ClipEncoder(device=device)
    query_texts = [str(record["text"]) for record in queries]
    query_batches = [
        encoder.encode_texts(query_texts[start : start + 32])
        for start in range(0, len(query_texts), 32)
    ]
    query_vectors = np.concatenate(query_batches, axis=0)
    max_k = max(top_ks)
    hits = {k: 0 for k in top_ks}
    ious: list[float] = []
    errors: list[float] = []
    normalized_errors: list[float] = []
    duplicate_rates: list[float] = []
    failures: list[dict[str, object]] = []
    for query, vector in zip(queries, query_vectors, strict=True):
        results = index.search(vector, top_k=max_k)
        relevant: list[tuple[int, dict[str, object], float, float]] = []
        query_ids = _action_ids(query)
        query_intervals = _action_intervals(query)
        for rank, result in enumerate(results, start=1):
            candidate_ids = _action_ids(result)
            shared = query_ids & candidate_ids
            candidate_interval = (float(result["start_s"]), float(result["end_s"]))
            overlap = max((interval_iou(candidate_interval, interval) for interval in query_intervals), default=0.0)
            if shared and overlap > 0:
                error = min(boundary_error(candidate_interval, interval) for interval in query_intervals)
                duration = float(query.get("duration_s", query.get("end_s", 0.0)))
                relevant.append((rank, result, overlap, error))
        for k in top_ks:
            if any(rank <= k for rank, _, _, _ in relevant):
                hits[k] += 1
        if relevant:
            _, best, best_iou, best_error = relevant[0]
            ious.append(best_iou)
            errors.append(best_error)
            target_duration = float(query.get("duration_s", query.get("end_s", 0.0)))
            if target_duration > 0:
                normalized_errors.append(best_error / target_duration)
            spans = [(float(item["start_s"]), float(item["end_s"])) for item in results]
            duplicate_rates.append(sum(interval_iou(spans[0], span) > 0 for span in spans[1:]) / max(1, len(spans) - 1))
        else:
            failures.append({"query_window_id": query["window_id"], "query_text": query["text"], "top_results": [item["window_id"] for item in results]})
    count = len(queries)
    metrics = {
        "query_count": count,
        "recall_at_k": {str(k): hits[k] / count for k in top_ks},
        "temporal_iou_mean": float(np.mean(ious)) if ious else 0.0,
        "temporal_iou_median": float(np.median(ious)) if ious else 0.0,
        "boundary_error_mean_s": float(np.mean(errors)) if errors else None,
        "boundary_error_median_s": float(np.median(errors)) if errors else None,
        "normalized_boundary_error_mean": float(np.mean(normalized_errors)) if normalized_errors else None,
        "normalized_boundary_error_median": float(np.median(normalized_errors)) if normalized_errors else None,
        "duplicate_rate_mean": float(np.mean(duplicate_rates)) if duplicate_rates else 0.0,
        "miss_count": len(failures),
        "index_count": len(index.records),
        "model_id": index.model_id,
        "model_revision": index.model_revision,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "failures.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in failures), encoding="utf-8")
    report = ["# Learned Charades video retrieval evaluation", "", f"Queries: {count}", f"Indexed windows: {len(index.records)}", "", "## Metrics", ""]
    report.extend([f"- Recall@{k}: {metrics['recall_at_k'][str(k)]:.4f}" for k in top_ks])
    report.extend([
        f"- Mean temporal IoU: {metrics['temporal_iou_mean']:.4f}",
        f"- Median temporal IoU: {metrics['temporal_iou_median']:.4f}",
        f"- Mean boundary error: {metrics['boundary_error_mean_s']:.3f} s" if metrics["boundary_error_mean_s"] is not None else "- Mean boundary error: unavailable",
        f"- Mean normalized boundary error: {metrics['normalized_boundary_error_mean']:.4f}" if metrics["normalized_boundary_error_mean"] is not None else "- Mean normalized boundary error: unavailable",
        f"- Duplicate rate: {metrics['duplicate_rate_mean']:.4f}",
        f"- Misses: {metrics['miss_count']}",
        "",
        "A miss means no top-k result shared an official action label and overlapped its annotated time interval. A miss is not proof that the action never occurred.",
    ])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics
