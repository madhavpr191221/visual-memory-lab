"""Auditable records and fusion utilities for multimodal video memory.

This module deliberately does not assume that every dataset contains every
sensor.  A record declares which RGB, audio, depth, and pose artifacts exist;
downstream models receive an explicit availability mask instead of fabricated
zeros that could be mistaken for real sensor data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn


MODALITIES = ("rgb", "audio", "depth", "pose")


@dataclass(frozen=True)
class TemporalAnnotation:
    """One timestamped event supplied by a dataset or reviewer."""

    start_s: float
    end_s: float
    label: str
    objects: tuple[str, ...] = ()
    source: str = "dataset"

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("annotation interval must satisfy 0 <= start_s < end_s")
        if not self.label.strip():
            raise ValueError("annotation label must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "label": self.label,
            "objects": list(self.objects),
            "source": self.source,
        }


@dataclass(frozen=True)
class MultimodalRecord:
    """A normalized recording entry used by preparation and evaluation."""

    video_id: str
    duration_s: float
    paths: dict[str, str] = field(default_factory=dict)
    annotations: tuple[TemporalAnnotation, ...] = ()
    split: str = "unspecified"
    summary: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        unknown = set(self.paths) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
        for annotation in self.annotations:
            if annotation.end_s > self.duration_s + 1e-6:
                raise ValueError(f"annotation exceeds duration for {self.video_id}")

    @property
    def modality_mask(self) -> dict[str, bool]:
        return {name: bool(self.paths.get(name)) for name in MODALITIES}

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "duration_s": self.duration_s,
            "paths": dict(self.paths),
            "annotations": [item.to_dict() for item in self.annotations],
            "split": self.split,
            "summary": self.summary,
            "metadata": self.metadata,
            "modality_mask": self.modality_mask,
        }


def record_from_dict(value: dict[str, object]) -> MultimodalRecord:
    annotations = tuple(
        TemporalAnnotation(
            start_s=float(item["start_s"]),
            end_s=float(item["end_s"]),
            label=str(item["label"]),
            objects=tuple(str(obj) for obj in item.get("objects", [])),
            source=str(item.get("source", "dataset")),
        )
        for item in value.get("annotations", [])
        if isinstance(item, dict)
    )
    paths = {str(key): str(path) for key, path in dict(value.get("paths", {})).items() if path}
    return MultimodalRecord(
        video_id=str(value["video_id"]),
        duration_s=float(value["duration_s"]),
        paths=paths,
        annotations=annotations,
        split=str(value.get("split", "unspecified")),
        summary=str(value.get("summary", "")),
        metadata=dict(value.get("metadata", {})),
    )


def load_records(path: Path) -> list[MultimodalRecord]:
    """Load JSONL records and validate every interval and modality name."""

    records: list[MultimodalRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(record_from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid multimodal record on line {line_number}: {error}") from error
    return records


def write_records(records: Iterable[MultimodalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def audit_records(records: Iterable[MultimodalRecord]) -> dict[str, object]:
    records = list(records)
    modality_counts = {
        modality: sum(record.modality_mask[modality] for record in records)
        for modality in MODALITIES
    }
    return {
        "record_count": len(records),
        "split_counts": {
            split: sum(record.split == split for record in records)
            for split in sorted({record.split for record in records})
        },
        "modality_counts": modality_counts,
        "all_modalities_count": sum(all(record.modality_mask.values()) for record in records),
        "annotation_count": sum(len(record.annotations) for record in records),
        "total_duration_s": round(sum(record.duration_s for record in records), 3),
        "label_count": len({annotation.label for record in records for annotation in record.annotations}),
    }


class MissingModalityFusion(nn.Module):
    """Project available modality vectors into one temporal representation.

    Inputs are ``{modality: [batch, time, dimension]}`` plus an optional
    boolean availability mask ``[batch, time, modalities]``.  Missing values
    contribute nothing and are tracked by a learned mask projection.
    """

    def __init__(self, input_dims: dict[str, int], hidden_dim: int) -> None:
        super().__init__()
        if not input_dims or hidden_dim < 1:
            raise ValueError("input_dims and hidden_dim must be non-empty/positive")
        unknown = set(input_dims) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
        self.modalities = tuple(input_dims)
        self.projections = nn.ModuleDict({name: nn.Linear(dim, hidden_dim) for name, dim in input_dims.items()})
        self.mask_projection = nn.Linear(len(self.modalities), hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: dict[str, Tensor], availability: Tensor | None = None) -> Tensor:
        if not features:
            raise ValueError("at least one modality feature tensor is required")
        first = next(iter(features.values()))
        if first.ndim != 3:
            raise ValueError("modality features must have shape [batch, time, dimension]")
        batch, time, _ = first.shape
        if availability is None:
            availability = torch.ones(batch, time, len(self.modalities), device=first.device, dtype=first.dtype)
        if availability.shape != (batch, time, len(self.modalities)):
            raise ValueError("availability must have shape [batch, time, modality_count]")
        total = self.mask_projection(availability.to(first.dtype))
        for index, modality in enumerate(self.modalities):
            value = features.get(modality)
            if value is None:
                continue
            if value.shape[:2] != (batch, time):
                raise ValueError("all modality tensors must share batch and time dimensions")
            mask = availability[..., index:index + 1].to(value.dtype)
            total = total + self.projections[modality](value) * mask
        return self.norm(total)
