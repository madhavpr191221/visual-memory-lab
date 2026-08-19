from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from visual_memory_lab.multimodal import (
    MissingModalityFusion,
    MultimodalRecord,
    TemporalAnnotation,
    audit_records,
    load_records,
    write_records,
)


def test_multimodal_record_round_trip_and_audit(tmp_path: Path) -> None:
    record = MultimodalRecord(
        video_id="inspection-01",
        duration_s=12.0,
        paths={"rgb": "rgb.mp4", "audio": "audio.wav", "depth": "depth/", "pose": "pose.json"},
        annotations=(TemporalAnnotation(1.0, 4.0, "open cabinet", ("cabinet",)),),
        split="test",
    )
    path = tmp_path / "records.jsonl"
    write_records([record], path)
    loaded = load_records(path)
    assert loaded == [record]
    summary = audit_records(loaded)
    assert summary["all_modalities_count"] == 1
    assert summary["modality_counts"]["audio"] == 1
    assert summary["annotation_count"] == 1


def test_record_rejects_unknown_modality_and_bad_interval() -> None:
    with pytest.raises(ValueError):
        MultimodalRecord("x", 2.0, paths={"thermal": "x"})
    with pytest.raises(ValueError):
        TemporalAnnotation(2.0, 1.0, "bad")


def test_missing_modality_fusion_respects_availability() -> None:
    torch.manual_seed(0)
    fusion = MissingModalityFusion({"rgb": 4, "audio": 3, "depth": 2}, hidden_dim=5)
    rgb = torch.randn(2, 3, 4)
    audio = torch.randn(2, 3, 3)
    depth = torch.randn(2, 3, 2)
    all_available = torch.ones(2, 3, 3)
    no_audio = all_available.clone()
    no_audio[..., 1] = 0
    with_audio = fusion({"rgb": rgb, "audio": audio, "depth": depth}, all_available)
    without_audio = fusion({"rgb": rgb, "audio": audio, "depth": depth}, no_audio)
    assert with_audio.shape == (2, 3, 5)
    assert not torch.allclose(with_audio, without_audio)


def test_load_records_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"video_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_records(path)
