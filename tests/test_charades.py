from __future__ import annotations

import json
from pathlib import Path

import torch

from visual_memory_lab.charades import build_temporal_windows, parse_actions, search_windows
from visual_memory_lab.temporal import ThreeHeadTemporalModel, TemporalWindowEncoder, symmetric_contrastive_loss, three_head_loss


def test_parse_actions_uses_human_readable_class_names() -> None:
    actions = parse_actions("c006 1.00 2.50;c008 3.00 4.00", {"c006": "Closing a door", "c008": "Opening a door"})
    assert actions[0].name == "Closing a door"
    assert actions[1].start_s == 3.0


def test_build_windows_preserves_overlapping_actions(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "video_id": "ABC12", "split": "train", "video_path": "ABC12.mp4",
        "length_s": 5.0, "objects": ["door"], "description": "A person opens a door",
        "actions": [{"action_id": "c008", "name": "Opening a door", "start_s": 1.0, "end_s": 2.0}],
    }) + "\n", encoding="utf-8")
    output = tmp_path / "windows"
    summary = build_temporal_windows(manifest, output, window_s=2.0, stride_s=1.0)
    assert summary["window_count"] == 4
    rows = [json.loads(line) for line in (output / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["actions"][0]["name"] == "Opening a door"


def test_search_windows_is_transparent_and_deterministic() -> None:
    windows = [{
        "window_id": "ABC12:0-4", "video_id": "ABC12", "actions": [{"name": "Opening a door"}],
        "objects": ["door"], "description": "A person opens a door.",
    }, {
        "window_id": "XYZ99:0-4", "video_id": "XYZ99", "actions": [{"name": "Sitting at a table"}],
        "objects": ["table"], "description": "A person sits.",
    }]
    result = search_windows(windows, "When did the person open the door?", top_k=2)
    assert result[0]["video_id"] == "ABC12"
    assert len(result) == 1
    assert result[0]["retrieval_mode"] == "annotation_lexical_baseline"

    sitting = [{
        "window_id": "SIT01:0-4", "video_id": "SIT01",
        "actions": [{"name": "Sitting at a table"}], "objects": ["table"],
        "description": "A person sits at a table.",
    }]
    assert search_windows(sitting, "When did the person sit down?")[0]["video_id"] == "SIT01"


def test_temporal_head_and_contrastive_loss_are_trainable() -> None:
    model = TemporalWindowEncoder(8, hidden_dim=16, output_dim=8, max_frames=4, heads=4)
    frames = torch.randn(3, 4, 8, requires_grad=True)
    output = model(frames)
    loss = symmetric_contrastive_loss(output, torch.randn(3, 8))
    loss.backward()
    assert output.shape == (3, 8)
    assert model.input_projection.weight.grad is not None


def test_three_head_temporal_model_produces_all_training_outputs() -> None:
    model = ThreeHeadTemporalModel(8, 3, output_dim=8, max_frames=4)
    frames = torch.randn(4, 4, 8)
    outputs = model(frames)
    loss, parts = three_head_loss(
        outputs,
        torch.randn(4, 8),
        torch.zeros(4, 3),
        torch.full((4, 2), 0.5),
        torch.ones(4, dtype=torch.bool),
    )
    loss.backward()
    assert outputs["retrieval"].shape == (4, 8)
    assert outputs["action_logits"].shape == (4, 3)
    assert outputs["boundary_logits"].shape == (4, 2)
    assert set(parts) == {"retrieval", "action", "boundary", "frame_refinement"}
    assert outputs["frame_refinement_logits"].shape == (4, 4, 3)
