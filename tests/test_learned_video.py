"""Fast tests for learned video manifests, losses, and exact retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from visual_memory_lab.learned_video import (
    LearnedVideoIndex,
    boundary_error,
    build_frame_manifest,
    diversify_video_results,
    interval_iou,
    sample_window_timestamps,
    window_text,
)


def test_sampling_is_even_and_inside_window() -> None:
    timestamps = sample_window_timestamps(2.0, 6.0, 8)
    assert len(timestamps) == 8
    assert timestamps[0] == 2.25
    assert timestamps[-1] == 5.75
    assert all(2.0 < value < 6.0 for value in timestamps)


def test_window_text_uses_actions_then_description() -> None:
    assert window_text({"actions": [{"name": "Sitting on a chair"}]}) == "A person is sitting on a chair."
    assert window_text({"actions": [], "description": "A person enters."}) == "A person enters."


def test_frame_manifest_preserves_order_and_text(tmp_path: Path) -> None:
    windows = tmp_path / "windows.jsonl"
    windows.write_text(json.dumps({
        "window_id": "v:0-4", "video_id": "v", "video_path": "v.mp4",
        "split": "train", "start_s": 0.0, "end_s": 4.0,
        "actions": [{"name": "Opening a door"}], "objects": ["door"], "description": "",
    }) + "\n", encoding="utf-8")
    summary = build_frame_manifest(windows, tmp_path / "frames")
    assert summary["frames_per_window"] == 8
    record = json.loads((tmp_path / "frames" / "frames.jsonl").read_text(encoding="utf-8"))
    assert record["timestamps_s"][0] == 0.25
    assert record["text"] == "A person is opening a door."


def test_exact_index_round_trip_and_search(tmp_path: Path) -> None:
    records = [{"window_id": "a"}, {"window_id": "b"}]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = LearnedVideoIndex(vectors=vectors, records=records)
    index.save(tmp_path / "index")
    loaded = LearnedVideoIndex.load(tmp_path / "index")
    assert loaded.search(np.asarray([0.9, 0.1], dtype=np.float32), top_k=1)[0]["window_id"] == "a"


def test_temporal_metrics_are_interpretable() -> None:
    assert interval_iou((0.0, 4.0), (2.0, 6.0)) == 1 / 3
    assert boundary_error((0.0, 4.0), (1.0, 5.0)) == 1.0


def test_diversification_removes_overlapping_neighbours() -> None:
    candidates = [
        {"video_id": "a", "start_s": 0.0, "end_s": 4.0, "score": 0.99},
        {"video_id": "a", "start_s": 2.0, "end_s": 6.0, "score": 0.98},
        {"video_id": "b", "start_s": 0.0, "end_s": 4.0, "score": 0.97},
    ]
    result = diversify_video_results(candidates, top_k=2)
    assert [(item["video_id"], item["start_s"]) for item in result] == [("a", 0.0), ("b", 0.0)]
