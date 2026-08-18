"""Fast tests for learned video manifests, losses, and exact retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from visual_memory_lab.learned_video import (
    LearnedVideoIndex,
    VideoActionResolver,
    boundary_error,
    build_frame_manifest,
    diversify_video_results,
    group_video_events,
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
    assert summary["frames_per_window"] == 16
    record = json.loads((tmp_path / "frames" / "frames.jsonl").read_text(encoding="utf-8"))
    assert record["timestamps_s"][0] == 0.125
    assert record["text"] == "A person is opening a door."


def test_exact_index_round_trip_and_search(tmp_path: Path) -> None:
    records = [{"window_id": "a"}, {"window_id": "b"}]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = LearnedVideoIndex(vectors=vectors, records=records)
    index.save(tmp_path / "index")
    loaded = LearnedVideoIndex.load(tmp_path / "index")
    assert loaded.search(np.asarray([0.9, 0.1], dtype=np.float32), top_k=1)[0]["window_id"] == "a"


def test_index_search_can_be_scoped_to_one_recording() -> None:
    records = [{"window_id": "a", "video_id": "one"}, {"window_id": "b", "video_id": "two"}]
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    index = LearnedVideoIndex(vectors=vectors, records=records)
    result = index.search(np.asarray([1.0, 0.0], dtype=np.float32), video_id="two")
    assert [item["window_id"] for item in result] == ["b"]


def test_index_search_can_require_an_action_label() -> None:
    records = [
        {"window_id": "door", "video_id": "v", "actions": [{"name": "Opening a door"}]},
        {"window_id": "chair", "video_id": "v", "actions": [{"name": "Sitting in a chair"}]},
    ]
    index = LearnedVideoIndex(vectors=np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32), records=records)
    result = index.search(np.asarray([1.0, 0.0], dtype=np.float32), action_names={"Opening a door"})
    assert [item["window_id"] for item in result] == ["door"]


def test_action_resolver_discards_labels_not_in_vocabulary(tmp_path: Path) -> None:
    class Response:
        output_text = '{"matched_action_names": ["Opening a cabinet", "Opening a door"], "reason": "test"}'
        model = "test-model"

    class Responses:
        def create(self, **_: object) -> Response:
            return Response()

    class Client:
        responses = Responses()

    resolver = VideoActionResolver(model="test-model", cache_dir=tmp_path, client=Client())
    result = resolver.resolve("open the cabinet", ["Opening a door"])
    assert result["matched_action_names"] == ["Opening a door"]


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


def test_group_video_events_merges_overlap_and_preserves_evidence() -> None:
    candidates = [
        {"window_id": "v-0", "video_id": "v", "start_s": 0.0, "end_s": 4.0, "score": 0.99,
         "actions": [{"action_id": "a1", "name": "Opening a door", "start_s": 0.0, "end_s": 3.0}], "objects": ["door"]},
        {"window_id": "v-1", "video_id": "v", "start_s": 2.0, "end_s": 6.0, "score": 0.98,
         "actions": [{"action_id": "a1", "name": "Opening a door", "start_s": 0.0, "end_s": 3.0}], "objects": ["hallway"]},
        {"window_id": "v-2", "video_id": "v", "start_s": 10.0, "end_s": 14.0, "score": 0.90,
         "actions": [{"action_id": "a2", "name": "Sitting down", "start_s": 10.0, "end_s": 14.0}], "objects": ["chair"]},
    ]
    result = group_video_events(candidates, top_k=3)
    assert len(result) == 2
    assert result[0]["evidence_window_ids"] == ["v-0", "v-1"]
    assert result[0]["start_s"] == 0.0 and result[0]["end_s"] == 6.0
    assert set(result[0]["objects"]) == {"door", "hallway"}
    assert result[1]["event_id"].startswith("v:10.000-")
