"""Tests for designated-reference-traversal retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from visual_memory_lab.cli import build_parser
from visual_memory_lab.memory import MemoryIndex
from visual_memory_lab.traversal_evaluation import (
    evaluate_cross_traversal_retrieval,
    write_traversal_evaluation,
)


def _record(observation_id: str, sequence: str | None, x: float) -> dict[str, object]:
    pose = np.eye(4)
    pose[0, 3] = x
    record: dict[str, object] = {
        "observation_id": observation_id,
        "step": 0,
        "image_path": "unused.png",
        "camera_pose": {"convention": "camera_to_world", "matrix": pose.tolist()},
    }
    if sequence is not None:
        record["sequence_id"] = sequence
    return record


def _index(
    tmp_path: Path,
    name: str,
    records: list[dict[str, object]],
    embeddings: list[list[float]],
    *,
    revision: str = "1",
) -> MemoryIndex:
    return MemoryIndex(
        root=tmp_path / name,
        manifest={"model": {"id": "test/clip", "revision": revision}},
        records=records,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        source=tmp_path,
        image_root=tmp_path,
    )


def _fixture_indexes(tmp_path: Path) -> tuple[MemoryIndex, MemoryIndex]:
    memory = _index(
        tmp_path,
        "memory",
        [
            _record("a-near", "target-a", 0.1),
            _record("a-far", "target-a", 2.0),
            _record("b-near", "target-b", 0.2),
            _record("b-far", "target-b", 3.0),
        ],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
    )
    queries = _index(
        tmp_path,
        "queries",
        [
            _record("query-near", "source-a", 0.0),
            _record("query-uncovered", "source-b", 10.0),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    return memory, queries


def test_cross_traversal_metrics_filter_targets_and_preserve_denominators(
    tmp_path: Path,
) -> None:
    memory, queries = _fixture_indexes(tmp_path)

    metrics, rows = evaluate_cross_traversal_retrieval(memory, queries, seed=7)

    assert metrics["query_count"] == 2
    assert metrics["query_target_count"] == 4
    assert metrics["pair_count"] == 4
    assert metrics["source_sequences"] == ["source-a", "source-b"]
    assert metrics["target_sequences"] == ["target-a", "target-b"]
    assert metrics["strict"]["coverage"] == 0.5
    assert metrics["strict"]["hit_at_1"]["eligible_query_targets"] == 2
    assert metrics["strict"]["hit_at_1"]["covered_rate"] == 0.5
    assert metrics["strict"]["hit_at_1"]["all_query_target_rate"] == 0.25

    target_a = next(
        row
        for row in rows
        if row["query_id"] == "query-near" and row["target_sequence_id"] == "target-a"
    )
    target_b = next(
        row
        for row in rows
        if row["query_id"] == "query-near" and row["target_sequence_id"] == "target-b"
    )
    assert target_a["retrievals"][0]["observation_id"] == "a-near"
    assert target_a["strict_hit_at_1"] is True
    assert target_b["retrievals"][0]["observation_id"] == "b-far"
    assert target_b["strict_hit_at_1"] is False
    assert target_b["strict_hit_at_5"] is True
    assert all(
        retrieval["observation_id"].startswith("b-")
        for retrieval in target_b["retrievals"]
    )


def test_cross_traversal_random_baseline_is_deterministic(tmp_path: Path) -> None:
    memory, queries = _fixture_indexes(tmp_path)

    first, first_rows = evaluate_cross_traversal_retrieval(memory, queries, seed=11)
    second, second_rows = evaluate_cross_traversal_retrieval(memory, queries, seed=11)

    assert first == second
    assert first_rows == second_rows


def test_cross_traversal_validation_rejects_invalid_protocol_inputs(tmp_path: Path) -> None:
    memory, queries = _fixture_indexes(tmp_path)

    with pytest.raises(ValueError, match="non-negative"):
        evaluate_cross_traversal_retrieval(memory, queries, seed=-1)
    with pytest.raises(ValueError, match="out of range"):
        evaluate_cross_traversal_retrieval(memory, queries, query_indices=[99])

    missing_sequence = _index(
        tmp_path,
        "missing-sequence",
        [_record("missing", None, 0.0)],
        [[1.0, 0.0]],
    )
    with pytest.raises(ValueError, match="sequence_id"):
        evaluate_cross_traversal_retrieval(memory, missing_sequence)

    invalid_pose_record = _record("invalid-pose", "source", 0.0)
    invalid_pose_record["camera_pose"] = {
        "convention": "camera_to_world",
        "matrix": [[1.0]],
    }
    invalid_pose = _index(
        tmp_path,
        "invalid-pose",
        [invalid_pose_record],
        [[1.0, 0.0]],
    )
    with pytest.raises(ValueError, match="finite 4x4 pose matrix"):
        evaluate_cross_traversal_retrieval(memory, invalid_pose)

    wrong_revision = _index(
        tmp_path,
        "wrong-revision",
        [_record("other", "source", 0.0)],
        [[1.0, 0.0]],
        revision="2",
    )
    with pytest.raises(ValueError, match="same encoder revision"):
        evaluate_cross_traversal_retrieval(memory, wrong_revision)

    overlapping = _index(
        tmp_path,
        "overlapping",
        [_record("a-near", "source", 0.0)],
        [[1.0, 0.0]],
    )
    with pytest.raises(ValueError, match="disjoint observation identities"):
        evaluate_cross_traversal_retrieval(memory, overlapping)


def test_writer_and_cli_contract(tmp_path: Path) -> None:
    memory, queries = _fixture_indexes(tmp_path)
    output = tmp_path / "evaluation"

    metrics = write_traversal_evaluation(
        memory=memory, queries=queries, output=output, seed=5
    )

    assert metrics["protocol"] == "designated-reference-traversal"
    assert json.loads((output / "metrics.json").read_text())["chronology"] == "not_available"
    assert len((output / "per_query_target.jsonl").read_text().splitlines()) == 4
    with pytest.raises(FileExistsError, match="not empty"):
        write_traversal_evaluation(memory=memory, queries=queries, output=output)

    args = build_parser().parse_args(
        [
            "evaluate-traversal-memory",
            "--memory-index",
            "memory",
            "--query-index",
            "queries",
            "--output",
            "evaluation",
            "--seed",
            "9",
        ]
    )
    assert args.command == "evaluate-traversal-memory"
    assert args.seed == 9
