"""Integration tests for the Phase 1 trajectory artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from visual_memory_lab.cli import main
from visual_memory_lab.trajectory import GenerationConfig, generate_trajectories


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _png_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.png"))
    }


def test_generation_writes_complete_contract(tmp_path: Path) -> None:
    output = tmp_path / "run"
    summary = generate_trajectories(
        GenerationConfig(output=output, episodes=2, base_seed=42, max_steps=100)
    )

    assert summary.episode_count == 2
    assert summary.observation_count == 76

    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    records = _jsonl(output / "observations.jsonl")
    assert manifest["observation_count"] == len(records) == 76
    assert len({record["observation_id"] for record in records}) == len(records)
    assert len({record["image_path"] for record in records}) == len(records)
    assert records[0]["step"] == records[0]["sim_time"] == 0
    assert records[0]["action"] is None
    assert records[1]["action"] in {"left", "right", "forward"}

    for record in records:
        frame_path = output / str(record["image_path"])
        assert frame_path.is_file()
        with Image.open(frame_path) as image:
            assert image.mode == "RGB"
            assert image.size == (56, 56)

    for episode in manifest["episodes"]:
        overview_path = output / str(episode["overview_path"])
        scene_path = output / str(episode["scene_path"])
        assert scene_path.is_file()
        with Image.open(overview_path) as image:
            assert image.mode == "RGB"
            assert image.size == (120, 72)


def test_generation_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = dict(episodes=2, base_seed=42, max_steps=100)
    generate_trajectories(GenerationConfig(output=first, **config))
    generate_trajectories(GenerationConfig(output=second, **config))

    assert (first / "run.json").read_bytes() == (second / "run.json").read_bytes()
    assert (first / "observations.jsonl").read_bytes() == (
        second / "observations.jsonl"
    ).read_bytes()
    assert _png_hashes(first) == _png_hashes(second)


def test_every_object_is_observed_from_the_required_poses(tmp_path: Path) -> None:
    output = tmp_path / "run"
    generate_trajectories(
        GenerationConfig(output=output, episodes=10, base_seed=42, max_steps=100)
    )
    records = _jsonl(output / "observations.jsonl")

    visible_poses: dict[tuple[str, str], set[tuple[int, int]]] = {}
    hidden_scene_object_seen = False
    for record in records:
        episode_id = str(record["episode_id"])
        pose = tuple(record["agent_position"])
        visible_ids = {str(obj["object_id"]) for obj in record["visible_objects"]}
        hidden_scene_object_seen |= len(visible_ids) < 3
        for obj in record["visible_objects"]:
            key = (episode_id, str(obj["object_id"]))
            visible_poses.setdefault(key, set()).add(pose)

    for episode_index in range(10):
        episode_id = f"episode-{episode_index:03d}"
        assert len(visible_poses[(episode_id, "red-ball-a")]) >= 2
        assert len(visible_poses[(episode_id, "red-ball-b")]) >= 2
        assert visible_poses[(episode_id, "blue-box")]
    assert hidden_scene_object_seen


def test_generation_refuses_occupied_output(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "keep.txt").write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        generate_trajectories(GenerationConfig(output=output))
    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not replace"


def test_cli_generates_a_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "cli-run"
    exit_code = main(
        [
            "generate",
            "--episodes",
            "1",
            "--seed",
            "7",
            "--max-steps",
            "100",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert (output / "run.json").is_file()
    assert "Generated 38 observations across 1 episodes" in capsys.readouterr().out
